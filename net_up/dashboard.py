"""
dashboard.py
------------
Real-Time Security Monitoring Dashboard built with Streamlit.

Run with:
    streamlit run dashboard.py

Features:
  • LIVE EVENT STREAM — real-time packet detection with sub-second updates
  • KPI cards — total packets, anomalies, zero-days, anomaly rate
  • Protocol breakdown pie chart
  • Anomaly trend over time (line chart)
  • Per-source-IP threat leaderboard
  • Model performance metrics loaded from training artefacts
"""

import os
import json
import time
import datetime
import subprocess
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGuard — AI Security Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Theming  (cyber / dark-terminal aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #c9d1e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1221;
    border-right: 1px solid #1e2d4a;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #0f1a30 0%, #162040 100%);
    border: 1px solid #1e3a6e;
    border-radius: 8px;
    padding: 18px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
}
.kpi-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.4rem;
    font-weight: bold;
    color: #00d4ff;
    line-height: 1;
}
.kpi-label {
    font-size: 0.8rem;
    color: #6a82a8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}
.kpi-card.danger .kpi-value  { color: #ff4d6d; }
.kpi-card.warning .kpi-value { color: #f59e0b; }
.kpi-card.success .kpi-value { color: #22c55e; }

/* Log table */
.stDataFrame { font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; }

/* Section headers */
h2, h3 { color: #00d4ff !important; letter-spacing: 1px; }

/* Alert boxes */
.alert-anomaly {
    background: rgba(255,77,109,0.12);
    border-left: 3px solid #ff4d6d;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    border-radius: 0 4px 4px 0;
}
.alert-zero-day {
    background: rgba(245,158,11,0.15);
    border-left: 3px solid #f59e0b;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    border-radius: 0 4px 4px 0;
}
.alert-normal {
    background: rgba(34,197,94,0.08);
    border-left: 3px solid #22c55e;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    border-radius: 0 4px 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE   = "detect_log.jsonl"
MODEL_DIR  = "models"

@st.cache_data(ttl=2)          # refresh every 2 seconds
def load_log(path: str = LOG_FILE) -> pd.DataFrame:
    """Read detection log into a DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp", "src_ip", "dst_ip", "protocol",
            "label", "confidence", "zero_day"
        ])
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=60)
def load_model_metrics():
    """Load saved training metrics from the models/ folder."""
    metrics = {}
    for name in ["xgboost", "autoencoder"]:
        path = os.path.join(MODEL_DIR, f"{name}_metrics.pkl")
        if os.path.exists(path):
            metrics[name] = joblib.load(path)
    return metrics


@st.cache_data(ttl=1)
def load_performance_metrics():
    """Load detector performance metrics."""
    if not os.path.exists("perf_metrics.json"):
        return None
    try:
        with open("perf_metrics.json") as f:
            return json.load(f)
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ NetGuard")
    st.markdown("**AI-Driven Network Anomaly Detection**")
    st.markdown("---")
    refresh_rate = st.slider("Auto-refresh (seconds)", 0.5, 30.0, 1.5)
    st.caption("⚡ Lower = faster live updates (best: 0.5-2.0s)")
    st.markdown("---")
    st.markdown("### 📂 System Status")

    log_exists   = os.path.exists(LOG_FILE)
    model_exists = os.path.exists(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    ae_exists    = os.path.exists(os.path.join(MODEL_DIR, "autoencoder.keras"))

    st.markdown(f"{'🟢' if log_exists else '🔴'} Detection Log")
    st.markdown(f"{'🟢' if model_exists else '🔴'} XGBoost Model")
    st.markdown(f"{'🟢' if ae_exists else '🟡'} Autoencoder (optional)")
    st.markdown("---")
    st.markdown("### 🔧 Quick Commands")
    st.code("python preprocessing.py\npython train.py\npython detect.py\npython inject.py", language="bash")
    st.markdown("---")

    if st.button("🗑️ Clear Log"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        st.cache_data.clear()
        st.success("Log cleared.")

    st.markdown("---")
    st.markdown("### 🚀 Live Attack Injection")
    st.markdown("Test detection by injecting simulated attacks")
    st.info("""
    **Run in a separate terminal:**
    ```bash
    sudo python inject.py
    ```
    Then click 'Detect Now' to refresh the dashboard.
    """)
    
    inject_method = st.radio("Choose injection method:", ["Manual (Terminal)", "Automated (requires sudo config)"], horizontal=True)
    
    inject_cols = st.columns(2)
    with inject_cols[0]:
        if st.button("🔴 Start Injection", key="inject_start", use_container_width=True):
            if inject_method == "Manual (Terminal)":
                st.info("📌 Run this in your terminal:\n```\nsudo python inject.py\n```")
            else:
                st.info("⏳ Launching inject.py with sudo... (needs passwordless sudo)")
                try:
                    result = subprocess.run(
                        ["sudo", "-n", "python", "inject.py"],  # -n: non-interactive
                        capture_output=True,
                        text=True,
                        timeout=15
                    )
                    if result.returncode == 0:
                        st.success("✅ Attack injection completed!")
                        st.dataframe(pd.DataFrame({
                            "Attack Type": ["TCP SYN Flood", "UDP Flood", "ICMP Flood", "Port Scan", "Fragmented"],
                            "Packets": [50, 50, 50, 50, 10],
                            "Status": ["✅ Injected"] * 5
                        }), hide_index=True, use_container_width=True)
                        st.cache_data.clear()
                    else:
                        st.error(f"❌ Needs passwordless sudo. Use Manual method instead.")
                except subprocess.TimeoutExpired:
                    st.warning("⚠️ Injection timed out")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with inject_cols[1]:
        if st.button("🟢 Detect Now", key="detect_refresh", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Detection log refreshed!")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ NetGuard — Real-Time Security Dashboard")
st.markdown(f"*Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Local Time)*")
st.markdown("---")

# ── Live Stream Placeholder (real-time updates) ────────────────────────────────
live_placeholder = st.empty()

df = load_log()

# ── KPI Row ───────────────────────────────────────────────────────────────────
total     = len(df)
anomalies = int((df["label"] == "anomaly").sum()) if total else 0
zero_days = int(df["zero_day"].sum()) if total and "zero_day" in df.columns else 0
rate      = round(anomalies / total * 100, 1) if total else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-label">Total Packets</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card danger">
        <div class="kpi-value">{anomalies:,}</div>
        <div class="kpi-label">Anomalies Detected</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card warning">
        <div class="kpi-value">{zero_days:,}</div>
        <div class="kpi-label">Zero-Day Candidates</div>
    </div>""", unsafe_allow_html=True)
with c4:
    cls = "danger" if rate > 30 else ("warning" if rate > 10 else "success")
    st.markdown(f"""<div class="kpi-card {cls}">
        <div class="kpi-value">{rate}%</div>
        <div class="kpi-label">Anomaly Rate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LIVE EVENT STREAM (updates every packet) ──────────────────────────────────
st.markdown("### 🔴 LIVE EVENT STREAM (Real-Time)")
live_feed_container = st.container()

with live_feed_container:
    col_live = st.columns(1)[0]
    with col_live:
        # Auto-scrolling live feed showing last 5 events
        if total > 0:
            latest = df.tail(5).sort_values("timestamp", ascending=False)
            for _, row in latest.iterrows():
                ts   = row["timestamp"].strftime("%H:%M:%S")
                conf = row.get("confidence", 0)
                proto = row["protocol"].upper()
                src_ip = row["src_ip"]
                dst_ip = row["dst_ip"]
                label = row["label"]
                zd = row.get("zero_day", False)
                
                if label == "anomaly":
                    if zd:
                        icon = "⚠️ ZERO-DAY"
                        color = "#f59e0b"
                    else:
                        icon = "🚨 ATTACK"
                        color = "#ff4d6d"
                else:
                    icon = "✅ SAFE"
                    color = "#22c55e"
                
                st.markdown(f"""
                <div style="background: rgba({color.lstrip('#')}, 0.1); border-left: 3px solid {color}; padding: 12px; margin: 4px 0; border-radius: 4px; font-family: 'Share Tech Mono', monospace; font-size: 0.85rem;">
                    <b>[{ts}] {icon}</b>  {src_ip} → {dst_ip} | {proto} | conf={conf:.3f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("⏳ Waiting for packets...")

st.markdown("---")
if total > 0:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("### Protocol Distribution")
        proto_counts = df["protocol"].value_counts().reset_index()
        proto_counts.columns = ["Protocol", "Count"]
        st.bar_chart(proto_counts.set_index("Protocol"), height=250)

    with col_r:
        st.markdown("### Anomaly Trend Over Time")
        if total > 0:
            try:
                df_time = df.copy()
                df_time["minute"] = df_time["timestamp"].dt.floor("min")
                
                # Group by minute and label, count occurrences
                trend = df_time.groupby(["minute", "label"]).size().unstack(fill_value=0).reset_index()
                trend.columns.name = None
                trend = trend.rename(columns={"minute": "Time"})
                
                # Ensure both columns exist
                if "anomaly" not in trend.columns:
                    trend["anomaly"] = 0
                if "normal" not in trend.columns:
                    trend["normal"] = 0
                
                # Display chart
                if len(trend) > 0:
                    st.line_chart(
                        trend.set_index("Time")[["normal", "anomaly"]],
                        height=250,
                        use_container_width=True
                    )
                else:
                    st.info("⏳ Waiting for time-series data...")
            except Exception as e:
                st.warning(f"⚠️ Chart error: {e}")
        else:
            st.info("⏳ Waiting for packets to plot trend...")

    st.markdown("---")

    # ── Threat Leaderboard ────────────────────────────────────────────────────
    col_tb, col_log = st.columns([1, 2])

    with col_tb:
        st.markdown("### 🎯 Top Threat Sources")
        threat_df = (
            df[df["label"] == "anomaly"]
            .groupby("src_ip")
            .size()
            .reset_index(name="Anomalies")
            .sort_values("Anomalies", ascending=False)
            .head(10)
        )
        if len(threat_df):
            st.dataframe(threat_df, use_container_width=True, hide_index=True)
        else:
            st.info("No threats detected yet.")

    with col_log:
        st.markdown("### 📋 Recent Events (last 50)")
        recent = df.tail(50).sort_values("timestamp", ascending=False)
        # Colour code the label column
        st.dataframe(
            recent[["timestamp", "src_ip", "dst_ip", "protocol", "label", "confidence", "zero_day"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Live Alert Feed ───────────────────────────────────────────────────────
    st.markdown("### 🔴 Live Alert Feed (last 20 alerts)")
    alerts = df[df["label"] == "anomaly"].tail(20).sort_values("timestamp", ascending=False)
    for _, row in alerts.iterrows():
        ts   = row["timestamp"].strftime("%H:%M:%S")
        zd   = " ⚠ ZERO-DAY?" if row.get("zero_day") else ""
        cls  = "alert-zero-day" if row.get("zero_day") else "alert-anomaly"
        st.markdown(
            f'<div class="{cls}">[{ts}]{zd}  {row["src_ip"]} → {row["dst_ip"]} | '
            f'{row["protocol"].upper()} | conf={row["confidence"]:.3f}</div>',
            unsafe_allow_html=True
        )

else:
    st.info("⏳ No detection data yet. Start `detect.py` to begin monitoring.")

# ── Model Performance Metrics ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 Model Performance (Training Evaluation)")
metrics = load_model_metrics()

if metrics:
    cols = st.columns(len(metrics))
    for i, (model_name, m) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"**{model_name.capitalize()}**")
            st.metric("Accuracy",  f"{m.get('accuracy', 0):.4f}")
            st.metric("Precision", f"{m.get('precision', 0):.4f}")
            st.metric("Recall",    f"{m.get('recall', 0):.4f}")
            st.metric("F1-Score",  f"{m.get('f1', 0):.4f}")
else:
    st.info("No model metrics found. Run `train.py` first.")

# ── Detector Performance (Real-Time) ─────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚡ Detector Performance (Real-Time)")
perf = load_performance_metrics()

if perf:
    pf1, pf2, pf3, pf4 = st.columns(4)
    with pf1:
        st.metric("Packets/sec", f"{perf.get('packets_per_second', 0):.1f}")
    with pf2:
        st.metric("Avg Latency", f"{perf.get('avg_detection_latency_ms', 0):.2f}ms")
    with pf3:
        st.metric("CPU Usage", f"{perf.get('cpu_percent', 0):.1f}%")
    with pf4:
        st.metric("Memory", f"{perf.get('memory_mb', 0):.1f}MB")
    
    st.caption(f"📊 Running for {perf.get('elapsed_seconds', 0):.0f}s | Peak latency: {perf.get('max_detection_latency_ms', 0):.2f}ms")
else:
    st.info("⏳ Performance metrics will appear when detector is running...")

# ─────────────────────────────────────────────────────────────────────────────
# Email Alerts Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📧 Email Alerts Configuration"):
    st.markdown("**Enable email notifications for attacks and zero-days**")
    
    col_config1, col_config2 = st.columns([1, 2])
    with col_config1:
        if st.button("📋 Setup Guide", use_container_width=True):
            st.info("""
            **Steps to enable email alerts:**
            
            1. **Gmail users:**
               - Go to: https://myaccount.google.com/apppasswords
               - Generate an App Password (16 characters)
            
            2. **Edit `alerts.py`:**
               - Set `EMAIL_ENABLED = True`
               - Add your email and app password
               - Set recipient email
            
            3. **Test:**
               ```bash
               python -c "from alerts import send_email_alert; send_email_alert('192.168.1.1', '192.168.1.6', 'Test', 0.95, 'tcp')"
               ```
            """)
    
    with col_config2:
        status_file = "alerts.py"
        if os.path.exists(status_file):
            try:
                with open(status_file) as f:
                    content = f.read()
                    is_enabled = "EMAIL_ENABLED    = True" in content
                    status = "✅ ENABLED" if is_enabled else "🔴 DISABLED"
                    st.metric("Email Alerts", status)
            except:
                st.warning("Could not read alerts.py")
        
        if st.button("🔧 Open alerts.py", use_container_width=True):
            st.code("""
# In alerts.py, change these:
EMAIL_ENABLED = True
SENDER_EMAIL = "your_gmail@gmail.com"
SENDER_PASSWORD = "xxxx xxxx xxxx xxxx"  # 16-char app password
RECIPIENT_EMAIL = "admin@example.com"
            """, language="python")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
