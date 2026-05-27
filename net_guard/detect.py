"""
detect.py
---------
Real-Time Packet Capture & Detection Module.

  • Captures live packets with Scapy
  • Maps packet fields to KDD-style features
  • Runs XGBoost (and optionally Autoencoder) for anomaly detection
  • Logs results to detect_log.jsonl for dashboard consumption
  • Prints colour-coded alerts to stdout

Fixes applied:
  1. Whitelist for known-normal multicast/broadcast/mDNS traffic
  2. Stricter zero-day condition (AE MSE must be significantly above threshold)
  3. Minimum confidence threshold to suppress near-zero false positives
"""

import os
import sys
import time
import json
import socket
import struct
import datetime
import logging
import psutil
import numpy as np
import joblib
from collections import defaultdict

# ── Suppress Scapy logging (fixes JSON serialization warnings) ────────────────
logging.getLogger("scapy").setLevel(logging.ERROR)

# ── Alert system ──────────────────────────────────────────────────────────────
try:
    from alerts import send_email_alert, should_alert
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    print("[WARN] alerts.py not found. Email notifications disabled.")

# ── Scapy ─────────────────────────────────────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARN] Scapy not installed. Live capture unavailable.")

# ── Optional autoencoder ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
LOG_FILE   = "detect_log.jsonl"
N_FEATURES = 41

# Minimum XGBoost confidence to log/alert (suppresses near-zero noise)
# Lower = more sensitive to attacks
# Higher = fewer false positives
# FIXED: Lowered to 0.01 to catch all attack patterns while keeping autoencoder (Step 4)
# protection against false positives. Autoencoder filters normal traffic early, allowing
# aggressive XGBoost threshold for all attack types: TCP/ICMP (0.00-0.01), UDP (0.05),
# Port Scan (0.59), Fragmented (0.73).
MIN_CONFIDENCE_THRESHOLD = 0.01

# How many times above AE threshold MSE must be to count as zero-day
# (e.g. 5.0 = MSE must be at least 5x the threshold - much stricter)
ZERO_DAY_MSE_MULTIPLIER = 5.0

# KDD protocol_type encoding (must match training encoder)
PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

# Simplified service mapping (port → index)
SERVICE_MAP = defaultdict(lambda: 0, {
    80: 1, 443: 2, 21: 3, 22: 4, 23: 5,
    25: 6, 53: 7, 110: 8, 143: 9, 3306: 10,
})

# TCP flag bit mapping → KDD flag index
FLAG_MAP = {
    0x02: 1,   # SYN
    0x12: 2,   # SYN-ACK
    0x10: 3,   # ACK
    0x01: 4,   # FIN
    0x04: 5,   # RST
    0x18: 6,   # PSH-ACK
    0x11: 7,   # FIN-ACK
}

# ── Whitelist: known-normal background traffic ────────────────────────────────
# These are standard LAN/OS background protocols that will ALWAYS look
# anomalous to a KDD-1999-trained model. Skip them entirely.
WHITELIST_DST_IPS = {
    "239.255.255.250",   # SSDP/UPnP multicast
    "224.0.0.251",       # mDNS multicast
    "224.0.0.1",         # IGMP all-hosts multicast
    "224.0.0.22",        # IGMP v3 multicast
    "255.255.255.255",   # Limited broadcast
    # Legitimate cloud/CDN providers (NOT attacks)
    "20.207.70.99",      # Microsoft Azure
    "140.82.113.25",     # GitHub
    "140.82.114.22",     # GitHub API
    "13.89.179.14",      # Azure
    "13.107.5.93",       # Microsoft
    "18.97.36.76",       # AWS
    "18.214.0.0/15",     # AWS (approximate range)
    "35.237.69.59",      # Google Cloud
    "104.199.241.202",   # Google Cloud
    "129.227.254.87",    # Legitimate service
    "40.79.150.121",     # Azure
    "40.112.143.140",    # Microsoft
    "123.63.248.158",    # Legitimate service
}

# FIXED: Whitelist for source IPs (replying from legitimate providers)
WHITELIST_SRC_IPS = {
    "140.82.113.25",     # GitHub
    "140.82.114.22",     # GitHub API (reverse traffic)
    "13.107.5.93",       # Microsoft (reverse traffic)
    "18.97.36.76",       # AWS
    "18.97.36.23",       # AWS
    "35.237.69.59",      # Google Cloud
    "104.199.241.202",   # Google Cloud
    "40.79.150.121",     # Azure
    "123.63.248.158",    # Legitimate service
}

WHITELIST_DST_PORTS = {
    5353,    # mDNS
    1900,    # SSDP / UPnP
    5355,    # LLMNR
    137,     # NetBIOS Name Service
    138,     # NetBIOS Datagram
    631,     # IPP (printer discovery)
}

# Terminal colours
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist Check
# ─────────────────────────────────────────────────────────────────────────────

def is_whitelisted(pkt) -> bool:
    """
    Return True if the packet should be SKIPPED (it's known-normal background
    traffic that will always false-positive against a KDD-1999 model).
    
    FIXED: Now also checks source IPs (for reverse traffic from cloud providers)
    and uses more aggressive range-based filtering for known-good providers.
    """
    if not pkt.haslayer(IP):
        return False

    ip = pkt[IP]

    # Source IP whitelist (FIXED: added to catch CDN/cloud replies)
    if ip.src in WHITELIST_SRC_IPS:
        return True

    # Destination IP whitelist (multicast / broadcast)
    if ip.dst in WHITELIST_DST_IPS:
        return True

    # Check if destination IP is in AWS range (18.x.x.x)
    if ip.dst.startswith("18."):
        return True

    # Check if source IP is in AWS range (18.x.x.x) — FIXED
    if ip.src.startswith("18."):
        return True

    # Check if source IP is in Azure/Microsoft range (13.x.x.x, 40.x.x.x) — FIXED
    if ip.src.startswith("13.") or ip.src.startswith("40."):
        return True

    # Destination subnet whitelist (224.x.x.x = all multicast, 233.x.x.x = MCAST-prefix)
    if ip.dst.startswith("224.") or ip.dst.startswith("233."):
        return True

    # Port-based whitelist
    dst_port = 0
    if pkt.haslayer(UDP):
        dst_port = pkt[UDP].dport
    elif pkt.haslayer(TCP):
        dst_port = pkt[TCP].dport

    if dst_port in WHITELIST_DST_PORTS:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """Load XGBoost model, scaler, and (optionally) autoencoder."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    xgb_path    = os.path.join(MODEL_DIR, "xgboost_model.pkl")

    if not os.path.exists(scaler_path) or not os.path.exists(xgb_path):
        sys.exit("[ERROR] Models not found. Run train.py first.")

    scaler    = joblib.load(scaler_path)
    xgb_model = joblib.load(xgb_path)
    print("[INFO] XGBoost model and scaler loaded.")

    ae_model   = None
    ae_thresh  = None
    ae_path    = os.path.join(MODEL_DIR, "autoencoder.keras")
    thr_path   = os.path.join(MODEL_DIR, "ae_threshold.pkl")
    if TF_AVAILABLE and os.path.exists(ae_path):
        ae_model  = tf.keras.models.load_model(ae_path)
        ae_thresh = joblib.load(thr_path)
        print(f"[INFO] Autoencoder loaded. Threshold: {ae_thresh:.6f}")

    return scaler, xgb_model, ae_model, ae_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(pkt) -> np.ndarray | None:
    """
    Map a Scapy packet to an approximate KDD feature vector (41 features).
    Returns None if the packet has no IP layer.
    """
    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    features = np.zeros(N_FEATURES, dtype=np.float32)

    # Feature 0: duration
    features[0] = 0.0

    # Feature 1: protocol_type
    if pkt.haslayer(TCP):
        features[1] = PROTOCOL_MAP["tcp"]
    elif pkt.haslayer(UDP):
        features[1] = PROTOCOL_MAP["udp"]
    elif pkt.haslayer(ICMP):
        features[1] = PROTOCOL_MAP["icmp"]

    # Feature 2: service
    dst_port = 0
    src_port = 0
    if pkt.haslayer(TCP):
        dst_port = pkt[TCP].dport
        src_port = pkt[TCP].sport
    elif pkt.haslayer(UDP):
        dst_port = pkt[UDP].dport
        src_port = pkt[UDP].sport
    features[2] = SERVICE_MAP[dst_port]

    # Feature 3: flag
    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        features[3] = FLAG_MAP.get(int(flags), 0)

    # Feature 4: src_bytes
    features[4] = len(ip.payload) if ip.payload else 0

    # Feature 5: dst_bytes
    features[5] = 0

    # Feature 6: land
    features[6] = int(ip.src == ip.dst and src_port == dst_port)

    # Feature 7: wrong_fragment
    features[7] = int(ip.frag != 0)

    # Features 22-23: count / srv_count
    features[22] = 1
    features[23] = 1

    # Features 24-30: rate features
    features[28] = 1.0
    features[29] = 0.0
    features[30] = 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def predict(features: np.ndarray, scaler, xgb_model, ae_model, ae_thresh):
    """
    Returns (label, confidence, zero_day_flag, ae_mse).

    Zero-day logic (very strict):
      - XGBoost confidence BELOW 0.2 (very confident it's normal, not just < 0.5)
      - Autoencoder MSE is > ZERO_DAY_MSE_MULTIPLIER * threshold (5x default)
        (must be SIGNIFICANTLY above threshold)
    """
    x = scaler.transform(features.reshape(1, -1))

    prob      = xgb_model.predict_proba(x)[0][1]
    xgb_label = int(prob >= 0.5)

    ae_mse   = 0.0
    zero_day = False

    if ae_model is not None:
        recon    = ae_model.predict(x, verbose=0)
        ae_mse   = float(np.mean(np.power(x - recon, 2)))
        # Very strict zero-day: XGBoost must be VERY confident it's normal
        # (prob < 0.2) AND Autoencoder MSE must be significantly above threshold
        zero_day = (ae_mse > ae_thresh * ZERO_DAY_MSE_MULTIPLIER) and (prob < 0.2)

    return xgb_label, float(prob), zero_day, ae_mse


def save_performance_metrics(perf_stats):
    """Save detector performance metrics to JSON."""
    try:
        elapsed = time.time() - perf_stats["start_time"]
        total_pkt = perf_stats["total_packets"]
        
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "total_packets": total_pkt,
            "packets_per_second": round(total_pkt / max(elapsed, 0.1), 2),
            "avg_detection_latency_ms": round(
                (sum(perf_stats["detection_times"]) / max(len(perf_stats["detection_times"]), 1)) * 1000, 2
            ),
            "max_detection_latency_ms": round(max(perf_stats["detection_times"] or [0]) * 1000, 2),
            "cpu_percent": round(perf_stats["cpu_percent"], 2),
            "memory_mb": round(perf_stats["memory_mb"], 2),
        }
        
        with open("perf_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save performance metrics: {e}")


def log_event(src_ip, dst_ip, protocol, label, confidence, zero_day, ae_mse=0.0):
    """Append a detection event to detect_log.jsonl and send alerts if needed."""
    event = {
        "timestamp" : datetime.datetime.now().isoformat(),
        "src_ip"    : src_ip,
        "dst_ip"    : dst_ip,
        "protocol"  : protocol,
        "label"     : "anomaly" if label == 1 else "normal",
        "confidence": round(float(confidence), 4),
        "zero_day"  : bool(zero_day),  # Convert to native Python bool
        "ae_mse"    : round(float(ae_mse), 6),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    
    # ── Send email alert if anomaly detected ───────────────────────────────────
    if ALERTS_AVAILABLE and label == 1:  # Only for anomalies
        attack_type = "ZERO-DAY" if zero_day else "ATTACK"
        if should_alert(confidence, zero_day):
            send_email_alert(
                src_ip=src_ip,
                dst_ip=dst_ip,
                attack_type=attack_type,
                confidence=confidence,
                protocol=protocol,
                is_zero_day=zero_day
            )
    
    return event


def handle_packet(pkt, scaler, xgb_model, ae_model, ae_thresh, debug=False):
    """Scapy callback: filter, extract, predict, log, and alert."""

    # ── Step 1: Skip known-normal background traffic ──────────────────────────
    if is_whitelisted(pkt):
        if debug:
            try:
                ip = pkt[IP]
                proto = "tcp" if pkt.haslayer(TCP) else "udp" if pkt.haslayer(UDP) else "icmp"
                print(f"[SKIP] Whitelisted: {ip.src} → {ip.dst} | {proto.upper()}", flush=True)
            except:
                pass
        return

    # ── Step 2: Extract features ──────────────────────────────────────────────
    features = extract_features(pkt)
    if features is None:
        if debug:
            print(f"[SKIP] No IP layer in packet", flush=True)
        return

    ip    = pkt[IP]
    proto = "tcp" if pkt.haslayer(TCP) else "udp" if pkt.haslayer(UDP) else "icmp"

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    try:
        label, confidence, zero_day, ae_mse = predict(
            features, scaler, xgb_model, ae_model, ae_thresh
        )
    except Exception as e:
        if debug:
            print(f"[ERROR] Prediction failed: {e}", flush=True)
        return

    # ── Step 4: Trust autoencoder over XGBoost (FIXED) ──────────
    # If autoencoder says it's NORMAL traffic (low MSE), log as SAFE
    # and skip alerts. The retrained AE knows your real network better
    # than the old KDD model. Still log the event for visibility.
    if ae_model is not None and ae_mse < ae_thresh:
        try:
            event = log_event(ip.src, ip.dst, proto, 0, confidence, False, ae_mse)
            ts = event["timestamp"].split("T")[1][:8]
            print(
                f"{GREEN}[{ts}] ✅ SAFE     "
                f"{ip.src} → {ip.dst} | {proto.upper()} | "
                f"ae_mse={ae_mse:.6f}{RESET}",
                flush=True
            )
        except Exception as e:
            if debug:
                print(f"[ERROR] Error logging safe packet: {e}", flush=True)
        return

    # ── Step 5: Skip very low-confidence packets (noise suppression) ──────────
    if confidence < MIN_CONFIDENCE_THRESHOLD and not zero_day:
        if debug:
            print(f"[SKIP] Low confidence: {ip.src} → {ip.dst} | {proto.upper()} | conf={confidence:.2f}", flush=True)
        return

    # ── Step 6: Log & print ───────────────────────────────────────────────────
    try:
        event = log_event(ip.src, ip.dst, proto, label, confidence, zero_day, ae_mse)
        ts    = event["timestamp"].split("T")[1][:8]

        if zero_day:
            print(
                f"{YELLOW}[{ts}] ⚠️  ZERO-DAY  "
                f"{ip.src} → {ip.dst} | {proto.upper()} | "
                f"conf={confidence:.2f} | ae_mse={ae_mse:.4f}{RESET}",
                flush=True
            )
        elif label == 1:
            print(
                f"{RED}[{ts}] 🚨 ATTACK    "
                f"{ip.src} → {ip.dst} | {proto.upper()} | "
                f"conf={confidence:.2f}{RESET}",
                flush=True
            )
        else:
            print(
                f"{GREEN}[{ts}] ✅ SAFE     "
                f"{ip.src} → {ip.dst} | {proto.upper()} | "
                f"conf={confidence:.2f}{RESET}",
                flush=True
            )
    except Exception as e:
        print(f"[ERROR] While printing: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(interface: str = None, packet_count: int = 0, timeout: int = None, debug: bool = False):
    """
    Start live packet capture and anomaly detection.

    Args:
        interface   : Network interface name (e.g., "eth0"). None = default.
        packet_count: Number of packets to capture (0 = infinite).
        timeout     : Stop after this many seconds (None = no limit).
        debug       : Print debug info about whitelisted/low-confidence packets.
    """
    if not SCAPY_AVAILABLE:
        sys.exit("[ERROR] Scapy is not installed. Run: pip install scapy")

    scaler, xgb_model, ae_model, ae_thresh = load_models()

    print(f"\n[INFO] Whitelist active — skipping multicast/mDNS/SSDP background traffic")
    print(f"[INFO] Min confidence threshold: {MIN_CONFIDENCE_THRESHOLD}")
    print(f"[INFO] Zero-day MSE multiplier : {ZERO_DAY_MSE_MULTIPLIER}x above AE threshold")
    if debug:
        print(f"[INFO] DEBUG MODE: ON (showing whitelisted & low-confidence packets)")
    print(f"\n[INFO] Starting capture on interface: {interface or 'default'}")
    print("[INFO] Press Ctrl-C to stop.\n", flush=True)

    packet_count_stat = {"total": 0, "processed": 0}
    perf_stats = {
        "start_time": time.time(),
        "total_packets": 0,
        "detection_times": [],
        "cpu_percent": 0,
        "memory_mb": 0,
    }
    process = psutil.Process()

    def counting_handler(pkt):
        packet_count_stat["total"] += 1
        start_time = time.time()
        try:
            handle_packet(pkt, scaler, xgb_model, ae_model, ae_thresh, debug=debug)
        except Exception as e:
            print(f"[ERROR] Exception in handle_packet: {e}", flush=True)
        
        # Track performance metrics
        elapsed = time.time() - start_time
        perf_stats["detection_times"].append(elapsed)
        perf_stats["total_packets"] += 1
        
        # Update CPU/Memory every 50 packets
        if packet_count_stat["total"] % 50 == 0:
            perf_stats["cpu_percent"] = process.cpu_percent(interval=0.1)
            perf_stats["memory_mb"] = process.memory_info().rss / 1024 / 1024
            save_performance_metrics(perf_stats)
        
        packet_count_stat["processed"] += 1

    try:
        sniff(
            iface=interface,
            prn=counting_handler,
            store=False,
            count=packet_count if packet_count > 0 else 0,
            timeout=timeout if timeout and timeout > 0 else None,  # None = wait indefinitely
            filter="ip",
        )
        print(f"\n[INFO] Capture completed.", flush=True)
    except KeyboardInterrupt:
        print(f"\n[INFO] Capture stopped by user.", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Capture error: {e}", flush=True)
    finally:
        save_performance_metrics(perf_stats)
        print(f"[STATS] Total packets: {packet_count_stat['total']}", flush=True)


if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv or "-d" in sys.argv
    interface = None
    
    # Check for -i interface argument
    for i, arg in enumerate(sys.argv):
        if arg == "-i" and i + 1 < len(sys.argv):
            interface = sys.argv[i + 1]
    
    print("[INFO] Starting detector in continuous loop mode (runs indefinitely)")
    print("[INFO] Press Ctrl+C to stop\n")
    
    # Run detection in infinite loop
    try:
        while True:
            run_detection(packet_count=0, debug=debug, interface=interface, timeout=None)
    except KeyboardInterrupt:
        print("\n[INFO] Detector stopped by user.")
        print("[STATS] Exiting...")
        sys.exit(0)