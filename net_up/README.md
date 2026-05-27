# 🛡️ NetGuard — AI-Driven Network Behavior Analytics

A modular Python system for **real-time network anomaly detection and security monitoring** using the KDD Cup 1999 dataset, XGBoost, and an optional deep-learning Autoencoder for zero-day discovery.

---

## 📁 Project Structure

```
network_analytics/
├── preprocessing.py    # Data loading, encoding, normalization, splitting
├── train.py            # XGBoost + Autoencoder training & evaluation
├── detect.py           # Live packet capture & real-time anomaly detection
├── inject.py           # Attack traffic simulation (SYN flood, port scan, …)
├── dashboard.py        # Streamlit real-time monitoring dashboard
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

Auto-generated at runtime:
```
models/                 # Saved models, scaler, encoders, metrics
data/                   # Preprocessed numpy arrays
detect_log.jsonl        # Real-time detection event log (newline-delimited JSON)
kddcup.data_10_percent.gz   # Downloaded KDD dataset
```

## 🔐 Before You Push to GitHub

The following items are generated at runtime and are usually best left out of a public repo:

- `detect_log.jsonl`
- `data/`
- `models/`
- `kddcup.data_10_percent.gz`
- `__pycache__/`
- `.DS_Store`
- `__MACOSX/` folders if they came from a ZIP archive

The project currently does not ship with a `.gitignore`, so these files can be committed accidentally unless you add one.

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow is optional (Autoencoder). If you skip it, set `train_ae=False` in `train.py`.  
> **Note:** Scapy requires **root/Administrator** privileges for live packet capture.

### 1.1 Configure Email Alerts

Email alerts are disabled by default until you configure SMTP credentials.

Open [alerts.py](alerts.py) and set:

- `EMAIL_ENABLED = True`
- `SENDER_EMAIL = "your_email@example.com"`
- `SENDER_PASSWORD = "your_app_password"`
- `RECIPIENT_EMAIL = "recipient@example.com"`

If you use Gmail, create an app password instead of using your normal password. For other providers, update `SMTP_SERVER` and `SMTP_PORT` accordingly.

### How to get the password

For Gmail, use a 16-character **app password**:

1. Turn on 2-Step Verification for your Google account.
2. Open Google Account settings and go to the App Passwords page.
3. Create a new app password for `Mail`.
4. Copy the generated 16-character password.
5. Paste it into `SENDER_PASSWORD` in [alerts.py](alerts.py).

Do not use your normal Gmail login password in the alert config. If you are using Outlook or Yahoo, create the equivalent SMTP/app password from that provider’s security settings.

You can test the alert path with:

```bash
python -c "from alerts import send_email_alert; send_email_alert('192.168.1.1', '192.168.1.6', 'TCP SYN Flood', 0.95, 'tcp')"
```

---

### 2. Preprocess the Dataset

```bash
python preprocessing.py
```

- Downloads the KDD 10% dataset (~18 MB) automatically
- Encodes `protocol_type`, `service`, `flag` using LabelEncoder
- Binarizes labels (`normal=0`, `attack=1`)
- Normalizes features with StandardScaler
- Saves splits to `data/` and scaler/encoders to `models/`

---

### 3. Train the Models

```bash
python train.py
```

- Trains **XGBoost** classifier (300 estimators, depth 6)
- Trains **Autoencoder** on normal traffic only (zero-day detection)
- Prints accuracy, precision, recall, F1-score + confusion matrix
- Saves models to `models/`

Typical results on KDD 10%:

| Model        | Accuracy | Precision | Recall | F1    |
|--------------|----------|-----------|--------|-------|
| XGBoost      | ~99.8%   | ~99.9%    | ~99.7% | ~99.8%|
| Autoencoder  | ~98.5%   | ~98.1%    | ~99.0% | ~98.5%|

---

### 4. Start Real-Time Detection

```bash
# Requires root/sudo on Linux/macOS
sudo python detect.py
```

- Captures live IP packets with Scapy
- Extracts ~41 KDD-style features per packet
- Runs XGBoost prediction + Autoencoder reconstruction check
- Prints colour-coded alerts:
  - 🟢 Normal
  - 🔴 Anomaly (known attack signature)
  - 🟡 Zero-Day candidate (AE flags it but XGB says normal)
- Appends all events to `detect_log.jsonl`

If you want email notifications, enable and configure them in [alerts.py](alerts.py) before running detection.

Optional arguments (edit inside `detect.py` or call `run_detection()`):
```python
run_detection(interface="eth0", packet_count=500, timeout=60)
```

---

### 5. Inject Simulated Attacks

```bash
# Requires root/sudo
sudo python inject.py --target 127.0.0.1 --count 100 --attack all
```

Attack types: `syn | udp | icmp | scan | frag | all`

| Attack          | Technique                          |
|-----------------|------------------------------------|
| TCP SYN Flood   | DoS via half-open connections      |
| UDP Flood       | Bandwidth exhaustion               |
| ICMP Flood      | Ping storm                         |
| Port Scan       | Reconnaissance SYN sweep           |
| Fragmented IP   | IDS evasion via IP fragmentation   |

> ⚠️ **Only test against hosts you own or have explicit permission to test.**

---

### 6. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

Dashboard features:
- 📊 KPI cards: total packets, anomalies, zero-days, anomaly rate
- 📈 Anomaly trend chart (per minute)
- 🥧 Protocol breakdown bar chart
- 🎯 Top threat source leaderboard
- 📋 Live event log table (last 50 events)
- 🔴 Live alert feed (colour-coded)
- 🧠 Model performance metrics
- Auto-refreshes every N seconds (configurable in sidebar)

---

## 🔬 Zero-Day Detection Logic

The system uses a **hybrid approach**:

1. **XGBoost** detects known attack patterns learned from KDD training labels.
2. **Autoencoder** is trained *only on normal traffic*. Any input that produces high reconstruction error (> 95th percentile of training MSE) is flagged as anomalous — even if XGBoost doesn't recognize it.
3. A packet is marked **Zero-Day candidate** when:
   - XGBoost predicts **normal** (unknown to supervised model)
   - Autoencoder reconstruction error **exceeds threshold** (behaviorally unusual)

This combination catches:
- Known attacks (XGBoost, high precision)
- Novel/unseen attack variants (Autoencoder, behavioral anomaly)

---

## 🛠️ Configuration & Extension

| File            | What to customise                                      |
|-----------------|--------------------------------------------------------|
| `preprocessing.py` | `test_size`, dataset path, feature list            |
| `train.py`      | XGBoost hyperparameters, AE architecture, epochs       |
| `detect.py`     | `PROTOCOL_MAP`, `SERVICE_MAP`, `FLAG_MAP`, threshold   |
| `inject.py`     | Target IPs, packet counts, inter-packet delay          |
| `dashboard.py`  | Refresh rate, chart types, log path                    |

### Alert Configuration

The email alert flow is implemented in [alerts.py](alerts.py). It uses SMTP and needs all three values to be configured before alerts can be sent:

- sender email address
- sender SMTP/app password
- recipient email address

By default, email alerts stay off so the project runs safely without credentials.

---

## 📦 Dependencies

| Library       | Purpose                        |
|---------------|--------------------------------|
| pandas        | Data manipulation              |
| numpy         | Numerical operations           |
| scikit-learn  | Preprocessing, metrics         |
| xgboost       | Gradient-boosted classifier    |
| tensorflow    | Autoencoder (optional)         |
| scapy         | Packet capture & injection     |
| streamlit     | Real-time dashboard            |
| joblib        | Model serialization            |

---

## 📜 License

For educational and research purposes only. The KDD Cup 1999 dataset is publicly available from the [UCI ML Repository](http://kdd.ics.uci.edu/databases/kddcup99/).
