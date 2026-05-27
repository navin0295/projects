"""
capture_normal.py
-----------------
Capture live NORMAL traffic from your network interface and save it as a
numpy array for retraining the Autoencoder on real traffic patterns.

This is the BEST fix for the false zero-day alert problem. The KDD-1999
autoencoder doesn't know what modern traffic looks like — this script
teaches it YOUR network's normal baseline.

Usage:
    sudo python capture_normal.py

Then retrain the autoencoder:
    python train.py --retrain-live

Then restart detection:
    sudo python detect.py

⚠  Run this ONLY during normal network usage.
   Do NOT run inject.py while capturing — it will corrupt the normal baseline.
"""

import os
import sys
import numpy as np

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[ERROR] Scapy not installed. Run: pip install scapy")
    sys.exit(1)

# ── Settings ──────────────────────────────────────────────────────────────────
CAPTURE_COUNT = 3000          # number of normal packets to capture
OUTPUT_DIR    = "data"
OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "X_normal_live.npy")
N_FEATURES    = 41

# ── Reuse the same whitelist and feature extractor from detect.py ─────────────
from collections import defaultdict

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
SERVICE_MAP  = defaultdict(lambda: 0, {
    80: 1, 443: 2, 21: 3, 22: 4, 23: 5,
    25: 6, 53: 7, 110: 8, 143: 9, 3306: 10,
})
FLAG_MAP = {
    0x02: 1, 0x12: 2, 0x10: 3,
    0x01: 4, 0x04: 5, 0x18: 6, 0x11: 7,
}

# Same whitelist as detect.py — skip background noise during capture too
WHITELIST_DST_IPS = {
    "239.255.255.250",
    "224.0.0.251",
    "224.0.0.1",
    "224.0.0.22",
    "255.255.255.255",
}
WHITELIST_DST_PORTS = {5353, 1900, 5355, 137, 138, 631}


def is_whitelisted(pkt) -> bool:
    if not pkt.haslayer(IP):
        return True   # skip non-IP
    ip = pkt[IP]
    if ip.dst in WHITELIST_DST_IPS:
        return True
    if ip.dst.startswith("224.") or ip.dst.startswith("233."):
        return True
    dst_port = 0
    if pkt.haslayer(UDP):
        dst_port = pkt[UDP].dport
    elif pkt.haslayer(TCP):
        dst_port = pkt[TCP].dport
    return dst_port in WHITELIST_DST_PORTS


def extract_features(pkt):
    if not pkt.haslayer(IP):
        return None
    ip       = pkt[IP]
    features = np.zeros(N_FEATURES, dtype=np.float32)
    features[0] = 0.0

    if pkt.haslayer(TCP):
        features[1] = PROTOCOL_MAP["tcp"]
    elif pkt.haslayer(UDP):
        features[1] = PROTOCOL_MAP["udp"]
    elif pkt.haslayer(ICMP):
        features[1] = PROTOCOL_MAP["icmp"]

    dst_port = src_port = 0
    if pkt.haslayer(TCP):
        dst_port = pkt[TCP].dport
        src_port = pkt[TCP].sport
    elif pkt.haslayer(UDP):
        dst_port = pkt[UDP].dport
        src_port = pkt[UDP].sport

    features[2]  = SERVICE_MAP[dst_port]
    if pkt.haslayer(TCP):
        features[3] = FLAG_MAP.get(int(pkt[TCP].flags), 0)
    features[4]  = len(ip.payload) if ip.payload else 0
    features[5]  = 0
    features[6]  = int(ip.src == ip.dst and src_port == dst_port)
    features[7]  = int(ip.frag != 0)
    features[22] = 1
    features[23] = 1
    features[28] = 1.0
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Main Capture Loop
# ─────────────────────────────────────────────────────────────────────────────

def capture_normal_traffic(count: int = CAPTURE_COUNT, interface: str = None):
    samples = []

    print(f"\n{'='*60}")
    print(f"  NetGuard — Normal Traffic Baseline Capture")
    print(f"{'='*60}")
    print(f"\n  Target samples : {count}")
    print(f"  Interface      : {interface or 'default'}")
    print(f"  Output         : {OUTPUT_PATH}")
    print(f"\n  ✅  Use your computer NORMALLY during this capture.")
    print(f"  ✅  Browse the web, stream video, use apps — all good.")
    print(f"  ❌  Do NOT run inject.py — it will corrupt the baseline.")
    print(f"\n  Starting capture...\n")

    def collect(pkt):
        if is_whitelisted(pkt):
            return
        feat = extract_features(pkt)
        if feat is not None:
            samples.append(feat)
            n = len(samples)
            if n % 500 == 0 or n == count:
                pct = n / count * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"  [{bar}] {n}/{count} packets ({pct:.0f}%)")

    sniff(
        iface=interface,
        filter="ip",
        prn=collect,
        count=count,
        store=False,
    )

    if not samples:
        print("[ERROR] No packets captured. Check interface name and sudo privileges.")
        return None

    X = np.array(samples, dtype=np.float32)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_PATH, X)

    print(f"\n[INFO] ✅  Captured {X.shape[0]} normal samples")
    print(f"[INFO] Saved → {OUTPUT_PATH}")
    print(f"\nNext step — retrain the autoencoder on this data:")
    print(f"  python train.py --retrain-live\n")
    return X


if __name__ == "__main__":
    if not SCAPY_AVAILABLE:
        sys.exit("[ERROR] Scapy required. Run: pip install scapy")

    import argparse
    parser = argparse.ArgumentParser(description="Capture normal traffic for AE retraining")
    parser.add_argument("--count",     type=int, default=CAPTURE_COUNT,
                        help=f"Number of packets to capture (default: {CAPTURE_COUNT})")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface (default: auto-detect)")
    args = parser.parse_args()

    capture_normal_traffic(count=args.count, interface=args.interface)
