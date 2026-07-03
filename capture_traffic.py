"""
Real Packet Capture -> Live Flow Classification
====================================================
Captures live network traffic on YOUR OWN machine, reconstructs it into
flows (the same way CICFlowMeter built the CICIDS2017 training data),
computes real flow statistics, and sends each finished flow to your
running NetSentry Flask app for classification.

REQUIREMENTS (run these first):
    pip install scapy requests
    Windows only: install Npcap from https://npcap.com/#download
                  (check "Install Npcap in WinPcap API-compatible mode")

MUST RUN AS ADMINISTRATOR (Windows) / with sudo (Linux/Mac) -- raw packet
capture requires elevated privileges.

LEGAL / ETHICAL NOTE:
Only run this against traffic on a network and device you own or are
explicitly authorized to monitor. Capturing traffic on a network you
don't control (e.g. shared/public Wi-Fi, a workplace network without
permission) is illegal in most jurisdictions.

Usage:
    python capture_traffic.py                  # capture for 30 seconds
    python capture_traffic.py --duration 60     # capture for 60 seconds
    python capture_traffic.py --interface "Wi-Fi"   # pick a specific interface
"""

import argparse
import statistics
import time
from collections import defaultdict

import requests

try:
    from scapy.all import sniff, IP, TCP, UDP, get_working_ifaces
except ImportError:
    raise SystemExit(
        "Scapy is not installed. Run:  pip install scapy\n"
        "On Windows you also need Npcap: https://npcap.com/#download"
    )

API_URL = "http://127.0.0.1:5000/api/predict_full"

TCP_FLAG_BITS = {
    "FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08,
    "ACK": 0x10, "URG": 0x20, "ECE": 0x40, "CWR": 0x80,
}


class Flow:
    """Accumulates packets belonging to one bidirectional flow and computes
    CICFlowMeter-style statistics from them."""

    def __init__(self, fwd_key, first_ts):
        self.fwd_key = fwd_key  # (src_ip, sport, dst_ip, dport, proto)
        self.start_time = first_ts
        self.last_time = first_ts
        self.fwd_lengths, self.bwd_lengths = [], []
        self.fwd_times, self.bwd_times, self.all_times = [], [], [first_ts]
        self.fwd_header_bytes, self.bwd_header_bytes = 0, 0
        self.flag_counts = defaultdict(int)
        self.fwd_psh, self.bwd_psh, self.fwd_urg, self.bwd_urg = 0, 0, 0, 0
        self.init_win_fwd, self.init_win_bwd = -1, -1
        self.act_data_pkt_fwd = 0
        self.min_seg_size_fwd = None

    def add(self, direction, length, header_len, ts, flags=None, win=None, payload_len=0):
        self.last_time = ts
        self.all_times.append(ts)
        if direction == "fwd":
            self.fwd_lengths.append(length)
            self.fwd_times.append(ts)
            self.fwd_header_bytes += header_len
            if payload_len > 0:
                self.act_data_pkt_fwd += 1
            if self.init_win_fwd == -1 and win is not None:
                self.init_win_fwd = win
            if self.min_seg_size_fwd is None or header_len < self.min_seg_size_fwd:
                self.min_seg_size_fwd = header_len
            if flags:
                if flags & TCP_FLAG_BITS["PSH"]:
                    self.fwd_psh += 1
                if flags & TCP_FLAG_BITS["URG"]:
                    self.fwd_urg += 1
        else:
            self.bwd_lengths.append(length)
            self.bwd_times.append(ts)
            self.bwd_header_bytes += header_len
            if self.init_win_bwd == -1 and win is not None:
                self.init_win_bwd = win
            if flags:
                if flags & TCP_FLAG_BITS["PSH"]:
                    self.bwd_psh += 1
                if flags & TCP_FLAG_BITS["URG"]:
                    self.bwd_urg += 1

        if flags:
            for name, bit in TCP_FLAG_BITS.items():
                if flags & bit:
                    self.flag_counts[name] += 1

    @staticmethod
    def _iat_stats(times):
        if len(times) < 2:
            return 0.0, 0.0, 0.0, 0.0
        diffs = [ (times[i+1] - times[i]) * 1_000_000 for i in range(len(times)-1) ]  # microseconds
        mean = statistics.mean(diffs)
        std = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        return mean, std, max(diffs), min(diffs)

    @staticmethod
    def _len_stats(lengths):
        if not lengths:
            return 0.0, 0.0, 0.0, 0.0
        mean = statistics.mean(lengths)
        std = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        return max(lengths), min(lengths), mean, std

    def to_features(self):
        duration_s = max(self.last_time - self.start_time, 1e-6)
        duration_us = duration_s * 1_000_000

        total_fwd = len(self.fwd_lengths)
        total_bwd = len(self.bwd_lengths)
        total_fwd_bytes = sum(self.fwd_lengths)
        total_bwd_bytes = sum(self.bwd_lengths)
        total_bytes = total_fwd_bytes + total_bwd_bytes
        total_packets = total_fwd + total_bwd

        fwd_max, fwd_min, fwd_mean, fwd_std = self._len_stats(self.fwd_lengths)
        bwd_max, bwd_min, bwd_mean, bwd_std = self._len_stats(self.bwd_lengths)
        all_lengths = self.fwd_lengths + self.bwd_lengths
        pkt_max, pkt_min, pkt_mean, pkt_std = self._len_stats(all_lengths)

        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = self._iat_stats(sorted(self.all_times))
        fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._iat_stats(self.fwd_times)
        bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._iat_stats(self.bwd_times)
        fwd_iat_total = sum(self.fwd_times[i+1]-self.fwd_times[i] for i in range(len(self.fwd_times)-1)) * 1e6 if len(self.fwd_times) > 1 else 0
        bwd_iat_total = sum(self.bwd_times[i+1]-self.bwd_times[i] for i in range(len(self.bwd_times)-1)) * 1e6 if len(self.bwd_times) > 1 else 0

        src_ip, sport, dst_ip, dport, proto = self.fwd_key

        return {
            "ip": src_ip,
            "features": {
                "Destination Port": dport,
                "Flow Duration": duration_us,
                "Total Fwd Packets": total_fwd,
                "Total Backward Packets": total_bwd,
                "Total Length of Fwd Packets": total_fwd_bytes,
                "Total Length of Bwd Packets": total_bwd_bytes,
                "Fwd Packet Length Max": fwd_max, "Fwd Packet Length Min": fwd_min,
                "Fwd Packet Length Mean": fwd_mean, "Fwd Packet Length Std": fwd_std,
                "Bwd Packet Length Max": bwd_max, "Bwd Packet Length Min": bwd_min,
                "Bwd Packet Length Mean": bwd_mean, "Bwd Packet Length Std": bwd_std,
                "Flow Bytes/s": total_bytes / duration_s,
                "Flow Packets/s": total_packets / duration_s,
                "Flow IAT Mean": flow_iat_mean, "Flow IAT Std": flow_iat_std,
                "Flow IAT Max": flow_iat_max, "Flow IAT Min": flow_iat_min,
                "Fwd IAT Total": fwd_iat_total, "Fwd IAT Mean": fwd_iat_mean,
                "Fwd IAT Std": fwd_iat_std, "Fwd IAT Max": fwd_iat_max, "Fwd IAT Min": fwd_iat_min,
                "Bwd IAT Total": bwd_iat_total, "Bwd IAT Mean": bwd_iat_mean,
                "Bwd IAT Std": bwd_iat_std, "Bwd IAT Max": bwd_iat_max, "Bwd IAT Min": bwd_iat_min,
                "Fwd PSH Flags": self.fwd_psh, "Bwd PSH Flags": self.bwd_psh,
                "Fwd URG Flags": self.fwd_urg, "Bwd URG Flags": self.bwd_urg,
                "Fwd Header Length": self.fwd_header_bytes, "Bwd Header Length": self.bwd_header_bytes,
                "Fwd Packets/s": total_fwd / duration_s, "Bwd Packets/s": total_bwd / duration_s,
                "Min Packet Length": pkt_min, "Max Packet Length": pkt_max,
                "Packet Length Mean": pkt_mean, "Packet Length Std": pkt_std,
                "Packet Length Variance": pkt_std ** 2,
                "FIN Flag Count": self.flag_counts["FIN"], "SYN Flag Count": self.flag_counts["SYN"],
                "RST Flag Count": self.flag_counts["RST"], "PSH Flag Count": self.flag_counts["PSH"],
                "ACK Flag Count": self.flag_counts["ACK"], "URG Flag Count": self.flag_counts["URG"],
                "CWE Flag Count": self.flag_counts["CWR"], "ECE Flag Count": self.flag_counts["ECE"],
                "Down/Up Ratio": (total_bwd / total_fwd) if total_fwd else 0,
                "Average Packet Size": total_bytes / total_packets if total_packets else 0,
                "Avg Fwd Segment Size": fwd_mean, "Avg Bwd Segment Size": bwd_mean,
                "Fwd Header Length.1": self.fwd_header_bytes,
                "Fwd Avg Bytes/Bulk": 0, "Fwd Avg Packets/Bulk": 0, "Fwd Avg Bulk Rate": 0,
                "Bwd Avg Bytes/Bulk": 0, "Bwd Avg Packets/Bulk": 0, "Bwd Avg Bulk Rate": 0,
                "Subflow Fwd Packets": total_fwd, "Subflow Fwd Bytes": total_fwd_bytes,
                "Subflow Bwd Packets": total_bwd, "Subflow Bwd Bytes": total_bwd_bytes,
                "Init_Win_bytes_forward": max(self.init_win_fwd, 0),
                "Init_Win_bytes_backward": max(self.init_win_bwd, 0),
                "act_data_pkt_fwd": self.act_data_pkt_fwd,
                "min_seg_size_forward": self.min_seg_size_fwd or 0,
                "Active Mean": 0, "Active Std": 0, "Active Max": 0, "Active Min": 0,
                "Idle Mean": 0, "Idle Std": 0, "Idle Max": 0, "Idle Min": 0,
            },
        }


flows = {}


def handle_packet(pkt):
    if IP not in pkt:
        return
    ts = pkt.time
    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst

    if TCP in pkt:
        proto, sport, dport = "TCP", pkt[TCP].sport, pkt[TCP].dport
        flags = int(pkt[TCP].flags)
        win = int(pkt[TCP].window)
        header_len = pkt[IP].ihl * 4 + pkt[TCP].dataofs * 4
        payload_len = len(pkt[TCP].payload)
    elif UDP in pkt:
        proto, sport, dport = "UDP", pkt[UDP].sport, pkt[UDP].dport
        flags, win = None, None
        header_len = pkt[IP].ihl * 4 + 8
        payload_len = len(pkt[UDP].payload)
    else:
        return  # skip ICMP/other for this simplified capture

    norm_key = tuple(sorted([(src_ip, sport), (dst_ip, dport)])) + (proto,)
    length = len(pkt)

    if norm_key not in flows:
        flows[norm_key] = Flow((src_ip, sport, dst_ip, dport, proto), ts)
        direction = "fwd"
    else:
        f = flows[norm_key]
        direction = "fwd" if (src_ip, sport) == f.fwd_key[:2] else "bwd"

    flows[norm_key].add(direction, length, header_len, ts, flags, win, payload_len)


def send_flows():
    print(f"\nSending {len(flows)} captured flows to NetSentry for classification...")
    sent, failed = 0, 0
    for key, flow in flows.items():
        if len(flow.fwd_lengths) + len(flow.bwd_lengths) < 2:
            continue  # skip trivial single-packet noise
        payload = flow.to_features()
        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            if r.ok:
                result = r.json()
                tag = "ATTACK" if result.get("is_attack") else "normal"
                print(f"  [{tag}] {payload['ip']} -> port {payload['features']['Destination Port']}: "
                      f"{result.get('attack_type')} ({result.get('risk_level')})")
                sent += 1
            else:
                print(f"  Request failed ({r.status_code}): {r.text[:300]}")
                failed += 1
        except requests.exceptions.RequestException as e:
            print(f"  Failed to reach NetSentry API ({API_URL}). Is app_v2.py running? Error: {e}")
            break
    print(f"\nDone. Sent {sent} flows, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(description="Capture live traffic and classify it with NetSentry")
    parser.add_argument("--duration", type=int, default=30, help="Seconds to capture (default 30)")
    parser.add_argument("--interface", type=str, default=None, help="Network interface name (optional)")
    args = parser.parse_args()

    print("Available interfaces:")
    for iface in get_working_ifaces():
        print(" -", iface.name)

    print(f"\nCapturing on {args.interface or 'default interface'} for {args.duration} seconds...")
    print("Browse the web, stream something, etc. -- generate some real traffic to capture.")
    print("(Requires Administrator/root privileges to capture packets.)\n")

    sniff(iface=args.interface, timeout=args.duration, prn=handle_packet, store=False)

    print(f"\nCapture complete. Reconstructed {len(flows)} flows.")
    if flows:
        send_flows()
    else:
        print("No flows captured -- try generating more traffic, or check you're running as Administrator.")


if __name__ == "__main__":
    main()
