"""
AI-Based Intrusion Detection System - Flask Backend
=====================================================
Architecture:
  Incoming traffic sample -> feature construction -> ML models
  -> alert classification -> SQLite storage -> dashboard / API

Run:  python3 app.py
Then open http://127.0.0.1:5000
"""

import os
import json
import sqlite3
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, g

MODEL_DIR = "models"
DB_PATH = "ids_alerts.db"

app = Flask(__name__)

# ---------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------
rf_binary = joblib.load(os.path.join(MODEL_DIR, "random_forest_binary.pkl"))
rf_category = joblib.load(os.path.join(MODEL_DIR, "random_forest_category.pkl"))
dt_binary = joblib.load(os.path.join(MODEL_DIR, "decision_tree_binary.pkl"))
iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, "risk_levels.json")) as f:
    RISK_LEVEL = json.load(f)
with open(os.path.join(MODEL_DIR, "baseline_row.json")) as f:
    BASELINE = json.load(f)

FRIENDLY_NAME = {
    "normal": "Normal Traffic",
    "DoS": "Denial of Service / DDoS",
    "Probe": "Port Scan / Probing",
    "R2L": "Brute Force / Unauthorized Remote Access",
    "U2R": "Privilege Escalation",
    "Other": "Unknown Suspicious Activity",
}


# ---------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ip TEXT NOT NULL,
            protocol TEXT NOT NULL,
            packets_per_sec REAL NOT NULL,
            failed_logins INTEGER NOT NULL,
            is_attack INTEGER NOT NULL,
            attack_type TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            confidence REAL NOT NULL,
            anomaly_flag INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------
# Feature construction: map the simplified real-time input
# (IP, packets/sec, failed logins, protocol) onto the full
# NSL-KDD-style feature vector the models were trained on.
#
# NOTE: This is a simplification for demo purposes. A production
# system would derive the full feature set (byte counts, service,
# flag, error rates, etc.) from real packet captures / NetFlow,
# e.g. via CICFlowMeter, rather than approximating it like this.
# ---------------------------------------------------------------
def build_feature_vector(packets_per_sec, failed_logins, protocol):
    row = dict(BASELINE)  # start from a "typical normal" baseline

    proto = protocol.lower().strip()
    proto_le = encoders["protocol_type"]
    if proto not in proto_le.classes_:
        proto = proto_le.classes_[0]
    row["protocol_type"] = proto_le.transform([proto])[0]

    # count / srv_count: connections to same host/service in a recent window
    # -- scaled directly from packets/sec, capped to the range seen in training
    count_val = min(packets_per_sec, 511)
    row["count"] = count_val
    row["srv_count"] = count_val

    # very high packet rate looks like a SYN-flood / DoS pattern:
    # high error / same-service rates, low success rate
    if packets_per_sec > 1000:
        row["serror_rate"] = 1.0
        row["srv_serror_rate"] = 1.0
        row["same_srv_rate"] = 1.0
        row["dst_host_count"] = 255
        row["dst_host_srv_count"] = 255
        row["src_bytes"] = 0

    row["num_failed_logins"] = failed_logins
    if failed_logins > 0:
        row["logged_in"] = 0
        row["hot"] = min(failed_logins, 30)
    else:
        row["logged_in"] = 1

    vector = [row[col] for col in FEATURE_COLS]
    return pd.DataFrame([vector], columns=FEATURE_COLS)


def classify(packets_per_sec, failed_logins, protocol):
    X = build_feature_vector(packets_per_sec, failed_logins, protocol)
    X_scaled = scaler.transform(X)

    rf_pred = rf_binary.predict(X_scaled)[0]           # 0 normal / 1 attack
    rf_proba = rf_binary.predict_proba(X_scaled)[0][1]  # attack probability
    dt_pred = dt_binary.predict(X_scaled)[0]
    iso_pred = iso_forest.predict(X_scaled)[0]          # 1 normal / -1 anomaly
    anomaly_flag = 1 if iso_pred == -1 else 0

    is_attack = 1 if (rf_pred == 1 or dt_pred == 1 or anomaly_flag == 1) else 0

    if is_attack:
        cat_idx = rf_category.predict(X_scaled)[0]
        category = encoders["attack_category"].inverse_transform([cat_idx])[0]
        if category == "normal":
            category = "Other"
    else:
        category = "normal"

    # Rule-based override for very obvious, textbook patterns --
    # keeps clear-cut cases labeled intuitively even if the ML
    # models (trained on a fixed academic dataset) are unsure.
    if packets_per_sec > 3000:
        category = "DoS"
        is_attack = 1
    elif failed_logins >= 10:
        category = "R2L"
        is_attack = 1

    risk = RISK_LEVEL.get(category, "Medium" if is_attack else "None")
    friendly = FRIENDLY_NAME.get(category, category)

    return {
        "is_attack": bool(is_attack),
        "attack_type": friendly,
        "risk_level": risk,
        "confidence": round(float(rf_proba if is_attack else 1 - rf_proba), 3),
        "anomaly_flag": bool(anomaly_flag),
    }


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        ip = str(data["ip"])
        packets_per_sec = float(data["packets_per_sec"])
        failed_logins = int(data["failed_logins"])
        protocol = str(data["protocol"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Expected fields: ip, packets_per_sec, failed_logins, protocol"}), 400

    result = classify(packets_per_sec, failed_logins, protocol)

    db = get_db()
    db.execute(
        """INSERT INTO alerts
           (timestamp, ip, protocol, packets_per_sec, failed_logins,
            is_attack, attack_type, risk_level, confidence, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ip, protocol, packets_per_sec, failed_logins,
            int(result["is_attack"]), result["attack_type"], result["risk_level"],
            result["confidence"], int(result["anomaly_flag"]),
        ),
    )
    db.commit()

    return jsonify({
        "ip": ip,
        "protocol": protocol,
        "packets_per_sec": packets_per_sec,
        "failed_logins": failed_logins,
        **result,
    })


@app.route("/api/alerts")
def get_alerts():
    only_attacks = request.args.get("attacks_only", "false").lower() == "true"
    limit = int(request.args.get("limit", 50))
    db = get_db()
    query = "SELECT * FROM alerts"
    if only_attacks:
        query += " WHERE is_attack = 1"
    query += " ORDER BY id DESC LIMIT ?"
    rows = db.execute(query, (limit,)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/stats")
def get_stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
    attacks = db.execute("SELECT COUNT(*) c FROM alerts WHERE is_attack = 1").fetchone()["c"]
    by_type = db.execute(
        "SELECT attack_type, COUNT(*) c FROM alerts WHERE is_attack = 1 GROUP BY attack_type"
    ).fetchall()
    return jsonify({
        "total_traffic_samples": total,
        "total_alerts": attacks,
        "normal_traffic": total - attacks,
        "by_type": {r["attack_type"]: r["c"] for r in by_type},
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Generates one random traffic sample (normal or attack-like) and classifies it -- for demo purposes."""
    scenario = random.choice(["normal", "ddos", "bruteforce", "probe"])
    ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    if scenario == "normal":
        packets_per_sec = round(random.uniform(1, 60), 1)
        failed_logins = random.choice([0, 0, 0, 1])
        protocol = random.choice(["tcp", "udp"])
    elif scenario == "ddos":
        packets_per_sec = round(random.uniform(2000, 9000), 1)
        failed_logins = 0
        protocol = random.choice(["tcp", "udp", "icmp"])
    elif scenario == "bruteforce":
        packets_per_sec = round(random.uniform(5, 40), 1)
        failed_logins = random.randint(10, 60)
        protocol = "tcp"
    else:  # probe
        packets_per_sec = round(random.uniform(100, 400), 1)
        failed_logins = 0
        protocol = random.choice(["tcp", "icmp"])

    result = classify(packets_per_sec, failed_logins, protocol)
    db = get_db()
    db.execute(
        """INSERT INTO alerts
           (timestamp, ip, protocol, packets_per_sec, failed_logins,
            is_attack, attack_type, risk_level, confidence, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ip, protocol, packets_per_sec, failed_logins,
            int(result["is_attack"]), result["attack_type"], result["risk_level"],
            result["confidence"], int(result["anomaly_flag"]),
        ),
    )
    db.commit()

    return jsonify({
        "ip": ip, "protocol": protocol, "packets_per_sec": packets_per_sec,
        "failed_logins": failed_logins, **result,
    })


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
