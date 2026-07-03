"""
AI-Based Intrusion Detection System - Flask Backend v2
=========================================================
Same architecture as v1, retrained on real CICIDS2017 network flow data:
  Incoming traffic sample -> feature construction -> ML models
  -> alert classification -> SQLite storage -> dashboard / API

IMPORTANT SCHEMA CHANGE FROM v1:
This CICIDS2017 release has NO protocol/failed-login columns -- it's
pure network-flow statistics (from CICFlowMeter) with IPs already
stripped out. So the simplified real-time input is now:
  - Source IP        (label only, for the dashboard -- not a model feature,
                       since this dataset doesn't include IP addresses)
  - Destination Port  (real feature)
  - Flow Duration     (real feature, microseconds)
  - Total Packets     (split into Fwd/Bwd -- real features)
  - Packets/sec       (maps DIRECTLY to the real 'Flow Packets/s' column --
                       no approximation needed, unlike the NSL-KDD version)

Run:  python3 app_v2.py
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

MODEL_DIR = "models_v2"
DB_PATH = "ids_alerts_v2.db"

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

DEMO_SAMPLES = pd.read_csv(os.path.join(MODEL_DIR, "demo_samples.csv"))

FRIENDLY_NAME = {
    "normal": "Normal Traffic",
    "DoS": "Denial of Service (Hulk/GoldenEye/Slowloris-class)",
    "DDoS": "Distributed Denial of Service",
    "Probe": "Port Scan / Reconnaissance",
    "Botnet": "Botnet Activity",
    "Infiltration": "Network Infiltration",
    "Heartbleed": "Heartbleed Exploit Attempt",
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
            dest_port INTEGER NOT NULL,
            flow_duration REAL NOT NULL,
            total_packets INTEGER NOT NULL,
            packets_per_sec REAL NOT NULL,
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
# Feature construction: map the simplified real-time input onto
# the full 78-feature CICFlowMeter vector the models expect.
#
# Fields that map DIRECTLY (no approximation, these are real columns):
#   Destination Port, Flow Duration, Total Fwd/Backward Packets,
#   Flow Packets/s
#
# Everything else is filled from the "typical normal flow" baseline,
# then lightly adjusted so a high packet rate also nudges related
# fields (packet size, flags) in a DDoS/flood-like direction --
# mirroring what a real flood actually looks like in CICFlowMeter output.
# ---------------------------------------------------------------
def build_feature_vector(dest_port, flow_duration, total_packets, packets_per_sec):
    row = dict(BASELINE)

    row["Destination Port"] = dest_port
    row["Flow Duration"] = flow_duration
    row["Flow Packets/s"] = packets_per_sec

    fwd_packets = max(1, int(total_packets * 0.7))
    bwd_packets = max(0, total_packets - fwd_packets)
    row["Total Fwd Packets"] = fwd_packets
    row["Total Backward Packets"] = bwd_packets
    row["Fwd Packets/s"] = packets_per_sec * 0.7
    row["Bwd Packets/s"] = packets_per_sec * 0.3
    row["Subflow Fwd Packets"] = fwd_packets
    row["Subflow Bwd Packets"] = bwd_packets

    # Very high packet rate over a short duration looks like a
    # flood (DoS/DDoS): many small packets, low per-packet bytes,
    # elevated SYN flag count.
    if packets_per_sec > 500:
        row["SYN Flag Count"] = min(fwd_packets, 100)
        row["Fwd Packet Length Mean"] = 40
        row["Packet Length Mean"] = 40
        row["Flow Bytes/s"] = packets_per_sec * 40

    vector = [row[col] for col in FEATURE_COLS]
    return pd.DataFrame([vector], columns=FEATURE_COLS)


def classify_vector(X_df, packets_per_sec=None, dest_port=None):
    """Runs a full 78-feature vector (as a 1-row DataFrame) through all three
    models. Used both for the heuristic manual-input vector and for real
    held-out demo rows."""
    X_scaled = scaler.transform(X_df)

    rf_pred = rf_binary.predict(X_scaled)[0]
    rf_proba = rf_binary.predict_proba(X_scaled)[0][1]
    dt_pred = dt_binary.predict(X_scaled)[0]
    iso_pred = iso_forest.predict(X_scaled)[0]
    anomaly_flag = 1 if iso_pred == -1 else 0

    # Require at least 2 of 3 models to agree before raising an alert.
    # Isolation Forest alone is known to be unreliable in this build (see
    # README) due to the class-imbalance introduced by downsampling, so a
    # lone anomaly_flag from it should not be enough to trigger an alert --
    # especially important on real live traffic, which differs statistically
    # from the 2017 training data (different OS/TLS versions, etc) and can
    # otherwise trip Isolation Forest into false positives.
    votes = int(rf_pred) + int(dt_pred) + int(anomaly_flag)
    is_attack = 1 if votes >= 2 else 0

    if is_attack:
        cat_idx = rf_category.predict(X_scaled)[0]
        category = encoders["attack_category"].inverse_transform([cat_idx])[0]
        if category == "normal":
            category = "Other"
    else:
        category = "normal"

    # Rule-based override for the most obvious textbook cases -- only
    # applied when we actually know packets_per_sec/dest_port (i.e. for
    # the simplified manual-entry path, not for real demo rows, which
    # the model already classifies correctly on their own).
    if packets_per_sec is not None:
        if packets_per_sec > 5000:
            category = "DDoS"
            is_attack = 1
        elif dest_port in (21, 22, 23, 3389) and packets_per_sec > 200:
            category = "Botnet"
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


def classify(dest_port, flow_duration, total_packets, packets_per_sec):
    X = build_feature_vector(dest_port, flow_duration, total_packets, packets_per_sec)
    return classify_vector(X, packets_per_sec=packets_per_sec, dest_port=dest_port)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard_v2.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        ip = str(data["ip"])
        dest_port = int(data["dest_port"])
        flow_duration = float(data["flow_duration"])
        total_packets = int(data["total_packets"])
        packets_per_sec = float(data["packets_per_sec"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Expected fields: ip, dest_port, flow_duration, total_packets, packets_per_sec"}), 400

    result = classify(dest_port, flow_duration, total_packets, packets_per_sec)

    db = get_db()
    db.execute(
        """INSERT INTO alerts
           (timestamp, ip, dest_port, flow_duration, total_packets,
            packets_per_sec, is_attack, attack_type, risk_level, confidence, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ip, dest_port, flow_duration, total_packets, packets_per_sec,
            int(result["is_attack"]), result["attack_type"], result["risk_level"],
            result["confidence"], int(result["anomaly_flag"]),
        ),
    )
    db.commit()

    return jsonify({
        "ip": ip, "dest_port": dest_port, "flow_duration": flow_duration,
        "total_packets": total_packets, "packets_per_sec": packets_per_sec,
        **result,
    })


@app.route("/api/predict_full", methods=["POST"])
def predict_full():
    """Accepts a FULL 78-feature real flow (e.g. from the local packet-capture
    tool) rather than the simplified 4-field form. No approximation --
    classifies the real computed flow statistics directly."""
    data = request.get_json(force=True)
    try:
        ip = str(data.get("ip", "unknown"))
        features = data["features"]
        row = {col: float(features.get(col, BASELINE.get(col, 0))) for col in FEATURE_COLS}
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Bad payload: {e}"}), 400

    X_df = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    result = classify_vector(X_df)

    dest_port = int(row.get("Destination Port", 0))
    flow_duration = float(row.get("Flow Duration", 0))
    total_packets = int(row.get("Total Fwd Packets", 0) + row.get("Total Backward Packets", 0))
    packets_per_sec = float(row.get("Flow Packets/s", 0))

    db = get_db()
    db.execute(
        """INSERT INTO alerts
           (timestamp, ip, dest_port, flow_duration, total_packets,
            packets_per_sec, is_attack, attack_type, risk_level, confidence, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ip, dest_port, flow_duration, total_packets, packets_per_sec,
            int(result["is_attack"]), result["attack_type"], result["risk_level"],
            result["confidence"], int(result["anomaly_flag"]),
        ),
    )
    db.commit()

    return jsonify({
        "ip": ip, "dest_port": dest_port, "flow_duration": flow_duration,
        "total_packets": total_packets, "packets_per_sec": packets_per_sec, **result,
    })


@app.route("/api/simulate_category", methods=["POST"])
def simulate_category():
    """Like /api/simulate, but lets the caller pick a specific category --
    useful for reliably demoing rare categories (Infiltration, Heartbleed)
    that would otherwise only show up ~2-8% of the time with random draws."""
    data = request.get_json(force=True)
    category = data.get("category", "")

    available = DEMO_SAMPLES["attack_category"].unique().tolist()
    if category not in available:
        return jsonify({"error": f"Unknown category. Choose from: {available}"}), 400

    subset = DEMO_SAMPLES[DEMO_SAMPLES["attack_category"] == category]
    row = subset.sample(n=1).iloc[0]
    ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    X_df = pd.DataFrame([row[FEATURE_COLS].values], columns=FEATURE_COLS)
    result = classify_vector(X_df)

    dest_port = int(row["Destination Port"])
    flow_duration = float(row["Flow Duration"])
    total_packets = int(row["Total Fwd Packets"] + row["Total Backward Packets"])
    packets_per_sec = float(row["Flow Packets/s"])

    db = get_db()
    db.execute(
        """INSERT INTO alerts
           (timestamp, ip, dest_port, flow_duration, total_packets,
            packets_per_sec, is_attack, attack_type, risk_level, confidence, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ip, dest_port, flow_duration, total_packets, packets_per_sec,
            int(result["is_attack"]), result["attack_type"], result["risk_level"],
            result["confidence"], int(result["anomaly_flag"]),
        ),
    )
    db.commit()

    return jsonify({
        "ip": ip, "dest_port": dest_port, "flow_duration": flow_duration,
        "total_packets": total_packets, "packets_per_sec": packets_per_sec,
        "true_label": row["attack_category"], **result,
    })


@app.route("/api/categories")
def get_categories():
    """Returns the list of categories available for forced simulation."""
    return jsonify(sorted(DEMO_SAMPLES["attack_category"].unique().tolist()))


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
    """Picks one real, held-out CICIDS2017 flow (never seen during training)
    and classifies it. This reflects genuine model behavior across all
    attack types -- unlike the manual /api/predict form, which can only
    approximate a full flow from 4 numbers."""
    row = DEMO_SAMPLES.sample(n=1).iloc[0]
    ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    X_df = pd.DataFrame([row[FEATURE_COLS].values], columns=FEATURE_COLS)
    result = classify_vector(X_df)  # no rule overrides -- pure model output on real data

    dest_port = int(row["Destination Port"])
    flow_duration = float(row["Flow Duration"])
    total_packets = int(row["Total Fwd Packets"] + row["Total Backward Packets"])
    packets_per_sec = float(row["Flow Packets/s"])

    db = get_db()
    db.execute(
        """INSERT INTO alerts
           (timestamp, ip, dest_port, flow_duration, total_packets,
            packets_per_sec, is_attack, attack_type, risk_level, confidence, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ip, dest_port, flow_duration, total_packets, packets_per_sec,
            int(result["is_attack"]), result["attack_type"], result["risk_level"],
            result["confidence"], int(result["anomaly_flag"]),
        ),
    )
    db.commit()

    return jsonify({
        "ip": ip, "dest_port": dest_port, "flow_duration": flow_duration,
        "total_packets": total_packets, "packets_per_sec": packets_per_sec,
        "true_label": row["attack_category"], **result,
    })


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
