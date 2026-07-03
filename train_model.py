"""
IDS Model Training Script
==========================
Trains three models on the NSL-KDD network intrusion dataset:
  - Random Forest      (supervised, binary normal/attack + attack-type)
  - Decision Tree      (supervised, binary normal/attack)
  - Isolation Forest   (unsupervised anomaly detector, trained on normal traffic only)

Run:  python3 train_model.py
Produces artifacts in ./models/
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Standard NSL-KDD attack -> category mapping
ATTACK_CATEGORY = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}
RISK_LEVEL = {"normal": "None", "Probe": "Medium", "DoS": "High", "R2L": "High", "U2R": "Critical"}


def load_dataset(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["attack_category"] = df["label"].apply(
        lambda x: "normal" if x == "normal" else ATTACK_CATEGORY.get(x, "Other")
    )
    df["binary_label"] = (df["label"] != "normal").astype(int)  # 0 = normal, 1 = attack
    return df


def build_encoders(train_df):
    encoders = {}
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        le.fit(train_df[col])
        encoders[col] = le
    cat_le = LabelEncoder()
    cat_le.fit(train_df["attack_category"])
    encoders["attack_category"] = cat_le
    return encoders


def encode(df, encoders):
    df = df.copy()
    for col in ["protocol_type", "service", "flag"]:
        le = encoders[col]
        # map unseen categories (e.g. in test set) to a fallback value
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    return df


def main():
    print("Loading NSL-KDD data...")
    train_df = load_dataset(os.path.join(DATA_DIR, "KDDTrain.txt"))
    test_df = load_dataset(os.path.join(DATA_DIR, "KDDTest.txt"))
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    encoders = build_encoders(train_df)
    train_enc = encode(train_df, encoders)
    test_enc = encode(test_df, encoders)

    feature_cols = [c for c in COLUMNS if c not in ("label", "difficulty")]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_enc[feature_cols])
    X_test = scaler.transform(test_enc[feature_cols])

    y_train_bin = train_enc["binary_label"]
    y_test_bin = test_enc["binary_label"]
    y_train_cat = encoders["attack_category"].transform(train_enc["attack_category"])
    y_test_cat = encoders["attack_category"].transform(
        test_enc["attack_category"].apply(
            lambda v: v if v in encoders["attack_category"].classes_ else "Other"
        )
    )

    # ---------- Random Forest (binary: normal vs attack) ----------
    print("\nTraining Random Forest (binary normal/attack)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_bin)
    rf_pred = rf.predict(X_test)
    print("Random Forest accuracy:", accuracy_score(y_test_bin, rf_pred))
    print(classification_report(y_test_bin, rf_pred, target_names=["normal", "attack"]))

    # ---------- Random Forest (multi-class: attack category) ----------
    print("\nTraining Random Forest (multi-class attack category)...")
    rf_cat = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    rf_cat.fit(X_train, y_train_cat)
    rf_cat_pred = rf_cat.predict(X_test)
    print("Category accuracy:", accuracy_score(y_test_cat, rf_cat_pred))

    # ---------- Decision Tree (binary) ----------
    print("\nTraining Decision Tree (binary normal/attack)...")
    dt = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt.fit(X_train, y_train_bin)
    dt_pred = dt.predict(X_test)
    print("Decision Tree accuracy:", accuracy_score(y_test_bin, dt_pred))

    # ---------- Isolation Forest (unsupervised anomaly detection) ----------
    print("\nTraining Isolation Forest (unsupervised, normal traffic only)...")
    normal_only = X_train[y_train_bin == 0]
    iso = IsolationForest(n_estimators=150, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(normal_only)
    iso_pred_raw = iso.predict(X_test)  # 1 = normal, -1 = anomaly
    iso_pred = (iso_pred_raw == -1).astype(int)
    print("Isolation Forest accuracy vs known labels:", accuracy_score(y_test_bin, iso_pred))
    print(confusion_matrix(y_test_bin, iso_pred))

    # ---------- Save everything ----------
    print("\nSaving model artifacts to ./models ...")
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_binary.pkl"))
    joblib.dump(rf_cat, os.path.join(MODEL_DIR, "random_forest_category.pkl"))
    joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree_binary.pkl"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(MODEL_DIR, "risk_levels.json"), "w") as f:
        json.dump(RISK_LEVEL, f)

    # Save a "typical normal" baseline row -- used by the API to fill in
    # feature values we don't collect from the simplified real-time input
    # (IP / packets-per-sec / failed logins / protocol).
    baseline = train_enc[train_enc["binary_label"] == 0][feature_cols].median().to_dict()
    with open(os.path.join(MODEL_DIR, "baseline_row.json"), "w") as f:
        json.dump(baseline, f)

    print("Done. Models saved in ./models/")


if __name__ == "__main__":
    main()
