"""
IDS Model Training Script v2 -- Real CICIDS2017 Data
=======================================================
Trains on genuine 2017 network flow data (CICFlowMeter features) covering:
  DoS (Hulk, GoldenEye, slowloris, Slowhttptest), DDoS, PortScan (Probe),
  Bot (Botnet), Infiltration, Heartbleed -- plus BENIGN traffic.

Unlike NSL-KDD, this dataset has NO protocol_type/service/flag categorical
columns and NO login-attempt fields -- it's pure network-flow statistics
extracted from real packet captures. All 78 features are numeric.

Run:  python3 train_model_v2.py
Produces artifacts in ./models_v2/
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "data_v2/cicids2017_sample.csv"
MODEL_DIR = "models_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

# Real CICIDS2017 label -> category mapping
ATTACK_CATEGORY = {
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Probe",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}
RISK_LEVEL = {
    "normal": "None", "Probe": "Medium", "DoS": "High", "DDoS": "High",
    "Botnet": "High", "Infiltration": "Critical", "Heartbleed": "Critical",
}


def main():
    print("Loading CICIDS2017 sample...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = [c.strip() for c in df.columns]  # CICFlowMeter columns have stray leading spaces
    print(f"Raw rows: {len(df):,}, columns: {len(df.columns)}")

    # Drop the 1 row with missing label
    df = df.dropna(subset=["Label"])

    # Destination Port loaded as string in this release -- convert to numeric
    df["Destination Port"] = pd.to_numeric(df["Destination Port"], errors="coerce")

    feature_cols = [c for c in df.columns if c != "Label"]

    # Known CICIDS2017 data-quality issue: Flow Bytes/s and Flow Packets/s
    # can be +/-infinity when Flow Duration is 0. Replace with NaN, then
    # impute using the column median (computed AFTER the inf->NaN swap).
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df["attack_category"] = df["Label"].apply(
        lambda x: "normal" if x == "BENIGN" else ATTACK_CATEGORY.get(x, "Other")
    )
    df["binary_label"] = (df["Label"] != "BENIGN").astype(int)

    print("\nLabel distribution:")
    print(df["Label"].value_counts())
    print("\nCategory distribution:")
    print(df["attack_category"].value_counts())

    cat_le = LabelEncoder()
    y_cat_all = cat_le.fit_transform(df["attack_category"])

    X = df[feature_cols]
    y_bin = df["binary_label"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat = train_test_split(
        X, y_bin, y_cat_all, test_size=0.25, random_state=42, stratify=y_bin
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Random Forest (binary, class-weight balanced since
    # attacks now outnumber benign traffic in this sample) ----------
    print("\nTraining Random Forest (binary normal/attack)...")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=20, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_scaled, y_train_bin)
    rf_pred = rf.predict(X_test_scaled)
    print("Random Forest accuracy:", accuracy_score(y_test_bin, rf_pred))
    print(classification_report(y_test_bin, rf_pred, target_names=["normal", "attack"]))

    # ---------- Random Forest (multi-class attack category) ----------
    print("\nTraining Random Forest (multi-class attack category)...")
    rf_cat = RandomForestClassifier(
        n_estimators=150, max_depth=20, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    rf_cat.fit(X_train_scaled, y_train_cat)
    rf_cat_pred = rf_cat.predict(X_test_scaled)
    print("Category accuracy:", accuracy_score(y_test_cat, rf_cat_pred))
    print(classification_report(
        y_test_cat, rf_cat_pred,
        target_names=cat_le.inverse_transform(sorted(set(y_test_cat))),
        zero_division=0,
    ))

    # ---------- Decision Tree (binary) ----------
    print("\nTraining Decision Tree (binary normal/attack)...")
    dt = DecisionTreeClassifier(max_depth=15, random_state=42, class_weight="balanced")
    dt.fit(X_train_scaled, y_train_bin)
    dt_pred = dt.predict(X_test_scaled)
    print("Decision Tree accuracy:", accuracy_score(y_test_bin, dt_pred))

    # ---------- Isolation Forest (unsupervised, normal traffic only) ----------
    print("\nTraining Isolation Forest (unsupervised, normal traffic only)...")
    normal_only = X_train_scaled[y_train_bin == 0]
    iso = IsolationForest(n_estimators=150, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(normal_only)
    iso_pred_raw = iso.predict(X_test_scaled)
    iso_pred = (iso_pred_raw == -1).astype(int)
    print("Isolation Forest accuracy vs known labels:", accuracy_score(y_test_bin, iso_pred))

    # ---------- Save everything ----------
    print("\nSaving model artifacts to ./models_v2 ...")
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_binary.pkl"))
    joblib.dump(rf_cat, os.path.join(MODEL_DIR, "random_forest_category.pkl"))
    joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree_binary.pkl"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump({"attack_category": cat_le}, os.path.join(MODEL_DIR, "encoders.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(MODEL_DIR, "risk_levels.json"), "w") as f:
        json.dump(RISK_LEVEL, f)

    # Baseline "typical normal" row -- used by the API to fill in feature
    # values we don't collect from the simplified real-time input.
    baseline = X_train[y_train_bin == 0].median().to_dict()
    with open(os.path.join(MODEL_DIR, "baseline_row.json"), "w") as f:
        json.dump(baseline, f)

    # Real held-out demo rows, per attack category, for honest live
    # simulation. These are genuine unseen flow vectors -- unlike the
    # simplified manual-entry form, they exercise the full 78-feature
    # signature each attack type actually has, so the dashboard's
    # "Simulate" button reflects true model behavior instead of a
    # 4-field approximation.
    test_df = X_test.copy()
    test_df["attack_category"] = cat_le.inverse_transform(y_test_cat)
    demo_rows = []
    for category in test_df["attack_category"].unique():
        subset = test_df[test_df["attack_category"] == category]
        n = min(25, len(subset))
        demo_rows.append(subset.sample(n=n, random_state=42))
    demo_df = pd.concat(demo_rows, ignore_index=True)
    demo_df.to_csv(os.path.join(MODEL_DIR, "demo_samples.csv"), index=False)
    print(f"Saved {len(demo_df)} real held-out demo rows across "
          f"{test_df['attack_category'].nunique()} categories to demo_samples.csv")

    print("Done. Models saved in ./models_v2/")


if __name__ == "__main__":
    main()
