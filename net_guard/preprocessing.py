"""
preprocessing.py
----------------
Data Preprocessing Module for AI-Driven Network Behavior Analytics.
Loads the KDD Cup 1999 dataset, encodes categorical features, normalizes,
and splits data for training and testing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import urllib.request

# ──────────────────────────────────────────────
# KDD Column Definitions
# ──────────────────────────────────────────────
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DATASET_URL = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
LOCAL_PATH = "kddcup.data_10_percent.gz"
MODEL_DIR = "models"


def download_dataset():
    """Download KDD dataset if not already present."""
    if not os.path.exists(LOCAL_PATH):
        print(f"[INFO] Downloading KDD dataset from {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, LOCAL_PATH)
        print("[INFO] Download complete.")
    else:
        print("[INFO] Dataset already exists locally.")


def load_dataset(path: str = LOCAL_PATH) -> pd.DataFrame:
    """Load KDD dataset from a local CSV/gz file."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, names=KDD_COLUMNS, compression="gzip")
    # Strip trailing period from labels (e.g., "normal." -> "normal")
    df["label"] = df["label"].str.rstrip(".")
    print(f"[INFO] Dataset loaded. Shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using LabelEncoder and
    binarize the label column (normal=0, anomaly=1).
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[INFO] Encoded '{col}': {list(le.classes_)[:5]} ...")

    # Binary label
    df["label"] = (df["label"] != "normal").astype(int)
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

    # Persist encoders
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    print(f"[INFO] Encoders saved to {MODEL_DIR}/label_encoders.pkl")
    return df


def normalize_features(df: pd.DataFrame):
    """
    Normalize numeric features using StandardScaler.
    Returns (X_train, X_test, y_train, y_test) and saves the scaler.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"[INFO] Scaler saved to {MODEL_DIR}/scaler.pkl")
    return X_scaled, y, feature_cols


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def run_preprocessing():
    """Full preprocessing pipeline — call this to prepare data."""
    download_dataset()
    df = load_dataset()
    df = encode_features(df)
    X, y, feature_cols = normalize_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Optionally persist splits for reuse
    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    joblib.dump(feature_cols, "data/feature_cols.pkl")
    print("[INFO] Preprocessed data saved to data/")
    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    run_preprocessing()
