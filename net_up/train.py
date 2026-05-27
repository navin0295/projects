"""
train.py
--------
Model Training Module for AI-Driven Network Behavior Analytics.

Trains:
  1. XGBoost classifier (primary supervised model)
  2. Autoencoder (optional deep-learning feature extractor / zero-day detector)

Both models are saved for use by the detection module.

Fixes applied:
  - Autoencoder threshold raised to 99th percentile (was 95th) to reduce
    false zero-day alerts on modern traffic
  - Added retrain_autoencoder_on_live_data() for retraining on captured
    real network traffic (run capture_normal.py first)
"""

import os
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── XGBoost ───────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── Optional deep-learning autoencoder ───────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed. Autoencoder training will be skipped.")

MODEL_DIR = "models"
DATA_DIR  = "data"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_splits():
    """Load preprocessed numpy arrays saved by preprocessing.py."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"[INFO] Loaded splits — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def print_metrics(y_true, y_pred, model_name: str):
    """Print evaluation metrics to stdout."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Full Report:\n{classification_report(y_true, y_pred)}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
    joblib.dump(metrics, os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}_metrics.pkl"))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. XGBoost Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate an XGBoost classifier."""
    print("\n[INFO] Training XGBoost classifier ...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred = model.predict(X_test)
    metrics = print_metrics(y_test, y_pred, "XGBoost")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] XGBoost model saved → {model_path}")
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int) -> Model:
    """Build a symmetric autoencoder for anomaly detection."""
    inp = Input(shape=(input_dim,))
    x   = Dense(128, activation="relu")(inp)
    x   = Dense(64,  activation="relu")(x)
    x   = Dense(32,  activation="relu")(x)   # bottleneck
    x   = Dense(64,  activation="relu")(x)
    x   = Dense(128, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)

    model = Model(inputs=inp, outputs=out, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X_train, X_test, y_train, y_test):
    """
    Train autoencoder only on NORMAL traffic (label=0).
    Anomalies will have higher reconstruction error.

    FIX: Threshold raised to 99th percentile (was 95th) to reduce
    false zero-day alerts on modern network traffic patterns.
    """
    if not TF_AVAILABLE:
        print("[WARN] Skipping autoencoder training (TensorFlow unavailable).")
        return None, None

    print("\n[INFO] Training Autoencoder on normal traffic ...")

    X_normal = X_train[y_train == 0]
    print(f"[INFO] Normal samples for AE training: {X_normal.shape[0]}")

    ae = build_autoencoder(X_train.shape[1])
    ae.summary()

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ae.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=512,
        validation_split=0.1,
        callbacks=[es],
        verbose=1,
    )

    recon_train = ae.predict(X_normal, verbose=0)
    mse_train   = np.mean(np.power(X_normal - recon_train, 2), axis=1)

    # FIX: Raised from 95th to 99th percentile to reduce false zero-day noise
    threshold = np.percentile(mse_train, 99)
    print(f"[INFO] Autoencoder anomaly threshold (99th pct): {threshold:.6f}")
    print(f"[INFO] Previous threshold (95th pct) would have been: "
          f"{np.percentile(mse_train, 95):.6f}")

    recon_test = ae.predict(X_test, verbose=0)
    mse_test   = np.mean(np.power(X_test - recon_test, 2), axis=1)
    y_pred_ae  = (mse_test > threshold).astype(int)
    metrics    = print_metrics(y_test, y_pred_ae, "Autoencoder")

    os.makedirs(MODEL_DIR, exist_ok=True)
    ae.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "ae_threshold.pkl"))
    print(f"[INFO] Autoencoder saved → {MODEL_DIR}/autoencoder.keras")
    return ae, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retrain Autoencoder on Live Captured Traffic (Optional, Recommended)
# ─────────────────────────────────────────────────────────────────────────────

def retrain_autoencoder_on_live_data():
    """
    Retrain the autoencoder using real normal traffic captured from your
    network by capture_normal.py.

    This is the best fix for false zero-day alerts because it trains the
    autoencoder on YOUR actual network patterns instead of 1999 KDD data.

    Run capture_normal.py first to collect live_normal_traffic, then call this.
    """
    if not TF_AVAILABLE:
        print("[WARN] TensorFlow not available. Cannot retrain autoencoder.")
        return

    live_path = os.path.join(DATA_DIR, "X_normal_live.npy")
    if not os.path.exists(live_path):
        print("[ERROR] Live normal data not found.")
        print("[ERROR] Run: sudo python capture_normal.py   first.")
        return

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print("[ERROR] Scaler not found. Run preprocessing.py first.")
        return

    X_normal_live = np.load(live_path)
    print(f"[INFO] Loaded live normal samples: {X_normal_live.shape}")

    scaler   = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_normal_live)

    ae = build_autoencoder(X_scaled.shape[1])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ae.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[es],
        verbose=1,
    )

    recon     = ae.predict(X_scaled, verbose=0)
    mse       = np.mean(np.power(X_scaled - recon, 2), axis=1)
    threshold = np.percentile(mse, 99)
    print(f"[INFO] New live-traffic threshold (99th pct): {threshold:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    ae.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "ae_threshold.pkl"))
    print("[INFO] Autoencoder retrained on live traffic and saved.")
    print("[INFO] Restart detect.py to use the new model.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_training(train_ae: bool = True):
    """Full training pipeline."""
    X_train, X_test, y_train, y_test = load_splits()

    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)

    ae_model = ae_metrics = None
    if train_ae:
        ae_model, ae_metrics = train_autoencoder(X_train, X_test, y_train, y_test)

    print("\n[INFO] Training complete. All models saved to 'models/'.")
    return xgb_model, ae_model


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--retrain-live":
        # Retrain autoencoder on live captured traffic
        # Usage: python train.py --retrain-live
        retrain_autoencoder_on_live_data()
    else:
        run_training(train_ae=TF_AVAILABLE)