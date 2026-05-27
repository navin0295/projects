"""
config.py
─────────
Central configuration for the Fingerprint Biometric Cryptosystem.
Edit this file to tune every aspect of the pipeline without touching source code.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DB_PATH         = BASE_DIR / "fingerprints.db"
MODEL_WEIGHTS   = BASE_DIR / "models" / "cnn_weights.weights.h5"
LOG_FILE        = BASE_DIR / "logs" / "system.log"

# Ensure directories exist at import time
for _d in [DATA_DIR, BASE_DIR / "models", BASE_DIR / "logs"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Image preprocessing ───────────────────────────────────────────────────────
IMG_SIZE        = (96, 96)          # (height, width) — must match model input
CLAHE_CLIP      = 3.0               # CLAHE clip limit
CLAHE_GRID      = (4, 4)            # CLAHE tile grid size
BLUR_KERNEL     = 3                 # Gaussian blur kernel size (must be odd)

# ── CNN Feature Extractor ─────────────────────────────────────────────────────
EMBEDDING_DIM   = 128               # Output embedding dimension
BACKBONE        = "mobilenetv2"     # Options: "mobilenetv2", "efficientnetv2s"
FINE_TUNE_LAYERS = 30               # Number of backbone layers to unfreeze for training
TRAIN_BATCH_SIZE = 16
TRAIN_STEPS     = 500               # Steps per epoch during triplet training
TRAIN_EPOCHS    = 10
TRIPLET_MARGIN  = 0.3               # Margin for triplet loss

# ── Feature Binarization ─────────────────────────────────────────────────────
BINARIZATION_METHOD = "median"      # "median" | "percentile" | "otsu"
RELIABLE_BIT_VARIANCE_THRESHOLD = 0.1   # Drop bits with intra-class variance above this

# ── Cryptography ──────────────────────────────────────────────────────────────
AES_KEY_SIZE    = 32                # 256-bit AES key
AES_MODE        = "CBC"             # CBC mode (swap to GCM for auth tag)
SALT            = b"FingerprintCryptosystem_SOCOFing_v1.0"
FUZZY_BITS      = EMBEDDING_DIM     # bits used in fuzzy extractor

# ── Authentication ────────────────────────────────────────────────────────────
COSINE_THRESHOLD     = 0.73         # Min cosine similarity to accept
HAMMING_THRESHOLD    = 0.35         # Max normalised Hamming distance to accept
MIN_ENROLL_IMAGES    = 1            # Minimum fingerprint images required for enrollment
RECOMMENDED_ENROLL   = 3           # Recommended for best key stability

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_GENUINE_N  = 50
EVAL_IMPOSTOR_N = 50

# ── Dataset (SOCOFing) ────────────────────────────────────────────────────────
SOCOFING_ROOT   = DATA_DIR / "SOCOFing"
SOCOFING_REAL   = SOCOFING_ROOT / "Real"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"            # DEBUG | INFO | WARNING | ERROR
LOG_TO_FILE     = True
LOG_TO_CONSOLE  = True
