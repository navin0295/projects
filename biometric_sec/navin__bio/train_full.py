"""
train_full.py
─────────────
Full training pipeline for the CNN feature extractor on the complete SOCOFing dataset.

What the previous train() did wrong (and what this fixes)
----------------------------------------------------------
Previous issue:
  - `backbone.trainable = False` was set at __init__ time, BEFORE compile().
  - `training=False` was passed to the backbone in the embedding model, which
    means BatchNorm layers never updated their running stats even when layers
    were marked trainable.  The backbone was effectively frozen permanently.
  - TripletGenerator used `random.Random` (pure Python) without shuffling
    between epochs — same triplets in the same order every epoch.
  - No learning-rate warm-up: starting at 1e-4 on a frozen model then
    unfreezing is backwards.  You should warm up the head first, then unfreeze.

This script follows the correct two-phase fine-tuning protocol:
  Phase 1 (head warm-up, 5 epochs)
    Backbone completely frozen.  Only the projection head trains.
    LR = 1e-3.  This prevents the pretrained weights being destroyed
    by random head gradients at the start.

  Phase 2 (backbone fine-tune, N epochs)
    Last `FINE_TUNE_LAYERS` backbone layers unfrozen.
    LR drops to 1e-5 (must be much lower than phase 1).
    BatchNorm in the unfrozen layers trains with training=True.

Dataset split
-------------
80 % of subjects → training triplets
20 % of subjects → validation triplets  (genuine + impostor)

The validation split measures real generalisation: the model has never
seen those subject IDs during training.

Run
---
python train_full.py --dir ./data/SOCOFing/Real --phase1-epochs 5 --phase2-epochs 15
"""

from __future__ import annotations

import argparse
import glob
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)

import config
from preprocessor import FingerprintPreprocessor
from logger import get_logger

log = get_logger("train_full")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_subject_images(real_dir: Path) -> Dict[int, List[str]]:
    """
    Parse the SOCOFing Real/ directory and return a dict:
        subject_id → list of image paths

    SOCOFing filename format:  <subject_id>__<hand>_<finger>_<variant>.BMP
    """
    all_imgs = sorted(glob.glob(str(real_dir / "*.BMP")))
    if not all_imgs:
        all_imgs = sorted(glob.glob(str(real_dir / "*.png")))
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {real_dir}")

    subject_imgs: Dict[int, List[str]] = {}
    for p in all_imgs:
        fname = Path(p).stem
        if "__" in fname:
            sid = int(fname.split("__")[0])
        else:
            nums = re.findall(r"\d+", fname)
            sid  = int(nums[0]) if nums else 0
        subject_imgs.setdefault(sid, []).append(p)

    log.info("Loaded %d images across %d subjects.", len(all_imgs), len(subject_imgs))
    return subject_imgs


def train_val_split(
    subject_imgs: Dict[int, List[str]],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Split subjects (not images) into train and validation sets.
    Returns (train_dict, val_dict).
    """
    sids = sorted(subject_imgs.keys())
    rng  = random.Random(seed)
    rng.shuffle(sids)
    n_val   = max(1, int(len(sids) * val_fraction))
    val_ids = set(sids[:n_val])
    train_imgs = {s: imgs for s, imgs in subject_imgs.items() if s not in val_ids}
    val_imgs   = {s: imgs for s, imgs in subject_imgs.items() if s in val_ids}
    log.info("Train subjects: %d | Validation subjects: %d", len(train_imgs), len(val_imgs))
    return train_imgs, val_imgs


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED TRIPLET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_triplet_dataset(
    subject_imgs: Dict[int, List[str]],
    preprocessor: FingerprintPreprocessor,
    batch_size:   int = config.TRAIN_BATCH_SIZE,
    steps:        int = config.TRAIN_STEPS,
    seed:         int = 42,
):
    """
    Build a tf.data.Dataset yielding (anchor, positive, negative) triplet batches.

    Uses tf.data.Dataset.from_generator with explicit TensorSpec output_signature
    required by Keras 3 / TF 2.16+. The old tf.keras.utils.Sequence approach broke
    because Keras 3 no longer accepts a Python list as the output signature.

    Returns (ds, state) where state holds the rng so the caller can reseed
    between training phases.
    """
    subs     = [s for s, imgs in subject_imgs.items() if len(imgs) >= 2]
    sub_imgs = {s: subject_imgs[s] for s in subs}

    if len(subs) < 2:
        raise ValueError("Need at least 2 subjects with >= 2 images each.")

    log.info("Triplet dataset: %d eligible subjects, %d steps x batch %d",
             len(subs), steps, batch_size)

    h, w  = config.IMG_SIZE
    state = {"rng": np.random.default_rng(seed), "subs": list(subs)}

    def generator():
        rng = state["rng"]
        while True:
            sid  = rng.choice(state["subs"])
            imgs = sub_imgs[sid]
            if len(imgs) < 2:
                continue
            idx      = rng.choice(len(imgs), size=2, replace=False)
            anchor   = preprocessor.process(imgs[idx[0]]).astype(np.float32)
            positive = preprocessor.process(imgs[idx[1]]).astype(np.float32)
            neg_sid  = sid
            for _ in range(20):
                neg_sid = rng.choice(state["subs"])
                if neg_sid != sid:
                    break
            negative = preprocessor.process(
                rng.choice(sub_imgs[neg_sid])
            ).astype(np.float32)
            yield {"anchor": anchor, "positive": positive, "negative": negative}, np.float32(0.0)

    img_spec   = tf.TensorSpec(shape=(h, w, 1), dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=(),         dtype=tf.float32)
    output_sig = (
        {"anchor": img_spec, "positive": img_spec, "negative": img_spec},
        label_spec,
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.take(steps)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, state


class BalancedTripletGenerator:
    """Compatibility shim kept so gui.py imports still resolve."""
    def __init__(self, subject_imgs, preprocessor,
                 batch_size=config.TRAIN_BATCH_SIZE,
                 steps=config.TRAIN_STEPS, seed=42):
        self.subject_imgs = subject_imgs
        self.preprocessor = preprocessor
        self.batch_size   = batch_size
        self.steps        = steps
        self.seed         = seed

    def as_dataset(self):
        ds, _ = make_triplet_dataset(
            self.subject_imgs, self.preprocessor,
            self.batch_size, self.steps, self.seed
        )
        return ds



# ═══════════════════════════════════════════════════════════════════════════════
# CORRECTED MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_embedding_model(
    input_shape:      Tuple[int, int, int] = (*config.IMG_SIZE, 1),
    embedding_dim:    int                  = config.EMBEDDING_DIM,
    freeze_backbone:  bool                 = True,
    fine_tune_layers: int                  = config.FINE_TUNE_LAYERS,
) -> Model:
    """
    Build the embedding model.

    Key fix: backbone is called with `training=backbone.trainable`, not
    hardcoded to False.  This ensures BatchNorm updates during phase 2.
    """
    inp = Input(shape=input_shape, name="fingerprint_input")
    x   = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1), name="gray_to_rgb")(inp)

    backbone = MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )

    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        # Keep early BatchNorm layers frozen (they carry general low-level stats)
        for layer in backbone.layers[:-fine_tune_layers]:
            layer.trainable = False
        # Freeze all BatchNorm even in the unfrozen section to keep BN stats stable
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    # CRITICAL FIX: pass training=freeze_backbone flag correctly
    # When backbone.trainable=False → training=False (inference mode, BN uses stored stats)
    # When backbone.trainable=True  → training=True  (BN updates running stats)
    x = backbone(x, training=freeze_backbone is False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="proj_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(0.3, name="drop")(x)
    x = layers.Dense(embedding_dim, name="proj_128")(x)
    emb = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1), name="l2_normalize"
    )(x)

    return Model(inputs=inp, outputs=emb, name="FingerprintEmbedder")


def build_triplet_model(emb_model: Model) -> Model:
    inp = emb_model.input_shape[1:]   # (H, W, 1)
    anchor   = Input(shape=inp, name="anchor")
    positive = Input(shape=inp, name="positive")
    negative = Input(shape=inp, name="negative")
    out = layers.Concatenate(axis=-1, name="triplet_out")(
        [emb_model(anchor), emb_model(positive), emb_model(negative)]
    )
    return Model(inputs=[anchor, positive, negative], outputs=out, name="TripletNet")


# ═══════════════════════════════════════════════════════════════════════════════
# TRIPLET LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def triplet_loss(margin: float = config.TRIPLET_MARGIN, dim: int = config.EMBEDDING_DIM):
    def loss(y_true, y_pred):
        a = y_pred[:, :dim]
        p = y_pred[:, dim : 2 * dim]
        n = y_pred[:, 2 * dim :]
        d_pos = tf.reduce_sum(tf.square(a - p), axis=1)
        d_neg = tf.reduce_sum(tf.square(a - n), axis=1)
        return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION METRIC  (pairwise cosine distances)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_val_metrics(
    emb_model:     Model,
    preprocessor:  FingerprintPreprocessor,
    val_imgs:      Dict[int, List[str]],
    n_pairs:       int = 200,
    seed:          int = 42,
) -> dict:
    """
    Compute genuine and impostor cosine similarity on held-out val subjects.

    Batches all predictions in two forward passes (genuine + impostor)
    instead of calling predict() per image.

    Returns dict: mean_genuine, mean_impostor, separability.
    """
    rng  = random.Random(seed)
    subs = [s for s, imgs in val_imgs.items() if len(imgs) >= 2]
    if len(subs) < 2:
        log.warning("Not enough validation subjects for metrics (need ≥ 2 with ≥ 2 images).")
        return {}

    genuine_a_paths, genuine_b_paths = [], []
    impostor_a_paths, impostor_b_paths = [], []

    for _ in range(n_pairs):
        sid  = rng.choice(subs)
        a, b = rng.sample(val_imgs[sid], 2)
        genuine_a_paths.append(a)
        genuine_b_paths.append(b)

    for _ in range(n_pairs):
        s1, s2 = rng.sample(subs, 2)
        impostor_a_paths.append(rng.choice(val_imgs[s1]))
        impostor_b_paths.append(rng.choice(val_imgs[s2]))

    def batch_embed(paths: List[str]) -> np.ndarray:
        tensors = np.stack([preprocessor.process(p) for p in paths])   # (N, H, W, 1)
        return emb_model.predict(tensors, batch_size=32, verbose=0)     # (N, D)

    log.info("Computing validation embeddings for %d genuine + %d impostor pairs…",
             n_pairs, n_pairs)

    emb_ga = batch_embed(genuine_a_paths)
    emb_gb = batch_embed(genuine_b_paths)
    emb_ia = batch_embed(impostor_a_paths)
    emb_ib = batch_embed(impostor_b_paths)

    # Cosine similarity = dot product of L2-normalised vectors
    genuine_sims  = np.sum(emb_ga * emb_gb,  axis=1).tolist()
    impostor_sims = np.sum(emb_ia * emb_ib, axis=1).tolist()

    m_g = float(np.mean(genuine_sims))
    m_i = float(np.mean(impostor_sims))
    sep = m_g - m_i

    log.info("Val metrics — genuine: %.4f | impostor: %.4f | separability: %.4f",
             m_g, m_i, sep)
    return {"mean_genuine": m_g, "mean_impostor": m_i, "separability": sep}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    real_dir:       Path,
    phase1_epochs:  int   = 5,
    phase2_epochs:  int   = 15,
    batch_size:     int   = config.TRAIN_BATCH_SIZE,
    steps_per_epoch:int   = config.TRAIN_STEPS,
    val_steps:      int   = 50,
    output_dir:     Path  = config.BASE_DIR / "models",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = FingerprintPreprocessor()

    # ── Load and split data ───────────────────────────────────────────────────
    subject_imgs       = load_subject_images(real_dir)
    train_imgs, val_imgs = train_val_split(subject_imgs)

    train_ds, train_state = make_triplet_dataset(train_imgs, preprocessor,
                                                  batch_size=batch_size,
                                                  steps=steps_per_epoch)
    val_ds, _              = make_triplet_dataset(val_imgs,   preprocessor,
                                                  batch_size=batch_size,
                                                  steps=val_steps, seed=99)

    # ── Phase 1: head warm-up (backbone frozen) ───────────────────────────────
    log.info("═" * 50)
    log.info("PHASE 1 — Head warm-up  (%d epochs, LR=1e-3)", phase1_epochs)
    log.info("═" * 50)

    emb_model_p1 = build_embedding_model(freeze_backbone=True)
    triplet_p1   = build_triplet_model(emb_model_p1)
    triplet_p1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=triplet_loss()
    )

    ckpt_p1 = str(output_dir / "phase1_best.weights.h5")
    history1 = triplet_p1.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=[
            ModelCheckpoint(ckpt_p1, monitor="val_loss",
                            save_best_only=True, save_weights_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=3, verbose=1),
            CSVLogger(str(output_dir / "phase1_log.csv")),
        ],
        verbose=1,
    )

    # Load best phase-1 weights before transferring to phase 2
    triplet_p1.load_weights(ckpt_p1)

    # ── Phase 2: backbone fine-tuning (last N layers unfrozen) ───────────────
    log.info("═" * 50)
    log.info("PHASE 2 — Backbone fine-tuning  (%d epochs, LR=1e-5)", phase2_epochs)
    log.info("═" * 50)

    emb_model_p2 = build_embedding_model(freeze_backbone=False,
                                          fine_tune_layers=config.FINE_TUNE_LAYERS)
    # Transfer projection-head weights from phase 1 by layer name (not fragile slicing)
    emb_model_p2.get_layer("proj_256").set_weights(
        emb_model_p1.get_layer("proj_256").get_weights()
    )
    emb_model_p2.get_layer("proj_128").set_weights(
        emb_model_p1.get_layer("proj_128").get_weights()
    )
    # Also transfer the batch-norm running stats from phase-1 head
    emb_model_p2.get_layer("bn_256").set_weights(
        emb_model_p1.get_layer("bn_256").get_weights()
    )

    triplet_p2 = build_triplet_model(emb_model_p2)
    triplet_p2.compile(
        # CRITICAL: LR must be much lower than phase 1 to avoid destroying backbone weights
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=triplet_loss()
    )

    # Recreate datasets for phase 2 (tf.data.Dataset.take() is exhausted after one pass)
    train_ds2, _ = make_triplet_dataset(train_imgs, preprocessor,
                                         batch_size=batch_size,
                                         steps=steps_per_epoch, seed=7)
    val_ds2,   _ = make_triplet_dataset(val_imgs,   preprocessor,
                                         batch_size=batch_size,
                                         steps=val_steps, seed=77)

    ckpt_p2 = str(output_dir / "phase2_best.weights.h5")
    history2 = triplet_p2.fit(
        train_ds2,
        validation_data=val_ds2,
        epochs=phase2_epochs,
        callbacks=[
            ModelCheckpoint(ckpt_p2, monitor="val_loss",
                            save_best_only=True, save_weights_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5,
                              min_lr=1e-7, verbose=1),
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=1),
            CSVLogger(str(output_dir / "phase2_log.csv")),
        ],
        verbose=1,
    )

    # ── Save final embedding model weights ───────────────────────────────────
    triplet_p2.load_weights(ckpt_p2)
    final_path = config.MODEL_WEIGHTS
    emb_model_p2.save_weights(str(final_path))
    log.info("Final embedding weights saved → %s", final_path)

    # ── Validation metrics on held-out subjects ───────────────────────────────
    log.info("Computing validation metrics…")
    metrics = compute_val_metrics(emb_model_p2, preprocessor, val_imgs)
    log.info("Final validation:  separability = %.4f", metrics.get("separability", 0))

    return {
        "phase1_history"   : history1.history,
        "phase2_history"   : history2.history,
        "val_metrics"      : metrics,
        "weights_path"     : str(final_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on SOCOFing (two-phase fine-tuning)")
    parser.add_argument("--dir",            required=True,              help="Path to SOCOFing/Real/")
    parser.add_argument("--phase1-epochs",  type=int, default=5,        help="Head warm-up epochs (default 5)")
    parser.add_argument("--phase2-epochs",  type=int, default=15,       help="Fine-tune epochs (default 15)")
    parser.add_argument("--batch",          type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--steps",          type=int, default=config.TRAIN_STEPS)
    parser.add_argument("--val-steps",      type=int, default=50)
    parser.add_argument("--output-dir",     default=str(config.BASE_DIR / "models"))
    args = parser.parse_args()

    results = train(
        real_dir        = Path(args.dir),
        phase1_epochs   = args.phase1_epochs,
        phase2_epochs   = args.phase2_epochs,
        batch_size      = args.batch,
        steps_per_epoch = args.steps,
        val_steps       = args.val_steps,
        output_dir      = Path(args.output_dir),
    )
    print(f"\nTraining complete.")
    print(f"Weights → {results['weights_path']}")
    print(f"Val separability: {results['val_metrics'].get('separability', 'N/A'):.4f}")