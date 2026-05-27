"""
feature_extractor.py
────────────────────
MobileNetV2-based fingerprint feature extractor.

Architecture
------------
Input (H, W, 1)
  → Lambda: repeat to 3 channels
  → MobileNetV2 backbone (ImageNet weights, configurable freeze)
  → GlobalAveragePooling2D
  → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(128)
  → L2-Normalize
  → 128-D unit embedding

Training (optional)
-------------------
build_triplet_model() returns a Siamese triplet network.
Call train() with a SOCOFing subject_images dict to fine-tune.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

import config
from logger import get_logger

log = get_logger("feature_extractor")


# ── Triplet loss ───────────────────────────────────────────────────────────────

def triplet_loss_fn(margin: float = config.TRIPLET_MARGIN):
    """
    Hard triplet loss.
    L = max(||a − p||² − ||a − n||² + margin, 0)
    """
    dim = config.EMBEDDING_DIM

    def loss(y_true, y_pred):
        a = y_pred[:, :dim]
        p = y_pred[:, dim : 2 * dim]
        n = y_pred[:, 2 * dim :]
        d_pos = tf.reduce_sum(tf.square(a - p), axis=1)
        d_neg = tf.reduce_sum(tf.square(a - n), axis=1)
        return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

    return loss


# ── Data generator ─────────────────────────────────────────────────────────────

class TripletGenerator(tf.keras.utils.Sequence):
    """
    Yields (anchor, positive, negative) batches from SOCOFing subject images.

    Only subjects with ≥ 2 images are used.
    """

    def __init__(
        self,
        subject_images: Dict[int, List[str]],
        preprocessor,
        batch_size: int = config.TRAIN_BATCH_SIZE,
        steps:      int = config.TRAIN_STEPS,
    ):
        super().__init__()                          # required by Keras >= 2.12
        import random as _random
        self.rng     = _random.Random(42)
        self.preproc = preprocessor
        self.batch   = batch_size
        self.steps   = steps
        self.subs    = [s for s, imgs in subject_images.items() if len(imgs) >= 2]
        self.sub_imgs = subject_images

    def __len__(self):
        return self.steps

    def __getitem__(self, _idx):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch):
            sid     = self.rng.choice(self.subs)
            a_p     = self.rng.sample(self.sub_imgs[sid], 2)
            neg_sid = self.rng.choice([s for s in self.subs if s != sid])
            n_path  = self.rng.choice(self.sub_imgs[neg_sid])

            anchors.append(self.preproc.process(a_p[0]))
            positives.append(self.preproc.process(a_p[1]))
            negatives.append(self.preproc.process(n_path))

        dummy = np.zeros((self.batch, 1), dtype=np.float32)
        # Return a tuple (not list) of inputs -- required by TF/Keras >= 2.12
        return (
            np.array(anchors),
            np.array(positives),
            np.array(negatives),
        ), dummy


# ── Feature extractor ──────────────────────────────────────────────────────────

class FingerprintCNNExtractor:
    """
    Wraps the embedding model.

    Usage
    -----
    >>> extractor = FingerprintCNNExtractor()
    >>> emb = extractor.extract(tensor)   # (128,)
    >>> extractor.save(); extractor.load()
    """

    EMBEDDING_DIM = config.EMBEDDING_DIM

    def __init__(
        self,
        input_shape:      Tuple[int, int, int] = (*config.IMG_SIZE, 1),
        fine_tune_layers: int                  = config.FINE_TUNE_LAYERS,
    ):
        self.input_shape      = input_shape
        self.fine_tune_layers = fine_tune_layers
        self.model            = self._build_model()
        self._weights_path    = config.MODEL_WEIGHTS

        # Auto-load saved weights if they exist
        if self._weights_path.exists():
            try:
                self.model.load_weights(str(self._weights_path))
                log.info("Loaded CNN weights from %s", self._weights_path)
            except Exception as exc:
                log.warning("Could not load weights (%s). Using ImageNet init.", exc)

    # ── build ──────────────────────────────────────────────────────────────────

    def _build_model(self) -> Model:
        inp = Input(shape=self.input_shape, name="fingerprint_input")

        # Grayscale → 3-channel (MobileNetV2 expects RGB)
        x = layers.Lambda(
            lambda t: tf.repeat(t, 3, axis=-1), name="gray_to_rgb"
        )(inp)

        # Backbone
        backbone = MobileNetV2(
            input_shape=(self.input_shape[0], self.input_shape[1], 3),
            include_top=False,
            weights="imagenet",
        )
        # Freeze all, selectively unfreeze last N for fine-tuning
        backbone.trainable = False
        if self.fine_tune_layers > 0:
            for layer in backbone.layers[-self.fine_tune_layers :]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

        x = backbone(x, training=False)

        # Projection head
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(256, activation="relu", name="proj_256")(x)
        x = layers.BatchNormalization(name="bn_256")(x)
        x = layers.Dropout(0.3, name="drop")(x)
        x = layers.Dense(self.EMBEDDING_DIM, name="proj_128")(x)

        # L2-normalise → unit hypersphere
        embeddings = layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1), name="l2_normalize"
        )(x)

        return Model(inputs=inp, outputs=embeddings, name="FingerprintEmbedder")

    def build_triplet_model(self) -> Model:
        """Return a model that accepts (anchor, positive, negative) triplets."""
        anchor   = Input(shape=self.input_shape, name="anchor")
        positive = Input(shape=self.input_shape, name="positive")
        negative = Input(shape=self.input_shape, name="negative")

        emb_a = self.model(anchor)
        emb_p = self.model(positive)
        emb_n = self.model(negative)

        out = layers.Concatenate(axis=-1, name="triplet_out")([emb_a, emb_p, emb_n])
        return Model(inputs=[anchor, positive, negative], outputs=out, name="TripletNet")

    # ── inference ─────────────────────────────────────────────────────────────

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single preprocessed image.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 1)  float32  [0, 1]

        Returns
        -------
        np.ndarray  shape (EMBEDDING_DIM,)  float32   ||v||₂ ≈ 1
        """
        batch = image[np.newaxis, ...]                       # (1, H, W, 1)
        return self.model.predict(batch, verbose=0)[0]       # (128,)

    def extract_batch(self, images: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Extract embeddings for a batch.

        Parameters
        ----------
        images : np.ndarray  shape (N, H, W, 1)

        Returns
        -------
        np.ndarray  shape (N, EMBEDDING_DIM)
        """
        return self.model.predict(images, batch_size=32, verbose=int(verbose))

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        subject_images: Dict[int, List[str]],
        preprocessor,
        epochs: int = config.TRAIN_EPOCHS,
    ) -> dict:
        """
        Fine-tune with triplet loss on SOCOFing.

        Returns Keras history dict.
        """
        triplet_net = self.build_triplet_model()
        triplet_net.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=triplet_loss_fn(config.TRIPLET_MARGIN),
        )

        gen = TripletGenerator(subject_images, preprocessor)

        log.info("Starting triplet training: %d epochs × %d steps", epochs, config.TRAIN_STEPS)
        history = triplet_net.fit(
            gen,
            epochs=epochs,
            callbacks=[
                ReduceLROnPlateau(monitor="loss", patience=2, factor=0.5, verbose=1),
                EarlyStopping(monitor="loss", patience=4, restore_best_weights=True),
            ],
            verbose=1,
        )
        self.save()
        return history.history

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None):
        p = path or self._weights_path
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(p))
        log.info("CNN weights saved → %s", p)

    def load(self, path: Optional[Path] = None):
        p = path or self._weights_path
        self.model.load_weights(str(p))
        log.info("CNN weights loaded ← %s", p)

    def summary(self):
        self.model.summary()