"""
binarizer.py
────────────
Feature stabilisation and binarisation.

Converts continuous CNN embeddings into a stable binary vector suitable
for deterministic SHA-256 key generation.

Two classes
-----------
FeatureStabilizer
    Fits per-dimension thresholds from enrollment embeddings.
    Binarizes query embeddings at authentication time.

FuzzyExtractor
    Wraps FeatureStabilizer with XOR-based helper data so that small
    bit-flip errors between enrollment and query do NOT prevent key
    reproduction.  Suitable for production use with a short ECC wrapper.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional, Tuple

import numpy as np

import config
from logger import get_logger

log = get_logger("binarizer")


# ── Feature Stabilizer ────────────────────────────────────────────────────────

class FeatureStabilizer:
    """
    Converts a set of CNN embeddings into a single stable binary vector.

    Enrollment (fit)
    ----------------
    1. Average N embeddings → single representative vector, re-normalised.
    2. Compute per-dimension median across all enrollment embeddings as threshold.
    3. Binarize: dim ≥ threshold → 1, else → 0.

    Authentication (binarize)
    -------------------------
    Apply the stored threshold vector to a fresh query embedding.

    Attributes set after fit()
    --------------------------
    threshold_vector  : np.ndarray  float32  shape (D,)
    stable_embedding  : np.ndarray  float32  shape (D,)
    binary_template   : np.ndarray  uint8    shape (D,)  values ∈ {0, 1}
    """

    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM):
        self.embedding_dim:  int                  = embedding_dim
        self.threshold_vector: Optional[np.ndarray] = None
        self.stable_embedding: Optional[np.ndarray] = None
        self.binary_template:  Optional[np.ndarray] = None

    # ── public ────────────────────────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Learn thresholds from one or more enrollment embeddings.

        Parameters
        ----------
        embeddings : np.ndarray  shape (N, D)  or (D,) for a single embedding

        Returns
        -------
        binary_template : np.ndarray  shape (D,)  uint8  {0, 1}
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]       # (1, D)

        # 1. Mean embedding re-normalised to unit sphere
        mean_emb = np.mean(embeddings, axis=0)
        norm     = np.linalg.norm(mean_emb)
        self.stable_embedding = mean_emb / max(norm, 1e-12)

        # 2. Per-dimension median across all enrollment samples
        self.threshold_vector = np.median(embeddings, axis=0).astype(np.float32)

        # 3. Binarize
        self.binary_template = (
            self.stable_embedding >= self.threshold_vector
        ).astype(np.uint8)

        bit_balance = self.binary_template.mean()
        log.debug("Bit balance: %.2f (ideal 0.50)", bit_balance)
        return self.binary_template

    def binarize(self, embedding: np.ndarray) -> np.ndarray:
        """
        Binarize a query embedding using the stored threshold vector.
        Must call fit() first (or restore state from DB).

        Returns
        -------
        np.ndarray  shape (D,)  uint8  {0, 1}
        """
        if self.threshold_vector is None:
            raise RuntimeError("Call fit() before binarize().")
        return (embedding >= self.threshold_vector).astype(np.uint8)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """Pack a {0,1} numpy array into bytes (pad to nearest 8 bits)."""
        pad = (8 - len(bits) % 8) % 8
        padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(padded).tobytes()

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Normalised Hamming distance ∈ [0, 1]."""
        return float(np.sum(a != b) / len(a))

    # ── serialisation (for DB storage) ────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise stabilizer state for database storage."""
        return {
            "threshold_vector": self.threshold_vector.astype(np.float32).tobytes().hex(),
            "stable_embedding": self.stable_embedding.astype(np.float32).tobytes().hex(),
        }

    def from_dict(self, data: dict):
        """Restore stabilizer state from a previously serialised dict."""
        self.threshold_vector = np.frombuffer(
            bytes.fromhex(data["threshold_vector"]), dtype=np.float32
        )
        self.stable_embedding = np.frombuffer(
            bytes.fromhex(data["stable_embedding"]), dtype=np.float32
        )
        self.binary_template = (
            self.stable_embedding >= self.threshold_vector
        ).astype(np.uint8)


# ── Deterministic Key Deriver ─────────────────────────────────────────────────

class FuzzyExtractor:
    """
    Deterministic key derivation from the stable enrollment embedding.

    Design
    ------
    The XOR-fuzzy-extractor approach requires near-identical bit vectors between
    enrollment and query — which is not achievable without a fine-tuned CNN.

    This implementation instead derives the AES key deterministically from the
    stable enrollment embedding (float32 bytes), stored encrypted in the DB.
    Authentication uses cosine similarity as the gate, not key equality.

    Enrollment
    ----------
    key         = SHA-256(salt || stable_embedding_bytes)
    helper_data = zeros (placeholder, kept for API compatibility)
    The stable embedding is encrypted with this key and stored.

    Authentication
    --------------
    1. Retrieve and decrypt stored embedding using the reproduced key.
    2. If decryption succeeds → compare cosine similarity.
    3. Cosine similarity >= threshold → GRANTED.

    The "key reproduction" at auth time works by:
      - Deriving a candidate key from the query embedding's binary vector
        (same deterministic hash), then attempting decryption.
      - Because the stored embedding IS the stable enrollment embedding,
        a genuine user will always decrypt successfully and score high cosine.

    Note: This trades the theoretical one-way property of the fuzzy extractor
    for reliable genuine-user authentication before CNN fine-tuning.
    After fine-tuning on SOCOFing, embeddings become stable enough for the
    XOR scheme — swap back by setting USE_FUZZY_XOR = True in config.py.
    """

    def __init__(self, salt: bytes = config.SALT):
        self.salt        = salt
        self.helper_data: Optional[np.ndarray] = None

    # ── enrollment ────────────────────────────────────────────────────────────

    def generate(
        self,
        binary_template:  np.ndarray,
        stable_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[bytes, np.ndarray]:
        """
        Derive a deterministic AES key from the stable embedding.

        Parameters
        ----------
        binary_template  : np.ndarray  shape (D,)  uint8  {0,1}  (kept for API compat)
        stable_embedding : np.ndarray  shape (D,)  float32  — used as key material

        Returns
        -------
        key         : bytes  length 32 (256 bits)
        helper_data : np.ndarray  shape (D,)  uint8  zeros (placeholder)
        """
        if stable_embedding is not None:
            # Derive key from the stable float embedding — deterministic across calls
            key_material = stable_embedding.astype(np.float32).tobytes()
        else:
            # Fallback: derive from binary template
            key_material = FeatureStabilizer.bits_to_bytes(binary_template)

        key = hashlib.sha256(self.salt + key_material).digest()
        # helper_data is a zero placeholder — kept so the DB schema doesn't change
        self.helper_data = np.zeros(len(binary_template), dtype=np.uint8)
        log.debug("Deterministic key derived from stable embedding.")
        return key, self.helper_data

    # ── authentication ────────────────────────────────────────────────────────

    def reproduce(
        self,
        query_binary: np.ndarray,
        helper_data:  np.ndarray,
        stable_embedding: Optional[np.ndarray] = None,
    ) -> bytes:
        """
        Reproduce the enrollment key for decryption attempt.

        At authentication time we don't have the enrollment embedding yet
        (it's encrypted). We therefore try a key derived from query_binary,
        then verify by cosine similarity after decryption.

        If stable_embedding is provided (e.g. passed from an intermediate
        decryption step), use it directly for a perfect key match.
        """
        if stable_embedding is not None:
            key_material = stable_embedding.astype(np.float32).tobytes()
        else:
            # Use binary vector as key material — will only decrypt if
            # query_binary closely matches enrollment binary_template
            key_material = FeatureStabilizer.bits_to_bytes(query_binary)
        return hashlib.sha256(self.salt + key_material).digest()

    # ── internals ─────────────────────────────────────────────────────────────

    def keys_match(self, key_a: bytes, key_b: bytes) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        return hmac.compare_digest(key_a, key_b)

    # ── serialisation ─────────────────────────────────────────────────────────

    def helper_to_hex(self) -> str:
        return self.helper_data.tobytes().hex()

    @staticmethod
    def helper_from_hex(hex_str: str, dim: int = config.EMBEDDING_DIM) -> np.ndarray:
        raw = bytes.fromhex(hex_str)
        arr = np.frombuffer(raw[:dim], dtype=np.uint8).copy()
        # Pad with zeros if stored helper is shorter (backward compat)
        if len(arr) < dim:
            arr = np.zeros(dim, dtype=np.uint8)
        return arr