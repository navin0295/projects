"""
authenticate.py
───────────────
End-to-end authentication pipeline.

Flow
----
1.  Preprocess the query fingerprint image.
2.  Extract CNN embedding.
3.  Binarise using the stored per-dimension threshold vector.
4.  Reproduce AES key via FuzzyExtractor (helper data from DB).
5.  Attempt AES-256 decryption of the stored template.
    → Wrong key causes ValueError → DENIED immediately.
6.  Compute cosine similarity between decrypted enrollment embedding
    and query embedding.
7.  Decision: cosine_sim ≥ COSINE_THRESHOLD → GRANTED.

Returns a rich AuthResult dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

import config
from binarizer import FuzzyExtractor
from crypto import AESCryptosystem
from database import FingerprintDatabase
from feature_extractor import FingerprintCNNExtractor
from preprocessor import FingerprintPreprocessor
from logger import get_logger

log = get_logger("authenticate")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AuthResult:
    """
    Structured authentication result.

    Attributes
    ----------
    granted      : True = GRANTED, False = DENIED
    user_id      : claimed user identity
    cosine_sim   : cosine similarity between stored and query embeddings (or 0.0)
    hamming_dist : normalised Hamming distance between binary vectors (or 1.0)
    reason       : human-readable denial reason (empty string on GRANT)
    query_path   : path to the query image (if provided)
    """
    granted:      bool
    user_id:      str
    cosine_sim:   float = 0.0
    hamming_dist: float = 1.0
    reason:       str   = ""
    query_path:   str   = ""

    @property
    def decision(self) -> str:
        return "GRANTED" if self.granted else "DENIED"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision
        return d

    def __str__(self):
        if self.granted:
            return (
                f"✅ GRANTED  user={self.user_id}  "
                f"cosine={self.cosine_sim:.4f}  hamming={self.hamming_dist:.4f}"
            )
        return (
            f"❌ DENIED   user={self.user_id}  "
            f"reason={self.reason}  cosine={self.cosine_sim:.4f}"
        )


# ── Authentication pipeline ───────────────────────────────────────────────────

class AuthenticationPipeline:
    """
    Verify a fingerprint against a stored encrypted template.

    Parameters
    ----------
    preprocessor : FingerprintPreprocessor
    extractor    : FingerprintCNNExtractor
    aes          : AESCryptosystem
    db           : FingerprintDatabase
    """

    def __init__(
        self,
        preprocessor: FingerprintPreprocessor,
        extractor:    FingerprintCNNExtractor,
        aes:          AESCryptosystem,
        db:           FingerprintDatabase,
    ):
        self.preprocessor = preprocessor
        self.extractor    = extractor
        self.aes          = aes
        self.db           = db

    # ── main entry point ──────────────────────────────────────────────────────

    def authenticate(
        self,
        user_id:    str,
        image_path: str | Path,
    ) -> AuthResult:
        """
        Verify a query fingerprint against a stored template.

        Parameters
        ----------
        user_id    : claimed identity
        image_path : path to the query fingerprint image

        Returns
        -------
        AuthResult — always returned (never raises on biometric mismatch)

        Raises
        ------
        FileNotFoundError — query image not found on disk
        """
        image_path = str(image_path)

        # ── step 0: look up user ──────────────────────────────────────────────
        record = self.db.fetch(user_id)
        if record is None:
            log.warning("Auth attempt for unknown user '%s'.", user_id)
            return AuthResult(
                granted=False, user_id=user_id,
                reason="User not enrolled", query_path=image_path,
            )

        # ── step 1: preprocess ────────────────────────────────────────────────
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        query_tensor = self.preprocessor.process(image_path)

        # ── step 2: CNN embedding ─────────────────────────────────────────────
        query_emb = self.extractor.extract(query_tensor)

        # ── step 3: binarise with stored threshold vector ─────────────────────
        thresh_bytes  = bytes.fromhex(record["threshold_vec"])
        thresh_vector = np.frombuffer(thresh_bytes, dtype=np.float32)

        if len(thresh_vector) != config.EMBEDDING_DIM:
            log.error("Threshold vector dim mismatch for user '%s'.", user_id)
            return AuthResult(
                granted=False, user_id=user_id,
                reason="Template dimension mismatch", query_path=image_path,
            )

        query_binary = (query_emb >= thresh_vector).astype(np.uint8)

        # ── step 4: reproduce AES key from stored stable embedding ────────────
        # The stable embedding is stored encrypted in the DB (hex).
        # We derive the same deterministic key used at enrollment:
        #   key = SHA-256(salt || stable_embedding_bytes)
        # This is possible because we stored stable_emb in the DB record.
        stored_stable_bytes = bytes.fromhex(record["stable_emb"])
        stored_stable_emb   = np.frombuffer(stored_stable_bytes, dtype=np.float32).copy()

        fuzzy     = FuzzyExtractor()
        repro_key = fuzzy.reproduce(
            query_binary,
            FuzzyExtractor.helper_from_hex(record["helper_data"]),
            stable_embedding=stored_stable_emb,   # deterministic key reproduction
        )

        # ── step 5: AES decryption ────────────────────────────────────────────
        try:
            plain = self.aes.decrypt(
                {"iv": record["iv"], "ciphertext": record["ciphertext"]},
                repro_key,
            )
            stored_emb = self.aes.bytes_to_embedding(plain)
        except (ValueError, Exception) as exc:
            log.debug("Decryption failed for '%s': %s", user_id, exc)
            return AuthResult(
                granted=False, user_id=user_id,
                reason="Key mismatch (decryption failed)", query_path=image_path,
            )

        # ── step 6: cosine similarity ─────────────────────────────────────────
        cos_sim = _cosine_similarity(stored_emb, query_emb)

        # Hamming distance (stored binary vs query binary)
        stored_binary = (stored_emb >= thresh_vector).astype(np.uint8)
        ham_dist      = float(np.mean(query_binary != stored_binary))

        # ── step 7: decision ──────────────────────────────────────────────────
        granted = cos_sim >= config.COSINE_THRESHOLD

        result = AuthResult(
            granted      = granted,
            user_id      = user_id,
            cosine_sim   = round(float(cos_sim), 4),
            hamming_dist = round(ham_dist, 4),
            reason       = "" if granted else f"Cosine {cos_sim:.4f} < {config.COSINE_THRESHOLD}",
            query_path   = image_path,
        )
        log.info("%s", result)
        return result

    # ── authenticate from raw bytes ───────────────────────────────────────────

    def authenticate_from_bytes(
        self,
        user_id:    str,
        image_data: bytes,
    ) -> AuthResult:
        """
        Authenticate from raw image bytes (e.g. from an HTTP upload).
        """
        record = self.db.fetch(user_id)
        if record is None:
            return AuthResult(granted=False, user_id=user_id, reason="User not enrolled")

        query_tensor  = self.preprocessor.process_from_bytes(image_data)
        query_emb     = self.extractor.extract(query_tensor)

        thresh_vector = np.frombuffer(bytes.fromhex(record["threshold_vec"]), dtype=np.float32)
        query_binary  = (query_emb >= thresh_vector).astype(np.uint8)

        helper_data = FuzzyExtractor.helper_from_hex(record["helper_data"])
        fuzzy       = FuzzyExtractor()
        repro_key   = fuzzy.reproduce(query_binary, helper_data)

        try:
            plain      = self.aes.decrypt(
                {"iv": record["iv"], "ciphertext": record["ciphertext"]}, repro_key
            )
            stored_emb = self.aes.bytes_to_embedding(plain)
        except Exception:
            return AuthResult(granted=False, user_id=user_id, reason="Key mismatch")

        cos_sim = _cosine_similarity(stored_emb, query_emb)
        granted = cos_sim >= config.COSINE_THRESHOLD
        stored_binary = (stored_emb >= thresh_vector).astype(np.uint8)
        ham_dist = float(np.mean(query_binary != stored_binary))

        return AuthResult(
            granted      = granted,
            user_id      = user_id,
            cosine_sim   = round(float(cos_sim), 4),
            hamming_dist = round(ham_dist, 4),
            reason       = "" if granted else f"Cosine {cos_sim:.4f} < {config.COSINE_THRESHOLD}",
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))