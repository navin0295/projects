"""
enroll.py
─────────
End-to-end enrollment pipeline.

Flow
----
1.  Preprocess each fingerprint image.
2.  Extract CNN embedding from each image.
3.  Stabilise embeddings (average + per-dim threshold).
4.  Binarise stable embedding.
5.  Generate AES key + helper data via FuzzyExtractor.
6.  Encrypt the stable (float) embedding with AES-256-CBC.
7.  Store ONLY encrypted template + helper data + threshold vector in DB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from binarizer import FeatureStabilizer, FuzzyExtractor
from crypto import AESCryptosystem
from database import FingerprintDatabase
from feature_extractor import FingerprintCNNExtractor
from preprocessor import FingerprintPreprocessor
from logger import get_logger

log = get_logger("enroll")


class EnrollmentPipeline:
    """
    Enroll one user from one or more fingerprint images.

    At least config.MIN_ENROLL_IMAGES images are required.
    Using config.RECOMMENDED_ENROLL (3) images produces a more stable key.

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

    def enroll(
        self,
        user_id:     str,
        image_paths: List[str | Path],
        overwrite:   bool = False,
    ) -> Dict:
        """
        Enroll a user.

        Parameters
        ----------
        user_id     : unique string identifier (e.g. "alice", "user_0001")
        image_paths : list of paths to fingerprint images (same finger)
        overwrite   : if False and user already exists, raises ValueError

        Returns
        -------
        dict with enrollment summary (no sensitive data)

        Raises
        ------
        ValueError  — user already exists and overwrite=False
        ValueError  — too few images
        FileNotFoundError — an image path does not exist
        """
        # ── guard checks ──────────────────────────────────────────────────────
        if not overwrite and self.db.exists(user_id):
            raise ValueError(
                f"User '{user_id}' is already enrolled. "
                "Pass overwrite=True to re-enroll."
            )

        import config
        if len(image_paths) < config.MIN_ENROLL_IMAGES:
            raise ValueError(
                f"At least {config.MIN_ENROLL_IMAGES} image(s) required; "
                f"got {len(image_paths)}."
            )

        log.info("Enrolling user '%s' with %d image(s)…", user_id, len(image_paths))

        # ── step 1: preprocess ────────────────────────────────────────────────
        tensors = []
        for p in image_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            tensors.append(self.preprocessor.process(p))
        log.debug("Preprocessing done.")

        # ── step 2: CNN embedding ─────────────────────────────────────────────
        embeddings = np.array([self.extractor.extract(t) for t in tensors])
        log.debug("Embeddings shape: %s", embeddings.shape)

        # ── step 3: stabilise ─────────────────────────────────────────────────
        stabilizer = FeatureStabilizer()
        binary_template = stabilizer.fit(embeddings)
        log.debug("Binary template: %d bits, %.2f%% ones",
                  len(binary_template), binary_template.mean() * 100)

        # ── step 4: derive AES key from stable embedding ─────────────────────
        fuzzy = FuzzyExtractor()
        aes_key, helper_data = fuzzy.generate(
            binary_template,
            stable_embedding=stabilizer.stable_embedding   # deterministic key source
        )
        log.debug("AES key derived (not stored).")

        # ── step 5: encrypt the stable float embedding ────────────────────────
        emb_bytes = self.aes.embedding_to_bytes(stabilizer.stable_embedding)
        enc_rec   = self.aes.encrypt(emb_bytes, aes_key)

        # ── step 6: persist (encrypted only) ─────────────────────────────────
        self.db.store(
            user_id       = user_id,
            ciphertext    = enc_rec["ciphertext"],
            iv            = enc_rec["iv"],
            helper_data   = fuzzy.helper_to_hex(),
            threshold_vec = stabilizer.threshold_vector.astype(np.float32).tobytes().hex(),
            stable_emb    = stabilizer.stable_embedding.astype(np.float32).tobytes().hex(),
            images_used   = len(image_paths),
        )

        summary = {
            "user_id"     : user_id,
            "status"      : "enrolled",
            "images_used" : len(image_paths),
            "bit_balance" : round(float(binary_template.mean()), 4),
        }
        log.info("Enrollment complete: %s", summary)
        return summary

    # ── convenience: enroll from bytes ────────────────────────────────────────

    def enroll_from_bytes(
        self,
        user_id:    str,
        images:     List[bytes],
        overwrite:  bool = False,
    ) -> Dict:
        """
        Enroll from raw image bytes (e.g. received over HTTP).
        """
        import config
        if len(images) < config.MIN_ENROLL_IMAGES:
            raise ValueError(
                f"Need at least {config.MIN_ENROLL_IMAGES} image(s); got {len(images)}."
            )

        if not overwrite and self.db.exists(user_id):
            raise ValueError(f"User '{user_id}' already enrolled.")

        tensors    = [self.preprocessor.process_from_bytes(b) for b in images]
        embeddings = np.array([self.extractor.extract(t) for t in tensors])

        stabilizer      = FeatureStabilizer()
        binary_template = stabilizer.fit(embeddings)

        fuzzy = FuzzyExtractor()
        aes_key, _ = fuzzy.generate(
            binary_template,
            stable_embedding=stabilizer.stable_embedding
        )

        emb_bytes = self.aes.embedding_to_bytes(stabilizer.stable_embedding)
        enc_rec   = self.aes.encrypt(emb_bytes, aes_key)

        self.db.store(
            user_id       = user_id,
            ciphertext    = enc_rec["ciphertext"],
            iv            = enc_rec["iv"],
            helper_data   = fuzzy.helper_to_hex(),
            threshold_vec = stabilizer.threshold_vector.astype(np.float32).tobytes().hex(),
            stable_emb    = stabilizer.stable_embedding.astype(np.float32).tobytes().hex(),
            images_used   = len(images),
        )
        return {"user_id": user_id, "status": "enrolled", "images_used": len(images)}