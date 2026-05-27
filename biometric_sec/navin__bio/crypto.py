"""
crypto.py
─────────
AES-256-CBC encryption / decryption for fingerprint templates.

Design decisions
----------------
* Fresh 16-byte IV per encryption → same plaintext never produces same ciphertext.
* IV is stored alongside the ciphertext (both as hex strings in the DB row).
* The AES key is derived externally by FuzzyExtractor.reproduce() and
  NEVER persisted in the database.
* Wrong key → ValueError on PKCS7 unpadding → authentication denied.
"""

from __future__ import annotations

import base64
from typing import Dict, Tuple

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

import config
from logger import get_logger

log = get_logger("crypto")

_BLOCK = 16   # AES block size in bytes


class AESCryptosystem:
    """
    AES-256-CBC encryption / decryption.

    All values stored as hex strings for easy JSON / SQLite serialisation.
    """

    # ── encryption ────────────────────────────────────────────────────────────

    def encrypt(self, plaintext: bytes, key: bytes) -> Dict[str, str]:
        """
        Encrypt plaintext with a fresh IV.

        Parameters
        ----------
        plaintext : bytes — the data to protect (e.g. serialised embedding)
        key       : bytes — 32-byte AES key

        Returns
        -------
        {"iv": <hex>, "ciphertext": <hex>}
        """
        assert len(key) == config.AES_KEY_SIZE, (
            f"Key must be {config.AES_KEY_SIZE} bytes; got {len(key)}."
        )
        iv     = get_random_bytes(_BLOCK)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct     = cipher.encrypt(pad(plaintext, _BLOCK))
        log.debug("Encrypted %d bytes → ciphertext %d bytes", len(plaintext), len(ct))
        return {"iv": iv.hex(), "ciphertext": ct.hex()}

    # ── decryption ────────────────────────────────────────────────────────────

    def decrypt(self, record: Dict[str, str], key: bytes) -> bytes:
        """
        Decrypt a record produced by encrypt().

        Raises
        ------
        ValueError  — wrong key or tampered ciphertext (PKCS7 padding check fails)
        """
        iv     = bytes.fromhex(record["iv"])
        ct     = bytes.fromhex(record["ciphertext"])
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plain  = unpad(cipher.decrypt(ct), _BLOCK)   # raises ValueError on bad key
        log.debug("Decrypted %d bytes → plaintext %d bytes", len(ct), len(plain))
        return plain

    # ── helper — serialise embedding ──────────────────────────────────────────

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Convert a float32 embedding to raw bytes."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(data: bytes, dim: int = config.EMBEDDING_DIM) -> np.ndarray:
        """Restore a float32 embedding from raw bytes."""
        arr = np.frombuffer(data, dtype=np.float32)
        if len(arr) != dim:
            raise ValueError(
                f"Expected {dim}-dim embedding; got {len(arr)} values. "
                "Possible key mismatch or data corruption."
            )
        return arr
