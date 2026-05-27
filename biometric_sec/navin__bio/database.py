"""
database.py
───────────
SQLite-backed secure biometric database.

Schema
------
users
    user_id         TEXT PRIMARY KEY
    enrolled_at     TEXT
    images_used     INTEGER
    ciphertext      TEXT   ← AES-256 encrypted embedding (hex)
    iv              TEXT   ← AES IV (hex)
    helper_data     TEXT   ← Fuzzy extractor helper (hex)
    threshold_vec   TEXT   ← Per-dimension binarisation thresholds (hex)
    stable_emb      TEXT   ← Stable float embedding (hex)  ← optional, for cosine match

NEVER stores: raw embeddings, AES keys, binary templates, or plaintext features.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import config
from logger import get_logger

log = get_logger("database")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    enrolled_at   TEXT NOT NULL,
    images_used   INTEGER NOT NULL DEFAULT 1,
    ciphertext    TEXT NOT NULL,
    iv            TEXT NOT NULL,
    helper_data   TEXT NOT NULL,
    threshold_vec TEXT NOT NULL,
    stable_emb    TEXT NOT NULL
);
"""


class FingerprintDatabase:
    """
    Thread-safe SQLite interface for encrypted fingerprint templates.

    Parameters
    ----------
    db_path : path to the SQLite file (created automatically if absent)
    """

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    # ── context manager ───────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Yield a connection with row_factory and auto-commit on success."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
        log.info("Database initialised at %s", self.db_path)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def store(
        self,
        user_id:      str,
        ciphertext:   str,
        iv:           str,
        helper_data:  str,
        threshold_vec: str,
        stable_emb:   str,
        images_used:  int = 1,
    ):
        """
        Insert or replace a user's encrypted template.

        All binary parameters are passed as hex strings.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users
                    (user_id, enrolled_at, images_used,
                     ciphertext, iv, helper_data, threshold_vec, stable_emb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, now, images_used,
                 ciphertext, iv, helper_data, threshold_vec, stable_emb),
            )
        log.info("Stored encrypted template for user '%s' (%d images)", user_id, images_used)

    def fetch(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve a user record.

        Returns
        -------
        dict with keys matching the schema columns, or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            log.debug("User '%s' not found in database.", user_id)
            return None
        return dict(row)

    def delete(self, user_id: str) -> bool:
        """Delete a user's record. Returns True if a row was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE user_id = ?", (user_id,)
            )
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("Deleted user '%s' from database.", user_id)
        else:
            log.warning("Delete requested for unknown user '%s'.", user_id)
        return deleted

    def exists(self, user_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row is not None

    def list_users(self) -> List[str]:
        """Return all enrolled user IDs."""
        with self._conn() as conn:
            rows = conn.execute("SELECT user_id FROM users ORDER BY enrolled_at").fetchall()
        return [r["user_id"] for r in rows]

    def user_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def user_info(self, user_id: str) -> Optional[Dict]:
        """
        Return non-sensitive metadata for a user (no cryptographic fields).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, enrolled_at, images_used FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def all_user_info(self) -> List[Dict]:
        """Return metadata for all users (no cryptographic fields)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT user_id, enrolled_at, images_used FROM users ORDER BY enrolled_at"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── backup / export ───────────────────────────────────────────────────────

    def export_encrypted(self, out_path: Path):
        """
        Export the full database (encrypted records only) to a JSON file.
        Safe to transfer — contains no plaintext biometric data.
        """
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM users").fetchall()
        data = [dict(r) for r in rows]
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Database exported to %s (%d records)", out_path, len(data))

    def import_encrypted(self, in_path: Path):
        """Import records from a JSON export file."""
        with open(in_path) as f:
            records = json.load(f)
        for r in records:
            self.store(
                user_id      = r["user_id"],
                ciphertext   = r["ciphertext"],
                iv           = r["iv"],
                helper_data  = r["helper_data"],
                threshold_vec= r["threshold_vec"],
                stable_emb   = r["stable_emb"],
                images_used  = r.get("images_used", 1),
            )
        log.info("Imported %d records from %s", len(records), in_path)
