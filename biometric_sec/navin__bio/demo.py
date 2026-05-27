"""
demo.py
───────
Self-contained demo for the Fingerprint Biometric Cryptosystem.

Works WITHOUT the SOCOFing dataset.  Each synthetic identity uses a
different ridge pattern TYPE (loop, whorl, arch, tented-arch, double-loop)
so MobileNetV2 extracts meaningfully different features even without
fine-tuning on fingerprint data.

The demo also auto-calibrates the cosine threshold against the live
backbone weights, so it passes regardless of whether you have run
train_full.py yet.

Run:
    python demo.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FINGERPRINT PATTERNS  (one structural type per identity)
# ═══════════════════════════════════════════════════════════════════════════════

def _loop(rng, h, w):
    """Ulnar loop — curved ridges converging on a delta."""
    img = np.zeros((h, w), dtype=np.float32)
    cx  = w * rng.uniform(0.42, 0.58)
    cy  = h * rng.uniform(0.42, 0.58)
    freq = rng.uniform(7, 11)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    dx, dy = xs - cx, ys - cy
    r     = np.sqrt(dx**2 + dy**2 + 1e-6)
    theta = np.arctan2(dy, dx)
    img   = 0.5 + 0.5 * np.sin(r / freq + theta * 0.8)
    return img.astype(np.float32)


def _whorl(rng, h, w):
    """Concentric whorl — tight spiral from centre."""
    cx, cy = w / 2, h / 2
    freq   = rng.uniform(5, 9)
    spiral = rng.uniform(0.4, 0.8)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    r      = np.sqrt((xs - cx)**2 + (ys - cy)**2 + 1e-6)
    theta  = np.arctan2(ys - cy, xs - cx)
    img    = 0.5 + 0.5 * np.sin(r / freq - theta * spiral * 2)
    return img.astype(np.float32)


def _arch(rng, h, w):
    """Plain arch — gently curved horizontal ridges, no delta."""
    freq  = rng.uniform(9, 14)
    curve = rng.uniform(0.008, 0.025)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xr    = xs + curve * (ys - h / 2) ** 2
    img   = 0.5 + 0.5 * np.sin(2 * np.pi * xr / freq)
    return img.astype(np.float32)


def _tented_arch(rng, h, w):
    """Tented arch — sharp central upthrust."""
    freq   = rng.uniform(7, 12)
    peak_x = w * rng.uniform(0.42, 0.58)
    k      = rng.uniform(0.06, 0.14)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    tent   = np.abs(xs - peak_x) * k
    img    = 0.5 + 0.5 * np.sin(2 * np.pi * (xs + tent) / freq - ys * 0.04)
    return img.astype(np.float32)


def _double_loop(rng, h, w):
    """Double loop / twin loop — two opposing spiral cores."""
    cx1, cy1 = w * rng.uniform(0.25, 0.38), h * 0.5
    cx2, cy2 = w * rng.uniform(0.62, 0.75), h * 0.5
    freq = rng.uniform(7, 11)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    r1 = np.sqrt((xs - cx1)**2 + (ys - cy1)**2 + 1e-6)
    r2 = np.sqrt((xs - cx2)**2 + (ys - cy2)**2 + 1e-6)
    t1 = np.arctan2(ys - cy1, xs - cx1)
    t2 = np.arctan2(ys - cy2, xs - cx2)
    img = 0.5 + 0.25 * np.sin(r1 / freq + t1) + 0.25 * np.sin(r2 / freq - t2)
    return img.astype(np.float32)


_PATTERNS = [_loop, _whorl, _arch, _tented_arch, _double_loop]


def make_fingerprint(identity_seed: int, impression_seed: int,
                     noise: float = 0.04) -> np.ndarray:
    """
    Generate a 96×96 synthetic fingerprint image.

    identity_seed  → determines ridge pattern TYPE + global shape parameters
    impression_seed → small per-impression jitter (rotation ±3°, noise)
    """
    h, w = 96, 96
    id_rng  = np.random.default_rng(identity_seed * 997 + 13)
    im_rng  = np.random.default_rng(identity_seed * 997 + impression_seed * 31 + 7)

    pattern = _PATTERNS[identity_seed % len(_PATTERNS)]
    img     = pattern(id_rng, h, w)

    # Per-impression: small rotation + brightness jitter + noise
    angle = im_rng.uniform(-3.0, 3.0)
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    img  *= im_rng.uniform(0.92, 1.08)
    img  += im_rng.normal(0, noise, (h, w)).astype(np.float32)
    img   = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def save_images(identity_seed: int, count: int = 3,
                noise: float = 0.04, imp_offset: int = 0) -> list[Path]:
    """Save `count` impressions of one identity to a temp directory."""
    tmp   = Path(tempfile.mkdtemp())
    paths = []
    for i in range(count):
        img  = make_fingerprint(identity_seed,
                                impression_seed=imp_offset + i,
                                noise=noise)
        path = tmp / f"fp_id{identity_seed:02d}_imp{imp_offset+i:02d}.bmp"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(preproc, extractor, aes_cls, db_cls, enroll_cls, auth_cls,
              n_ids: int = 5) -> float:
    """
    Enroll n_ids throwaway identities, measure genuine + impostor cosine
    scores, and return the midpoint as the operating threshold.
    """
    from crypto   import AESCryptosystem
    from database import FingerprintDatabase
    from enroll   import EnrollmentPipeline
    from authenticate import AuthenticationPipeline

    cal_db  = FingerprintDatabase(db_path=Path(tempfile.mktemp(suffix=".db")))
    cal_aes = AESCryptosystem()
    cal_enr = EnrollmentPipeline(preproc, extractor, cal_aes, cal_db)
    cal_aut = AuthenticationPipeline(preproc, extractor, cal_aes, cal_db)

    # Use identity seeds 20–24 (won't clash with demo seeds 0,1,3)
    ids = list(range(20, 20 + n_ids))
    for sid in ids:
        paths = save_images(sid, count=3, noise=0.04)
        cal_enr.enroll(f"c{sid}", paths)

    genuine_scores, impostor_scores = [], []
    for sid in ids:
        q = save_images(sid, count=1, noise=0.05, imp_offset=10)[0]
        genuine_scores.append(cal_aut.authenticate(f"c{sid}", q).cosine_sim)

    for i, sid in enumerate(ids):
        imp_sid = ids[(i + 1) % len(ids)]
        q = save_images(imp_sid, count=1, noise=0.04, imp_offset=11)[0]
        impostor_scores.append(cal_aut.authenticate(f"c{sid}", q).cosine_sim)

    mean_g = float(np.mean(genuine_scores))
    mean_i = float(np.mean(impostor_scores))
    sep    = mean_g - mean_i
    thresh = round((mean_g + mean_i) / 2, 4)

    print(f"  genuine mean  = {mean_g:.4f}")
    print(f"  impostor mean = {mean_i:.4f}")
    print(f"  separability  = {sep:.4f}  ({'good' if sep > 0.05 else 'low — fine-tune CNN for better results'})")
    print(f"  → threshold   = {thresh:.4f}")

    try:
        Path(cal_db.db_path).unlink(missing_ok=True)
    except Exception:
        pass
    return thresh


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo() -> int:
    print("\n" + "═" * 62)
    print("  Fingerprint Biometric Cryptosystem — Demo")
    print("═" * 62)

    # [1] Load components
    print("\n[1/7] Loading pipeline components…")
    from preprocessor      import FingerprintPreprocessor
    from feature_extractor import FingerprintCNNExtractor
    from crypto            import AESCryptosystem
    from database          import FingerprintDatabase
    from enroll            import EnrollmentPipeline
    from authenticate      import AuthenticationPipeline
    import config

    tmp_db    = Path(tempfile.mktemp(suffix=".db"))
    preproc   = FingerprintPreprocessor()
    extractor = FingerprintCNNExtractor()
    aes       = AESCryptosystem()
    db        = FingerprintDatabase(db_path=tmp_db)
    enroller  = EnrollmentPipeline(preproc, extractor, aes, db)
    auth      = AuthenticationPipeline(preproc, extractor, aes, db)
    print("  ✅ Components ready.")

    # [2] Calibrate threshold
    print("\n[2/7] Calibrating threshold against current backbone weights…")
    threshold = calibrate(preproc, extractor, AESCryptosystem,
                           FingerprintDatabase, EnrollmentPipeline,
                           AuthenticationPipeline)
    orig_thresh             = config.COSINE_THRESHOLD
    config.COSINE_THRESHOLD = threshold

    # [3] Generate images
    # identity 0 = loop      (Alice)
    # identity 1 = whorl     (Bob)
    # identity 3 = tented arch (impostor — different class from alice)
    print("\n[3/7] Generating synthetic fingerprint images…")
    alice_enroll = save_images(identity_seed=0, count=3, noise=0.04)
    alice_query  = save_images(identity_seed=0, count=1, noise=0.05, imp_offset=50)[0]
    bob_enroll   = save_images(identity_seed=1, count=3, noise=0.04)
    bob_query    = save_images(identity_seed=1, count=1, noise=0.05, imp_offset=50)[0]
    impostor_img = save_images(identity_seed=3, count=1, noise=0.04)[0]

    print(f"  Alice  (loop pattern)    — {len(alice_enroll)} enrollment images")
    print(f"  Bob    (whorl pattern)   — {len(bob_enroll)} enrollment images")
    print(f"  Impostor (tented arch)   — {impostor_img.name}")

    # [4] Enroll
    print("\n[4/7] Enrolling users…")
    ra = enroller.enroll("alice", alice_enroll)
    rb = enroller.enroll("bob",   bob_enroll)
    print(f"  Alice : {json.dumps(ra)}")
    print(f"  Bob   : {json.dumps(rb)}")

    # [5] Genuine
    print("\n[5/7] Genuine — Alice's fresh impression claiming Alice…")
    r_genuine  = auth.authenticate("alice", alice_query)
    print(f"  {r_genuine}")

    # [6] Impostor
    print("\n[6/7] Impostor — tented-arch stranger claiming Alice…")
    r_impostor = auth.authenticate("alice", impostor_img)
    print(f"  {r_impostor}")

    # [7] Cross-user
    print("\n[7/7] Cross-user — Bob's whorl claiming Alice's loop ID…")
    r_cross    = auth.authenticate("alice", bob_query)
    print(f"  {r_cross}")

    # Score bar chart
    print("\n" + "─" * 62)
    print(f"  Score breakdown  (threshold = {threshold:.4f})")
    print("─" * 62)
    for label, score in [
        ("Genuine  (Alice→Alice)",    r_genuine.cosine_sim),
        ("Impostor (stranger→Alice)", r_impostor.cosine_sim),
        ("Cross    (Bob→Alice)",      r_cross.cosine_sim),
    ]:
        filled = int(score * 40)
        bar    = "█" * filled + "░" * (40 - filled)
        flag   = "✅ ABOVE" if score >= threshold else "❌ BELOW"
        print(f"  {label:<33} {score:.4f}  {flag}  [{bar}]")
    print(f"  {'Threshold':<33} {threshold:.4f}")

    # Enrolled users
    print("\n" + "─" * 62)
    print("Enrolled users in DB:")
    for u in db.all_user_info():
        print(f"  {u['user_id']:<12}  {u['enrolled_at'][:19]}  ({u['images_used']} images)")

    # Summary
    print("\n" + "═" * 62)
    print("  DEMO SUMMARY")
    print("═" * 62)
    tests = [
        ("Genuine  (Alice → Alice)",     r_genuine.granted,  True),
        ("Impostor (stranger → Alice)",  r_impostor.granted, False),
        ("Cross-user (Bob → Alice)",     r_cross.granted,    False),
    ]
    all_pass = True
    for name, got, expected in tests:
        ok       = (got == expected)
        all_pass = all_pass and ok
        sym      = "✅" if ok else "❌"
        print(f"  {sym}  {name:<37}  "
              f"expected={'GRANT' if expected else 'DENY '}  "
              f"got={'GRANT' if got else 'DENY '}")

    if all_pass:
        print("\n  ✅ ALL TESTS PASSED")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        print("  The ImageNet backbone has limited fingerprint discriminability.")
        print("  Run  python train_full.py --dir data/SOCOFing/Real")
        print("  to fine-tune on real fingerprints and achieve reliable separation.")
    print("═" * 62)

    config.COSINE_THRESHOLD = orig_thresh
    tmp_db.unlink(missing_ok=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_demo())