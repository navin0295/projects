"""
main.py
───────
Command-line interface for the Fingerprint Biometric Cryptosystem.

Usage examples
--------------
# Enroll a user with 3 fingerprint images
python main.py enroll --user alice --images img1.bmp img2.bmp img3.bmp

# Verify a query fingerprint
python main.py verify --user alice --image query.bmp

# List all enrolled users
python main.py list-users

# Show info for a specific user
python main.py user-info --user alice

# Delete a user
python main.py delete --user alice

# Bulk enroll from a SOCOFing directory
python main.py bulk-enroll --dir ./data/SOCOFing/Real --subjects 10 --images-per 3

# Run evaluation
python main.py evaluate --subjects 5 --genuine 20 --impostor 20

# Export / import database
python main.py export --out backup.json
python main.py import-db --file backup.json

# Train the CNN
python main.py train --dir ./data/SOCOFing/Real --epochs 10
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# ── shared component factory ──────────────────────────────────────────────────

def _build_components():
    """Instantiate all pipeline components (shared across commands)."""
    from preprocessor      import FingerprintPreprocessor
    from feature_extractor import FingerprintCNNExtractor
    from crypto            import AESCryptosystem
    from database          import FingerprintDatabase
    from enroll            import EnrollmentPipeline
    from authenticate      import AuthenticationPipeline

    preprocessor = FingerprintPreprocessor()
    extractor    = FingerprintCNNExtractor()
    aes          = AESCryptosystem()
    db           = FingerprintDatabase()
    enroller     = EnrollmentPipeline(preprocessor, extractor, aes, db)
    authenticator = AuthenticationPipeline(preprocessor, extractor, aes, db)

    return preprocessor, extractor, aes, db, enroller, authenticator


# ── command handlers ──────────────────────────────────────────────────────────

def cmd_enroll(args):
    _, _, _, _, enroller, _ = _build_components()
    result = enroller.enroll(
        user_id     = args.user,
        image_paths = args.images,
        overwrite   = args.overwrite,
    )
    print(json.dumps(result, indent=2))


def cmd_verify(args):
    _, _, _, _, _, authenticator = _build_components()
    result = authenticator.authenticate(
        user_id    = args.user,
        image_path = args.image,
    )
    print(result)
    sys.exit(0 if result.granted else 1)


def cmd_list_users(args):
    from database import FingerprintDatabase
    db    = FingerprintDatabase()
    users = db.all_user_info()
    if not users:
        print("No users enrolled.")
        return
    print(f"\n{'USER ID':<25}  {'ENROLLED AT':<30}  {'IMAGES'}")
    print("─" * 70)
    for u in users:
        print(f"{u['user_id']:<25}  {u['enrolled_at']:<30}  {u['images_used']}")
    print(f"\nTotal: {len(users)} user(s)")


def cmd_user_info(args):
    from database import FingerprintDatabase
    db   = FingerprintDatabase()
    info = db.user_info(args.user)
    if info is None:
        print(f"User '{args.user}' not found.")
        sys.exit(1)
    print(json.dumps(info, indent=2))


def cmd_delete(args):
    from database import FingerprintDatabase
    db = FingerprintDatabase()
    if not args.force:
        confirm = input(f"Delete user '{args.user}'? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return
    deleted = db.delete(args.user)
    print("Deleted." if deleted else f"User '{args.user}' not found.")


def cmd_bulk_enroll(args):
    """Enroll multiple SOCOFing subjects automatically."""
    import re
    from preprocessor      import FingerprintPreprocessor
    from feature_extractor import FingerprintCNNExtractor
    from crypto            import AESCryptosystem
    from database          import FingerprintDatabase
    from enroll            import EnrollmentPipeline

    real_dir = Path(args.dir)
    if not real_dir.exists():
        print(f"Directory not found: {real_dir}")
        sys.exit(1)

    # Group images by subject
    all_imgs = sorted(glob.glob(str(real_dir / "*.BMP")))
    if not all_imgs:
        all_imgs = sorted(glob.glob(str(real_dir / "*.png")))
    if not all_imgs:
        print("No .BMP or .png files found.")
        sys.exit(1)

    subject_imgs: dict = {}
    for p in all_imgs:
        fname = Path(p).stem
        sid   = int(fname.split("__")[0]) if "__" in fname else int(re.split(r"[^0-9]", fname)[0])
        subject_imgs.setdefault(sid, []).append(p)

    subjects_to_enroll = sorted(subject_imgs.keys())[:args.subjects]

    preprocessor = FingerprintPreprocessor()
    extractor    = FingerprintCNNExtractor()
    aes          = AESCryptosystem()
    db           = FingerprintDatabase()
    enroller     = EnrollmentPipeline(preprocessor, extractor, aes, db)

    enrolled, skipped, failed = 0, 0, 0
    for sid in subjects_to_enroll:
        uid    = f"user_{sid:04d}"
        images = subject_imgs[sid][: args.images_per]
        if db.exists(uid) and not args.overwrite:
            skipped += 1
            continue
        try:
            enroller.enroll(uid, images, overwrite=args.overwrite)
            enrolled += 1
        except Exception as exc:
            print(f"  FAILED {uid}: {exc}")
            failed += 1

    print(f"\nBulk enrollment: {enrolled} enrolled, {skipped} skipped, {failed} failed.")
    print(f"Total in DB: {db.user_count()}")


def cmd_evaluate(args):
    import glob, re
    from preprocessor      import FingerprintPreprocessor
    from feature_extractor import FingerprintCNNExtractor
    from crypto            import AESCryptosystem
    from database          import FingerprintDatabase
    from authenticate      import AuthenticationPipeline
    from evaluate          import BiometricEvaluator

    db = FingerprintDatabase()
    enrolled_users = db.list_users()
    if not enrolled_users:
        print("No enrolled users. Run bulk-enroll first.")
        sys.exit(1)

    # Load subject images from SOCOFing
    import config
    real_dir = config.SOCOFING_REAL
    if not real_dir.exists():
        print(f"SOCOFing dataset not found at {real_dir}. Set DATA_DIR in config.py.")
        sys.exit(1)

    all_imgs = sorted(glob.glob(str(real_dir / "*.BMP")))
    subject_imgs: dict = {}
    for p in all_imgs:
        fname = Path(p).stem
        sid   = int(fname.split("__")[0])
        subject_imgs.setdefault(sid, []).append(p)

    preprocessor  = FingerprintPreprocessor()
    extractor     = FingerprintCNNExtractor()
    aes           = AESCryptosystem()
    authenticator = AuthenticationPipeline(preprocessor, extractor, aes, db)

    # Build uid → subject_id mapping that works for ANY user ID format.
    # Strategy: for each enrolled user, find the subject whose images
    # are most likely to match by extracting any integer from the uid and
    # checking if it exists in subject_imgs.  Falls back gracefully if the
    # uid uses a non-numeric format (e.g. "alice", "user_navin1").
    import re as _re
    uid_to_sid: dict = {}
    selected_users = enrolled_users[:args.subjects]
    for uid in selected_users:
        nums = _re.findall(r"\d+", uid)
        for n in reversed(nums):           # try last number first
            sid = int(n)
            if sid in subject_imgs:
                uid_to_sid[uid] = sid
                break
    mappable = list(uid_to_sid.keys())
    if not mappable:
        print("\nNo enrolled users could be mapped to SOCOFing subject IDs.")
        print("Evaluation requires users enrolled via bulk-enroll with")
        print("'user_NNNN' IDs that match SOCOFing subject numbers.\n")
        print("Currently enrolled users:", selected_users[:10])
        sys.exit(1)

    print(f"Evaluating {len(mappable)} users (out of {len(selected_users)} selected).")
    evaluator = BiometricEvaluator(
        authenticator, subject_imgs, mappable,
        uid_to_sid=uid_to_sid
    )

    metrics = evaluator.run(n_genuine=args.genuine, n_impostor=args.impostor)
    if not metrics:
        print("Evaluation returned no results.")
        sys.exit(1)
    evaluator.plot(metrics, save_path=Path("evaluation_results.png"))
    print("Plot saved → evaluation_results.png")


def cmd_export(args):
    from database import FingerprintDatabase
    db = FingerprintDatabase()
    db.export_encrypted(Path(args.out))
    print(f"Exported {db.user_count()} user(s) → {args.out}")


def cmd_import_db(args):
    from database import FingerprintDatabase
    db = FingerprintDatabase()
    db.import_encrypted(Path(args.file))


def cmd_train(args):
    import glob, re
    from preprocessor      import FingerprintPreprocessor
    from feature_extractor import FingerprintCNNExtractor

    real_dir = Path(args.dir)
    all_imgs = sorted(glob.glob(str(real_dir / "*.BMP")))
    subject_imgs: dict = {}
    for p in all_imgs:
        fname = Path(p).stem
        sid   = int(fname.split("__")[0])
        subject_imgs.setdefault(sid, []).append(p)

    preprocessor = FingerprintPreprocessor()
    extractor    = FingerprintCNNExtractor()
    extractor.train(subject_imgs, preprocessor, epochs=args.epochs)
    print("Training complete. Weights saved.")


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingerprint_system",
        description="ML-Assisted Fingerprint Biometric Cryptosystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a user")
    p_enroll.add_argument("--user",      required=True,        help="User ID")
    p_enroll.add_argument("--images",    required=True, nargs="+", help="Fingerprint image paths")
    p_enroll.add_argument("--overwrite", action="store_true",  help="Re-enroll if user exists")

    # verify
    p_verify = sub.add_parser("verify", help="Verify a fingerprint against stored template")
    p_verify.add_argument("--user",  required=True, help="Claimed user ID")
    p_verify.add_argument("--image", required=True, help="Query fingerprint image path")

    # list-users
    sub.add_parser("list-users", help="List all enrolled users")

    # user-info
    p_info = sub.add_parser("user-info", help="Show info for a user")
    p_info.add_argument("--user", required=True, help="User ID")

    # delete
    p_del = sub.add_parser("delete", help="Delete an enrolled user")
    p_del.add_argument("--user",  required=True, help="User ID")
    p_del.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # bulk-enroll
    p_bulk = sub.add_parser("bulk-enroll", help="Enroll multiple SOCOFing subjects")
    p_bulk.add_argument("--dir",        required=True, help="Path to SOCOFing/Real/")
    p_bulk.add_argument("--subjects",   type=int, default=10, help="Number of subjects to enroll")
    p_bulk.add_argument("--images-per", type=int, default=3,  help="Images per subject")
    p_bulk.add_argument("--overwrite",  action="store_true",  help="Re-enroll existing users")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run FAR/FRR/EER evaluation")
    p_eval.add_argument("--subjects",  type=int, default=5,  help="Enrolled subjects to test")
    p_eval.add_argument("--genuine",   type=int, default=50, help="Genuine trial count")
    p_eval.add_argument("--impostor",  type=int, default=50, help="Impostor trial count")

    # export
    p_exp = sub.add_parser("export", help="Export encrypted DB to JSON")
    p_exp.add_argument("--out", required=True, help="Output JSON file path")

    # import-db
    p_imp = sub.add_parser("import-db", help="Import encrypted DB from JSON")
    p_imp.add_argument("--file", required=True, help="Input JSON file path")

    # train
    p_train = sub.add_parser("train", help="Fine-tune CNN on SOCOFing with triplet loss")
    p_train.add_argument("--dir",    required=True,        help="Path to SOCOFing/Real/")
    p_train.add_argument("--epochs", type=int, default=10, help="Training epochs")

    return parser


# ── entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "enroll"      : cmd_enroll,
    "verify"      : cmd_verify,
    "list-users"  : cmd_list_users,
    "user-info"   : cmd_user_info,
    "delete"      : cmd_delete,
    "bulk-enroll" : cmd_bulk_enroll,
    "evaluate"    : cmd_evaluate,
    "export"      : cmd_export,
    "import-db"   : cmd_import_db,
    "train"       : cmd_train,
}

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    COMMANDS[args.command](args)