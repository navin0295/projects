"""
camera_preprocessor.py
──────────────────────
Preprocessing pipeline specifically for camera-captured fingerprint photos.

The key difference from the standard preprocessor
--------------------------------------------------
Optical sensor images (SOCOFing):
    Ridges = DARK  (touch glass → high conductance → dark pixels)
    Valleys = BRIGHT

Camera photos (smartphone macro):
    Ridges = BRIGHT (raised surface catches reflected light)
    Valleys = DARK  (shadowed)

This module handles the inversion + crop + enhancement pipeline
to convert camera photos into sensor-compatible images before
passing them to the CNN extractor.

Usage
-----
    from camera_preprocessor import CameraFingerprintPreprocessor
    preproc = CameraFingerprintPreprocessor()
    tensor  = preproc.process("my_finger_photo.jpg")   # (96, 96, 1)

CLI usage
---------
    python camera_preprocessor.py preprocess --input photo.jpg --output processed.bmp
    python camera_preprocessor.py enroll --user navin --images fp1.jpg fp2.jpg fp3.jpg
    python camera_preprocessor.py verify --user navin --image fp4.jpg
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class CameraFingerprintPreprocessor:
    """
    Converts a camera-captured finger photo into a sensor-compatible image.

    Pipeline
    --------
    1. Load → grayscale
    2. Crop to fingertip region  (removes background)
    3. Invert pixel values       (camera: ridges bright → sensor: ridges dark)
    4. CLAHE enhancement         (sharpen ridge-valley contrast)
    5. Gaussian denoising
    6. Normalize → resize to 96x96
    7. Add channel dim → (96, 96, 1) float32 [0, 1]
    """

    def __init__(
        self,
        crop_top:    float = 0.05,
        crop_bottom: float = 0.20,
        crop_left:   float = 0.15,
        crop_right:  float = 0.15,
        clahe_clip:  float = 3.0,
        clahe_grid:  Tuple[int, int] = (8, 8),
        blur_kernel: int   = 3,
        target_size: Tuple[int, int] = (96, 96),
    ):
        self.crop_top    = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left   = crop_left
        self.crop_right  = crop_right
        self.clahe       = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.blur_kernel = blur_kernel
        self.target_size = target_size

    def process(self, path: str | Path) -> np.ndarray:
        """
        Full pipeline. Returns (H, W, 1) float32 [0, 1] tensor.
        """
        img = self._load_gray(path)
        img = self._crop(img)
        img = self._invert(img)
        img = self._enhance(img)
        img = self._denoise(img)
        img = self._resize(img)
        img = self._normalize(img)
        return img[:, :, np.newaxis]

    def process_and_save(self, input_path: str | Path,
                          output_path: str | Path) -> Path:
        """Process and save as BMP. Returns output path."""
        tensor    = self.process(input_path)
        img_uint8 = (tensor[:, :, 0] * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), img_uint8)
        return Path(output_path)

    def process_to_tempfile(self, input_path: str | Path) -> Path:
        """Process and save to a temp BMP. Caller must delete when done."""
        tmp = Path(tempfile.mktemp(suffix=".bmp"))
        return self.process_and_save(input_path, tmp)

    # ── pipeline steps ────────────────────────────────────────────────────────

    def _load_gray(self, path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return img

    def _crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        y1 = int(h * self.crop_top)
        y2 = int(h * (1 - self.crop_bottom))
        x1 = int(w * self.crop_left)
        x2 = int(w * (1 - self.crop_right))
        return img[y1:y2, x1:x2]

    def _invert(self, img: np.ndarray) -> np.ndarray:
        """
        Invert: camera photos have bright ridges, sensors have dark ridges.
        Inversion aligns camera images with the SOCOFing training domain.
        """
        return 255 - img

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        return self.clahe.apply(img)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        mn, mx = img_f.min(), img_f.max()
        return (img_f - mn) / (mx - mn + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_components():
    from preprocessor      import FingerprintPreprocessor
    from feature_extractor import FingerprintCNNExtractor
    from crypto            import AESCryptosystem
    from database          import FingerprintDatabase
    from enroll            import EnrollmentPipeline
    from authenticate      import AuthenticationPipeline

    preproc   = FingerprintPreprocessor()
    extractor = FingerprintCNNExtractor()
    aes       = AESCryptosystem()
    db        = FingerprintDatabase()
    enroller  = EnrollmentPipeline(preproc, extractor, aes, db)
    auth      = AuthenticationPipeline(preproc, extractor, aes, db)
    return preproc, extractor, aes, db, enroller, auth


def cmd_preprocess(args):
    cam = CameraFingerprintPreprocessor()
    out = args.output or str(Path(args.input).stem) + "_processed.bmp"
    cam.process_and_save(args.input, out)
    print(f"Saved → {out}")


def cmd_enroll(args):
    import json
    cam = CameraFingerprintPreprocessor()
    _, _, _, _, enroller, _ = _build_components()

    tmp_paths = []
    for img_path in args.images:
        tmp = cam.process_to_tempfile(img_path)
        tmp_paths.append(tmp)
        print(f"  Preprocessed: {Path(img_path).name}")

    try:
        result = enroller.enroll(
            user_id     = args.user,
            image_paths = [str(p) for p in tmp_paths],
            overwrite   = args.overwrite,
        )
        print(json.dumps(result, indent=2))
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


def cmd_verify(args):
    import sys
    cam = CameraFingerprintPreprocessor()
    _, _, _, _, _, auth = _build_components()

    tmp = cam.process_to_tempfile(args.image)
    try:
        result = auth.authenticate(args.user, tmp)
        print(result)
        sys.exit(0 if result.granted else 1)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Camera fingerprint preprocessor"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--input",  required=True)
    p_pre.add_argument("--output")

    p_enr = sub.add_parser("enroll")
    p_enr.add_argument("--user",      required=True)
    p_enr.add_argument("--images",    required=True, nargs="+")
    p_enr.add_argument("--overwrite", action="store_true")

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--user",  required=True)
    p_ver.add_argument("--image", required=True)

    args   = parser.parse_args()
    cmds   = {"preprocess": cmd_preprocess, "enroll": cmd_enroll, "verify": cmd_verify}
    cmds[args.command](args)