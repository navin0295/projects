"""
preprocessor.py
───────────────
Fingerprint image preprocessing pipeline.

Pipeline
--------
1. Load → grayscale
2. CLAHE ridge enhancement
3. Gaussian denoising
4. Resize to target size
5. Min-max normalisation → float32 [0, 1]
6. Add channel dimension → shape (H, W, 1)

Also exposes binarize() for visualisation and stage-by-stage output
for the visualisation notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import config
from logger import get_logger

log = get_logger("preprocessor")


class FingerprintPreprocessor:
    """
    Stateless preprocessing pipeline.

    All methods are pure functions of their inputs — safe to call from
    multiple threads or processes.

    Parameters
    ----------
    target_size : (W, H) tuple   — cv2 resize convention
    clahe_clip  : CLAHE clip limit (higher → more contrast, more noise)
    clahe_grid  : CLAHE tile grid size
    blur_kernel : Gaussian kernel size (must be odd)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = config.IMG_SIZE,
        clahe_clip:  float           = config.CLAHE_CLIP,
        clahe_grid:  Tuple[int, int] = config.CLAHE_GRID,
        blur_kernel: int             = config.BLUR_KERNEL,
    ):
        self.target_size = target_size
        self.clahe       = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    # ── low-level ops ──────────────────────────────────────────────────────────

    def load_gray(self, path: str | Path) -> np.ndarray:
        """Load any image file and return as uint8 grayscale."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def enhance_ridges(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement — makes ridge/valley contrast sharper."""
        return self.clahe.apply(gray)

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blur to suppress salt-and-pepper noise."""
        k = self.blur_kernel
        return cv2.GaussianBlur(img, (k, k), 0)

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Lanczos resize to target size."""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Min-max normalize to float32 [0, 1]."""
        img_f = img.astype(np.float32)
        mn, mx = img_f.min(), img_f.max()
        if mx - mn > 1e-6:
            return (img_f - mn) / (mx - mn)
        return np.zeros_like(img_f)

    def binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive Gaussian threshold → uint8 binary image.
        Used for visualisation only, NOT fed into the CNN.
        """
        return cv2.adaptiveThreshold(
            img.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )

    # ── main API ───────────────────────────────────────────────────────────────

    def process(
        self,
        path: str | Path,
        return_stages: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Full pipeline for a single image.

        Parameters
        ----------
        path          : path to a fingerprint image (BMP, PNG, JPEG …)
        return_stages : if True, also return intermediate stages dict

        Returns
        -------
        tensor : np.ndarray  shape (H, W, 1)  float32  [0, 1]
        stages : dict  (only when return_stages=True)
        """
        gray     = self.load_gray(path)
        enhanced = self.enhance_ridges(gray)
        denoised = self.denoise(enhanced)
        resized  = self.resize(denoised)
        normed   = self.normalize(resized)
        tensor   = normed[:, :, np.newaxis]          # (H, W, 1)

        if return_stages:
            return tensor, {
                "1_original"  : gray,
                "2_enhanced"  : enhanced,
                "3_denoised"  : denoised,
                "4_binarized" : self.binarize(self.resize(denoised)),
                "5_final"     : resized,
            }
        return tensor

    def process_from_bytes(self, data: bytes) -> np.ndarray:
        """
        Process a raw image from bytes (e.g. uploaded via HTTP or read from DB blob).

        Returns tensor shape (H, W, 1)  float32  [0, 1].
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image bytes.")
        enhanced = self.enhance_ridges(img)
        denoised = self.denoise(enhanced)
        resized  = self.resize(denoised)
        return self.normalize(resized)[:, :, np.newaxis]

    def batch_process(
        self,
        paths: List[str | Path],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Preprocess a list of image paths.

        Returns
        -------
        np.ndarray  shape (N, H, W, 1)  float32  [0, 1]
        """
        it = tqdm(paths, desc="Preprocessing") if show_progress else paths
        tensors = []
        for p in it:
            try:
                tensors.append(self.process(p))
            except FileNotFoundError as exc:
                log.warning("Skipping missing file: %s", exc)
        if not tensors:
            raise ValueError("No valid images were processed.")
        return np.stack(tensors)
