"""
evaluate.py
───────────
Biometric system evaluation.

Metrics
-------
GAR  — Genuine Acceptance Rate  (1 - FRR)
FAR  — False Acceptance Rate
FRR  — False Rejection Rate
EER  — Equal Error Rate (operating point where FAR ≈ FRR)
AUC  — Area Under the ROC Curve
Key Stability — fraction of genuine attempts that reproduce the same key
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import config
from authenticate import AuthenticationPipeline, AuthResult
from database import FingerprintDatabase
from logger import get_logger

log = get_logger("evaluate")


class BiometricEvaluator:
    """
    Runs genuine and impostor authentication trials and computes
    standard biometric evaluation metrics.

    Parameters
    ----------
    auth_pipeline   : AuthenticationPipeline
    subject_images  : dict mapping subject_id (int) → list of image paths
    enrolled_users  : list of user_id strings already enrolled in the DB
    """

    def __init__(
        self,
        auth_pipeline:  AuthenticationPipeline,
        subject_images: Dict[int, List[str]],
        enrolled_users: List[str],
        seed:           int = 42,
        uid_to_sid:     Optional[Dict[str, int]] = None,
    ):
        """
        Parameters
        ----------
        auth_pipeline  : AuthenticationPipeline
        subject_images : dict  subject_id (int) → list of image paths
        enrolled_users : list of user_id strings in the DB
        uid_to_sid     : optional explicit mapping user_id → subject_id.
                         If None, the evaluator attempts to parse the subject ID
                         from the user_id (works for "user_0001" style IDs).
                         Pass this explicitly when user IDs don't encode subject IDs.
        """
        self.auth          = auth_pipeline
        self.subject_imgs  = subject_images
        self.enrolled      = enrolled_users
        self.rng           = random.Random(seed)

        # Build uid → sid mapping
        if uid_to_sid is not None:
            self._uid_to_sid = uid_to_sid
        else:
            self._uid_to_sid = {}
            import re
            for uid in enrolled_users:
                # Try to extract any integer from the uid
                nums = re.findall(r"\d+", uid)
                if nums:
                    sid = int(nums[-1])   # last number in the uid
                    if sid in subject_images:
                        self._uid_to_sid[uid] = sid

        self._enrolled_sids = set(self._uid_to_sid.values())

    # ── main evaluation ───────────────────────────────────────────────────────

    def run(
        self,
        n_genuine:  int = config.EVAL_GENUINE_N,
        n_impostor: int = config.EVAL_IMPOSTOR_N,
    ) -> Dict:
        """
        Run a balanced evaluation and return a metrics dictionary.

        Parameters
        ----------
        n_genuine  : number of genuine authentication trials
        n_impostor : number of impostor authentication trials

        Returns
        -------
        dict with keys: GAR, FAR, FRR, EER, AUC,
                        genuine_scores, impostor_scores, fpr, tpr
        """
        genuine_scores, impostor_scores = [], []

        # ── genuine trials ────────────────────────────────────────────────────
        # Only use enrolled users whose subject_id we can map to image files
        testable = [u for u in self.enrolled if u in self._uid_to_sid]
        if not testable:
            log.warning(
                "No enrolled users could be mapped to subject_images. "
                "Pass uid_to_sid= explicitly or use 'user_NNNN' format IDs."
            )
            return {}

        log.info("Running %d genuine trials across %d testable users…",
                 n_genuine, len(testable))
        for _ in tqdm(range(n_genuine), desc="Genuine trials"):
            uid  = self.rng.choice(testable)
            sid  = self._uid_to_sid[uid]
            path = self.rng.choice(self.subject_imgs[sid])
            res  = self.auth.authenticate(uid, path)
            genuine_scores.append(res.cosine_sim)

        # ── impostor trials ───────────────────────────────────────────────────
        log.info("Running %d impostor trials…", n_impostor)
        non_enrolled = [s for s in self.subject_imgs
                        if s not in self._enrolled_sids]

        if not non_enrolled:
            # Fall back: use cross-user same-DB attacks
            log.warning("No non-enrolled subjects found; using cross-user impostor trials.")
            non_enrolled = list(self._enrolled_sids)

        for _ in tqdm(range(n_impostor), desc="Impostor trials"):
            target_uid = self.rng.choice(self.enrolled)
            imp_sid    = self.rng.choice(non_enrolled)
            imp_path   = self.rng.choice(self.subject_imgs[imp_sid])
            res        = self.auth.authenticate(target_uid, imp_path)
            impostor_scores.append(res.cosine_sim)

        # ── compute metrics ───────────────────────────────────────────────────
        threshold   = config.COSINE_THRESHOLD
        gar = float(np.mean([s >= threshold for s in genuine_scores]))
        far = float(np.mean([s >= threshold for s in impostor_scores]))
        frr = 1.0 - gar

        all_scores = genuine_scores + impostor_scores
        all_labels = [1] * n_genuine + [0] * n_impostor
        fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr_arr, tpr_arr)

        fnr_arr = 1 - tpr_arr
        eer_idx = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
        eer     = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2)

        metrics = {
            "GAR"             : round(gar, 4),
            "FAR"             : round(far, 4),
            "FRR"             : round(frr, 4),
            "EER"             : round(eer, 4),
            "AUC"             : round(roc_auc, 4),
            "genuine_scores"  : genuine_scores,
            "impostor_scores" : impostor_scores,
            "fpr"             : fpr_arr.tolist(),
            "tpr"             : tpr_arr.tolist(),
        }

        self._print_metrics(metrics)
        return metrics

    # ── pretty print ──────────────────────────────────────────────────────────

    @staticmethod
    def _print_metrics(m: Dict):
        print("\n" + "═" * 45)
        print("  EVALUATION RESULTS")
        print("═" * 45)
        print(f"  Genuine Acceptance Rate (GAR) : {m['GAR']*100:.1f}%")
        print(f"  False Acceptance Rate   (FAR) : {m['FAR']*100:.1f}%")
        print(f"  False Rejection Rate    (FRR) : {m['FRR']*100:.1f}%")
        print(f"  Equal Error Rate        (EER) : {m['EER']*100:.2f}%")
        print(f"  ROC AUC                       : {m['AUC']:.4f}")
        print("═" * 45 + "\n")

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, metrics: Dict, save_path: Optional[Path] = None):
        """
        Plot score distributions and ROC curve side by side.

        Parameters
        ----------
        metrics   : dict returned by run()
        save_path : optional file path to save the figure (PNG/PDF)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Biometric Cryptosystem — Evaluation", fontsize=13)

        # Score distributions
        bins = np.linspace(0, 1, 30)
        ax1.hist(metrics["genuine_scores"],  bins=bins, color="#2196F3",
                 alpha=0.7, label="Genuine", density=True)
        ax1.hist(metrics["impostor_scores"], bins=bins, color="#F44336",
                 alpha=0.7, label="Impostor", density=True)
        ax1.axvline(config.COSINE_THRESHOLD, color="orange", linestyle="--",
                    linewidth=1.5, label=f"Threshold {config.COSINE_THRESHOLD}")
        ax1.set_title("Score Distributions")
        ax1.set_xlabel("Cosine Similarity")
        ax1.set_ylabel("Density")
        ax1.legend()

        # ROC curve
        ax2.plot(metrics["fpr"], metrics["tpr"],
                 color="#4CAF50", lw=2, label=f"AUC = {metrics['AUC']:.3f}")
        ax2.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        ax2.scatter([metrics["FAR"]], [metrics["GAR"]],
                    color="orange", s=100, zorder=5,
                    label=f"EER ≈ {metrics['EER']*100:.1f}%")
        ax2.set_title("ROC Curve")
        ax2.set_xlabel("False Positive Rate (FAR)")
        ax2.set_ylabel("True Positive Rate (GAR)")
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Evaluation plot saved → %s", save_path)
        plt.show()