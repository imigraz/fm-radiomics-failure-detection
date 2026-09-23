"""
Mahalanobis distance to the reference site, the scorer behind every "-MD"
method (Rad-MD-Img, Rad-MD-Mask, and the DINOv3 / RadDINO / 3DINO -MD scores).

    norm="l2"      Mahalanobis++ (Müller & Hein, ICML 2025): each feature
                   vector is L2-normalised onto the unit sphere, then scored
                   with a Ledoit-Wolf covariance. Used for the
                   high-dimensional banks (FM embeddings, FRD radiomics).
    norm="zscore"  Standardise with the reference mean / std, then
                   Ledoit-Wolf. Used for the few mask-shape features, where
                   L2 normalisation would discard the magnitude signal.

The score is the squared Mahalanobis distance MD²; higher means farther from
the reference. Query cases are scored against a fit on the whole reference.
Reference cases are scored leave-one-out (``loo_scores``), so that they form
an unbiased null distribution; they are not part of the evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

NORMS = ("l2", "zscore")


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; all-zero rows are left unchanged."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms < 1e-12, 1.0, norms)


class MahalanobisScorer:
    """
    Normalisation + Ledoit-Wolf squared Mahalanobis distance.

    ``fit`` and ``score`` take raw features; the normalisation is applied
    inside, so reference and query always go through the same transform.
    The input dtype is kept (the FM caches are float32).
    """

    def __init__(self, norm: str = "l2"):
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.norm = norm

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.norm == "zscore":
            return self.scaler_.transform(X)
        return l2_normalize(X)

    def fit(self, X_ref: np.ndarray) -> "MahalanobisScorer":
        if self.norm == "zscore":
            self.scaler_ = StandardScaler().fit(X_ref)
        Xt = self._transform(X_ref)
        self.mean_ = Xt.mean(axis=0)
        lw = LedoitWolf().fit(Xt)
        self.precision_ = lw.precision_
        self.shrinkage_ = float(lw.shrinkage_)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of each row to the reference."""
        diff = self._transform(X) - self.mean_
        return np.einsum("ij,jk,ik->i", diff, self.precision_, diff)

    def loo_scores(self, X_ref: np.ndarray) -> np.ndarray:
        """Score each reference case against a fit on the other n - 1 cases."""
        n = len(X_ref)
        out = np.empty(n)
        for i in range(n):
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            out[i] = MahalanobisScorer(self.norm).fit(X_ref[keep]).score(X_ref[[i]])[0]
        return out
