"""
Rad-MD-Mask: Mahalanobis distance of the predicted mask's 3D shape.

1. Extraction. PyRadiomics shape features (14, original image type) of the
   reference ground-truth masks and the query majority-vote masks, after
   1 mm isotropic resampling (``resampledPixelSpacing``). Bilateral sites are
   split into L/R halves first (no flip: shape features are mirror-invariant).
2. Selection, on the reference only:
     Filter: drop near-constant features (coefficient of variation < CV_MIN)
             and heavy-tailed ones (excess kurtosis > KURTOSIS_MAX);
     Prune:  rank the rest by stability (1 / std, most stable first) and
             greedily drop the feature with the highest mean |r| while any
             pair has |r| > R_CORR_THRESHOLD.
   Selected: prostate 6, hippocampus 11 features.
3. Score: z-score with the reference statistics, then Ledoit-Wolf MD²
   (``MahalanobisScorer("zscore")``); LOO for the reference cases.

The paper's original experiment code used a 5th-percentile cut on the variance
of all 2632 radiomic features (shape, in-mask texture, whole-image) as the filter.
That rule depends on feature units and unrelated features; the CV rule used
here selects the identical shape sets on both benchmarks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats

from .io import (case_base_id, find_images, gt_mask_path, mv_mask_path, split_nifti_half,
                 yaml_axis_to_numpy_zyx)
from .scoring import MahalanobisScorer

CV_MIN = 0.01
KURTOSIS_MAX = 5.0
R_CORR_THRESHOLD = 0.90

RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "normalize": True,
    "normalizeScale": 100,
    "removeOutliers": 3,
    "interpolator": "sitkBSpline",
    "resampledPixelSpacing": [1.0, 1.0, 1.0],
    "padDistance": 10,
}


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_extractor():
    from radiomics import featureextractor
    ex = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("shape")
    ex.disableAllImageTypes()
    ex.enableImageTypeByName("Original")
    return ex


def _binarise(mask_path: Path, out_path: Path) -> bool:
    """Write (mask > 0) with the mask's dtype and geometry; False if empty."""
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)
    binary = (arr > 0).astype(arr.dtype)
    if binary.sum() == 0:
        return False
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(out_path))
    return True


def extract_case(img_path: Path, mask_path: Path, extractor) -> Optional[dict]:
    """Shape features ``original_shape_*`` of one case; None if the mask is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        mask_bin = Path(tmp) / "mask.nii.gz"
        if not _binarise(mask_path, mask_bin):
            return None
        result = extractor.execute(str(img_path), str(mask_bin), label=1)
    return {k: float(v) for k, v in result.items() if k.startswith("original_shape_")}


def extract_site(site: str, dataset_dir: Path, is_reference: bool, mv_dir: Path,
                 extractor, bilateral_axis: Optional[int] = None) -> pd.DataFrame:
    """One row per case: ``case_id, site`` and the shape features."""
    rows = []
    for img_path in find_images(dataset_dir):
        base_id = case_base_id(img_path)
        mask_path = gt_mask_path(dataset_dir, img_path) if is_reference \
            else mv_mask_path(mv_dir, site, base_id)
        if mask_path is None or not mask_path.exists():
            continue
        if bilateral_axis is None:
            halves = [(base_id, img_path, mask_path)]
        else:
            halves = []
            ax = yaml_axis_to_numpy_zyx(bilateral_axis)
        with tempfile.TemporaryDirectory() as tmp:
            if bilateral_axis is not None:
                for side in ("L", "R"):
                    img_h, mask_h = Path(tmp) / f"img_{side}.nii.gz", Path(tmp) / f"mask_{side}.nii.gz"
                    split_nifti_half(img_path, img_h, ax, side)
                    split_nifti_half(mask_path, mask_h, ax, side, is_mask=True)
                    halves.append((f"{base_id}_{side}", img_h, mask_h))
            for case_id, img, mask in halves:
                try:
                    feats = extract_case(img, mask, extractor)
                except ValueError as e:
                    # PyRadiomics rejects single-voxel and line-shaped masks.
                    print(f"  WARNING [{site}] {case_id}: no shape features ({e})")
                    continue
                if feats is not None:
                    rows.append({"case_id": case_id, "site": site, **feats})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────

def _filled(ref: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return ref[list(cols)].fillna(ref[list(cols)].mean()).values.astype(np.float64)


def filter_features(ref: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """Features passing the CV / kurtosis filter, most stable (1/std) first."""
    X = _filled(ref, cols)
    std = X.std(axis=0)
    cv = std / np.maximum(np.abs(X.mean(axis=0)), 1e-12)
    kurt = stats.kurtosis(X, axis=0, fisher=True, nan_policy="omit")
    passing = [f for f, c, k in zip(cols, cv, kurt) if c >= CV_MIN and k <= KURTOSIS_MAX]
    stability = pd.Series({f: 1.0 / (s + 1e-12) for f, s in zip(cols, std) if f in passing})
    return stability.sort_values(ascending=False).index.tolist()


def correlation_prune(ref: pd.DataFrame, candidates: Sequence[str],
                      r_threshold: float = R_CORR_THRESHOLD) -> List[str]:
    """Greedy: drop the feature with the higher mean |r| of the most correlated pair."""
    if len(candidates) <= 1:
        return list(candidates)
    X = pd.DataFrame(_filled(ref, candidates), columns=list(candidates)).dropna(axis=1)
    while True:
        corr = X.corr().abs()
        arr = corr.values.copy()
        np.fill_diagonal(arr, 0)
        if arr.max() <= r_threshold:
            return list(X.columns)
        r, c = np.unravel_index(arr.argmax(), arr.shape)
        drop = corr.columns[r] if arr[r].mean() >= arr[c].mean() else corr.columns[c]
        X = X.drop(columns=[drop])


def select_features(ref: pd.DataFrame) -> List[str]:
    cols = [c for c in ref.columns if c.startswith("original_shape_")]
    return correlation_prune(ref, filter_features(ref, cols))


# ─────────────────────────────────────────────────────────────────────────────
# Score
# ─────────────────────────────────────────────────────────────────────────────

def score(features: pd.DataFrame, selected: Sequence[str], reference_site: str,
          query_sites: Sequence[str]) -> pd.DataFrame:
    """Rad-MD-Mask, long format; LOO for the reference cases."""
    X = lambda d: d[list(selected)].fillna(0.0).values.astype(np.float64)
    ref = features[features["site"] == reference_site]
    scorer = MahalanobisScorer("zscore").fit(X(ref))
    parts = [pd.DataFrame({"case_id": ref["case_id"].values, "site": reference_site,
                           "method": "rad_md_mask", "score": scorer.loo_scores(X(ref))})]
    for site in query_sites:
        q = features[features["site"] == site]
        if q.empty:
            continue
        parts.append(pd.DataFrame({"case_id": q["case_id"].values, "site": site,
                                   "method": "rad_md_mask", "score": scorer.score(X(q))}))
    return pd.concat(parts, ignore_index=True)


def run(features: pd.DataFrame, reference_site: str, query_sites: Sequence[str],
        out_dir: Path) -> Tuple[List[str], pd.DataFrame]:
    """Select on the reference, score, write ``shape_features.csv`` and ``scores/shape.csv``."""
    selected = select_features(features[features["site"] == reference_site])
    scores = score(features, selected, reference_site, query_sites)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": selected}).to_csv(out_dir / "shape_features.csv", index=False)
    scores.to_csv(out_dir / "scores" / "shape.csv", index=False)
    return selected, scores
