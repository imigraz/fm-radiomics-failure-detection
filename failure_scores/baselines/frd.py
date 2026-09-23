"""
Rad-L2 and Rad-MD-Img: whole-image radiomics as in FRD (Konz et al. 2026).

Features: frd-score's v1 3D extractor (1014 values per volume, of which ~550
are numeric PyRadiomics diagnostics, as in the published FRD) on the image
alone, after 1 mm isotropic resampling; bilateral volumes are split after
resampling. Features are float32, as in frd-score.

Columns that are not finite after z-scoring the reference are dropped.

    rad_md_img  Mahalanobis++ (L2 normalisation -> Ledoit-Wolf MD²)
    rad_l2      L2 norm of the reference-z-scored feature vector, the per-case
                distance suggested by the FRD authors

Reference cases are scored leave-one-out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.preprocessing import StandardScaler

from ..io import resample_to_isotropic, split_nifti_half, yaml_axis_to_numpy_zyx
from ..scoring import MahalanobisScorer


def extract_site(image_paths: Sequence[Path], work_dir: Path,
                 bilateral_axis=None) -> Tuple[np.ndarray, List[str]]:
    """(features float32 (n, 1014), case_ids) of the images of one site."""
    from frd_score.frd import compute_features, get_feature_extractor

    work_dir.mkdir(parents=True, exist_ok=True)
    paths, ids = [], []
    for src in image_paths:
        base_id = src.name.replace("_0000.nii.gz", "")
        dst = work_dir / src.name
        if not dst.exists():
            img, _ = resample_to_isotropic(sitk.ReadImage(str(src)), None, spacing=1.0)
            sitk.WriteImage(img, str(dst))
        if bilateral_axis is None:
            paths.append(dst)
            ids.append(base_id)
            continue
        for side in ("L", "R"):
            half = work_dir / "bilateral" / f"{base_id}_{side}_0000.nii.gz"
            split_nifti_half(dst, half, yaml_axis_to_numpy_zyx(bilateral_axis), side)
            paths.append(half)
            ids.append(f"{base_id}_{side}")

    feats, _, _ = compute_features(files=[str(p) for p in paths],
                                   feature_extractor=get_feature_extractor(frd_version="v1", image_dim=3),
                                   masks=None, frd_version="v1", verbose=False)
    feats = feats.astype(np.float32)
    ok = ~np.isnan(feats).any(axis=1)
    return feats[ok], [c for c, k in zip(ids, ok) if k]


def keep_columns(ref: np.ndarray) -> np.ndarray:
    """Columns finite after z-scoring the reference with its own statistics."""
    Z = StandardScaler().fit(ref).transform(ref)
    return np.isfinite(Z).all(axis=0)


def _l2(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(StandardScaler().fit(ref).transform(X), axis=1)


def score(features: Dict[str, Tuple[np.ndarray, List[str]]], reference_site: str,
          query_sites: Sequence[str]) -> pd.DataFrame:
    """rad_md_img and rad_l2 for all sites, long format."""
    ref, ref_ids = features[reference_site]
    keep = keep_columns(ref)
    ref = ref[:, keep]

    n = len(ref_ids)
    md, l2 = np.empty(n), np.empty(n)
    for i in range(n):
        # Boolean row mask (C-ordered copy), not np.delete: ``ref`` is
        # F-ordered after the column filter and np.delete would keep that
        # layout, which changes the float32 Ledoit-Wolf fit in the last digits.
        rest = ref[np.arange(n) != i]
        md[i] = MahalanobisScorer("l2").fit(rest).score(ref[[i]])[0]
        # Norm of the (1, d) array without ``axis`` (numpy's dot path);
        # ``axis=1`` can differ in the last bit.
        l2[i] = float(np.linalg.norm(StandardScaler().fit(rest).transform(ref[[i]])))
    rows = [(ref_ids, reference_site, md, l2)]

    scorer = MahalanobisScorer("l2").fit(ref)
    for site in query_sites:
        X, ids = features[site]
        X = X[:, keep]
        rows.append((ids, site, scorer.score(X).astype(float), _l2(ref, X).astype(float)))

    parts = []
    for ids, site, m, l in rows:
        parts.append(pd.DataFrame({"case_id": ids, "site": site, "method": "rad_md_img", "score": m}))
        parts.append(pd.DataFrame({"case_id": ids, "site": site, "method": "rad_l2", "score": l}))
    return pd.concat(parts, ignore_index=True)
