"""
Ens-Mpd (Roy et al. 2019): mean pairwise Dice between the fold predictions.

For each query case, the mean over all fold pairs (C(5, 2) = 10) of the Dice
between the two fold masks (> 0). Bilateral cases use the matching half of
each fold mask (split in the native grid). High agreement means a confident
ensemble, so the failure score is the negated mean pairwise Dice.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..io import split_array_half, yaml_axis_to_numpy_zyx


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two binary arrays; NaN if both are empty."""
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / denom if denom > 0 else float("nan")


def _fold_masks(case_id: str, fold_pred_dirs: Sequence[Path],
                bilateral_axis: Optional[int]) -> List[np.ndarray]:
    side = case_id[-1] if bilateral_axis is not None else None
    base_id = case_id[:-2] if side is not None else case_id
    masks = []
    for d in fold_pred_dirs:
        path = next((d / f"{base_id}{ext}" for ext in (".nii.gz", ".nii")
                     if (d / f"{base_id}{ext}").exists()), None)
        if path is None:
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
        if side is not None:
            arr = split_array_half(arr, yaml_axis_to_numpy_zyx(bilateral_axis), side)
        masks.append(arr.astype(np.uint8))
    return masks


def mean_pairwise_dice(masks: Sequence[np.ndarray]) -> float:
    if len(masks) < 2:
        return float("nan")
    pairs = [dice(masks[i], masks[j]) for i, j in itertools.combinations(range(len(masks)), 2)]
    if np.isnan(pairs).all():
        return float("nan")   # every fold mask is empty
    return float(np.nanmean(pairs))


def score_site(site: str, case_ids: Sequence[str], fold_dirs: Sequence[Path],
               pred_subdir: str, bilateral_axis: Optional[int] = None) -> pd.DataFrame:
    """Ens-Mpd failure scores (``-mean pairwise Dice``) of one query site."""
    fold_pred_dirs = [fd / pred_subdir for fd in fold_dirs if (fd / pred_subdir).exists()]
    mpd = [mean_pairwise_dice(_fold_masks(c, fold_pred_dirs, bilateral_axis)) for c in case_ids]
    return pd.DataFrame({"case_id": list(case_ids), "site": site, "method": "ens_mpd",
                         "score": -np.asarray(mpd, dtype=float)})
