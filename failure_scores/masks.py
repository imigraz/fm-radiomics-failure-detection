"""
Predicted masks and the Dice that defines the per-case risk (1 - Dice).

The audited model is a five-fold nnU-Net ensemble trained on the reference
site; predictions live in ``<pred_root>/fold_<k>/<pred_subdir>/<case>.nii.gz``.

Masks that get scored:
  reference site  ground-truth mask (the reference describes correct anatomy)
  query sites     majority vote of the fold predictions: a voxel is foreground
                  if >= 3 folds predict it. Built once, cached as
                  ``<mv_dir>/<site>_<case>_mv.nii.gz``, read by every method.

Dice (the risk), as used for the paper:
  reference site  Dice of the fold that held the case out (fold JSONs);
  query sites     mean over the folds of each fold's Dice. This is NOT the
                  Dice of the majority-vote mask.
Dice is computed on label 1 of prediction and ground truth, per bilateral
half where the site is split.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .io import find_labels, split_nifti_half, yaml_axis_to_numpy_zyx

MV_THRESHOLD = 3


def discover_folds(pred_root: Path) -> List[Path]:
    folds = sorted(d for d in pred_root.iterdir() if d.is_dir() and d.name.startswith("fold_"))
    if not folds:
        raise FileNotFoundError(f"no fold_* directories under {pred_root}")
    return folds


def reference_fold_membership(fold_dirs: Sequence[Path], fold_json_name: str,
                              task_name_to_tag: Dict[str, str],
                              reference_site: str) -> Dict[str, str]:
    """{case_id: fold name} of the fold each reference case was validated in."""
    membership: Dict[str, str] = {}
    for fold_dir in fold_dirs:
        path = fold_dir / fold_json_name
        if not path.exists():
            print(f"  WARNING: {path} missing; its reference cases are left out")
            continue
        with open(path) as f:
            raw = json.load(f)
        last_epoch = raw[sorted(raw)[-1]]
        for task, cases in last_epoch.items():
            if task_name_to_tag.get(task) == reference_site:
                for case_id in cases:
                    membership[case_id] = fold_dir.name
    return membership


# ─────────────────────────────────────────────────────────────────────────────
# Majority-vote mask
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(fold_paths: Sequence[Path], threshold: int = MV_THRESHOLD) -> Optional[sitk.Image]:
    """Voxel-wise vote over the fold masks (> 0); geometry of the first fold."""
    ref = sitk.ReadImage(str(fold_paths[0]))
    if len(fold_paths) == 1:
        return ref
    arrays = []
    for p in fold_paths:
        arr = (sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0).astype(np.uint8)
        if arrays and arr.shape != arrays[0].shape:
            continue
        arrays.append(arr)
    if len(arrays) < 2:
        return None
    vote = sitk.GetImageFromArray((np.stack(arrays).sum(axis=0) >= threshold).astype(np.uint8))
    vote.CopyInformation(ref)
    return vote


def build_mv_masks(site: str, fold_dirs: Sequence[Path], pred_subdir: str,
                   mv_dir: Path) -> int:
    """Write the majority-vote mask of every case of a query site (idempotent)."""
    mv_dir.mkdir(parents=True, exist_ok=True)
    case_ids = sorted({p.name.replace(".nii.gz", "")
                       for fd in fold_dirs for p in (fd / pred_subdir).glob("*.nii.gz")})
    written = 0
    for case_id in case_ids:
        out = mv_dir / f"{site}_{case_id}_mv.nii.gz"
        if out.exists():
            continue
        folds = [fd / pred_subdir / f"{case_id}.nii.gz" for fd in fold_dirs]
        mv = majority_vote([p for p in folds if p.exists()])
        if mv is None:
            print(f"  [mv] {site}/{case_id}: vote failed")
            continue
        sitk.WriteImage(mv, str(out))
        written += 1
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Dice
# ─────────────────────────────────────────────────────────────────────────────

def dice_label1(pred_path: Path, gt_path: Path) -> float:
    """Dice of label 1; NaN if either mask is empty."""
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(np.int16) == 1
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.int16) == 1
    if not pred.any() or not gt.any():
        return float("nan")
    return float(2.0 * int((pred & gt).sum()) / (int(pred.sum()) + int(gt.sum())))


def has_labels(dataset_dir: Path) -> bool:
    """Whether a site has ground-truth labels (``labelsTr/`` or ``labels/``)."""
    return bool(find_labels(dataset_dir))


def site_dice(site: str, dataset_dir: Path, fold_dirs: Sequence[Path], pred_subdir: str,
              is_reference: bool, held_out: Optional[Dict[str, str]] = None,
              bilateral_axis: Optional[int] = None) -> pd.DataFrame:
    """
    ``case_id, site, dice`` for every labelled case, sorted by case_id.
    Reference: held-out fold only. Query: mean over the folds with a valid Dice.
    """
    gt_paths = find_labels(dataset_dir)
    sides = ("L", "R") if bilateral_axis is not None else (None,)
    values: Dict[str, List[float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fold_dir in fold_dirs:
            for gt_path in gt_paths:
                base_id = gt_path.name.replace(".nii.gz", "")
                if is_reference and (held_out or {}).get(base_id) != fold_dir.name:
                    continue
                pred_path = fold_dir / pred_subdir / f"{base_id}.nii.gz"
                if not pred_path.exists():
                    continue
                for side in sides:
                    if side is None:
                        case_id, d = base_id, dice_label1(pred_path, gt_path)
                    else:
                        case_id = f"{base_id}_{side}"
                        ax = yaml_axis_to_numpy_zyx(bilateral_axis)
                        split_nifti_half(gt_path, tmp / "gt.nii.gz", ax, side, is_mask=True)
                        split_nifti_half(pred_path, tmp / "pred.nii.gz", ax, side, is_mask=True)
                        d = dice_label1(tmp / "pred.nii.gz", tmp / "gt.nii.gz")
                    if np.isfinite(d):
                        values.setdefault(case_id, []).append(d)
    rows = [{"case_id": c, "site": site, "dice": float(np.mean(v))}
            for c, v in sorted(values.items())]
    return pd.DataFrame(rows, columns=["case_id", "site", "dice"])
