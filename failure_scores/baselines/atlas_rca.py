"""
Atlas-RCA (Valindria et al. 2017): reverse classification accuracy by
registration.

For a query case (image I, predicted mask S): register I to every reference
image J_k (ANTs SyNRA: rigid + affine + SyN, both resampled to 1 mm if not
within 10 %), warp S into J_k's space (nearest neighbour), and compute the
Dice against J_k's ground truth. The quality estimate is the maximum over the
references; the failure score is its negation. All references are used (no
retrieval) and images are not cropped.

Bilateral sites (``atlas_flip_r_half``): R halves are flipped to L orientation
before registration, since the references are left structures and SyN cannot
mirror (set for the hippocampus benchmark). Dice is computed on label 1.

Registrations take hours, so each query case's per-reference Dice is
cached in ``<cache_dir>/atlas_rca_{site}_{case_id}.csv`` (``ref_case_id,
dsc, reg_seconds``); only missing pairs are registered. Cases with an empty
mask are skipped without registering.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..io import (case_base_id, find_images, gt_mask_path, split_nifti_half,
                  yaml_axis_to_numpy_zyx)

TRANSFORM = "SyNRA"


def reference_cases(dataset_dir: Path, held_out: Dict[str, str]) -> List[dict]:
    """Reference cases with a fold assignment: ``case_id, img_path, gt_path``."""
    images = {case_base_id(p): p for p in find_images(dataset_dir)}
    cases = []
    for case_id in sorted(held_out):
        img = images.get(case_id)
        gt = gt_mask_path(dataset_dir, img) if img is not None else None
        if gt is not None:
            cases.append({"case_id": case_id, "img_path": img, "gt_path": gt})
    return cases


def query_cases(site: str, dataset_dir: Path, mv_dir: Path, tmp_dir: Path,
                bilateral_axis: Optional[int] = None) -> List[dict]:
    """Query cases with their majority-vote mask: ``case_id, img_path, pred_path``."""
    cases = []
    for img in find_images(dataset_dir):
        base_id = case_base_id(img)
        mv = mv_dir / f"{site}_{base_id}_mv.nii.gz"
        if not mv.exists():
            continue
        if bilateral_axis is None:
            cases.append({"case_id": base_id, "img_path": img, "pred_path": mv})
            continue
        ax = yaml_axis_to_numpy_zyx(bilateral_axis)
        for side in ("L", "R"):
            img_h = tmp_dir / f"{base_id}_0000_{side}_img.nii.gz"
            mv_h = tmp_dir / f"{base_id}_0000_{side}_mv.nii.gz"
            split_nifti_half(img, img_h, ax, side)
            split_nifti_half(mv, mv_h, ax, side, is_mask=True)
            cases.append({"case_id": f"{base_id}_{side}", "img_path": img_h, "pred_path": mv_h})
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def _load(img_path: Path, mask_path: Path):
    import ants
    return (ants.image_read(str(img_path), pixeltype="float"),
            ants.image_read(str(mask_path), pixeltype="float"))


def _resample(img, mask, spacing: float = 1.0):
    """1 mm isotropic (linear / nearest neighbour), unless within 10 %."""
    import ants
    if all(abs(s - spacing) / spacing < 0.10 for s in img.spacing):
        return img, mask
    sp = (spacing,) * img.dimension
    return (ants.resample_image(img, sp, use_voxels=False, interp_type=0),
            ants.resample_image(mask, sp, use_voxels=False, interp_type=1))


def _flip_lr(img, mask):
    """Mirror along ANTs axis 0 (x = left-right), keeping the header."""
    return (img.new_image_like(img.numpy()[::-1, :, :].copy()),
            mask.new_image_like(mask.numpy()[::-1, :, :].copy()))


def register_and_warp(moving_img, moving_mask, fixed_img, transform: str = TRANSFORM):
    import ants
    tmp = tempfile.mkdtemp(prefix="atlas_rca_")
    reg = None
    try:
        reg = ants.registration(fixed=fixed_img, moving=moving_img,
                                type_of_transform=transform, outprefix=f"{tmp}/reg_",
                                verbose=False)
        return ants.apply_transforms(fixed=fixed_img, moving=moving_mask,
                                     transformlist=reg["fwdtransforms"],
                                     interpolator="nearestNeighbor")
    except Exception as e:
        print(f"    registration failed: {type(e).__name__}: {e}")
        return None
    finally:
        del reg
        gc.collect()   # ANTs objects hold large fields; free them per call
        shutil.rmtree(tmp, ignore_errors=True)


def warped_dice(warped: np.ndarray, gt: np.ndarray) -> float:
    """Dice of label 1; NaN if either mask is empty."""
    p, g = warped.astype(np.int16) == 1, gt.astype(np.int16) == 1
    if not p.any() or not g.any():
        return float("nan")
    return 2.0 * int((p & g).sum()) / (int(p.sum()) + int(g.sum()))


def case_dsc(case: dict, refs: Sequence[dict], cache_path: Path, flip_right: bool,
             spacing: float = 1.0) -> Dict[str, float]:
    """Dice against every reference, from the cache or by registration."""
    cached: Dict[str, Dict[str, float]] = {}
    if cache_path.exists():
        for _, r in pd.read_csv(cache_path).iterrows():
            cached[str(r["ref_case_id"])] = {"dsc": float(r["dsc"]),
                                             "reg_seconds": float(r.get("reg_seconds", np.nan))}
    missing = [r for r in refs if r["case_id"] not in cached]
    if missing:
        img, pred = _load(case["img_path"], case["pred_path"])
        if flip_right and case["case_id"].endswith("_R"):
            img, pred = _flip_lr(img, pred)
        img, pred = _resample(img, pred, spacing)
        for ref in missing:
            ref_img, ref_gt = _resample(*_load(ref["img_path"], ref["gt_path"]), spacing)
            t0 = time.perf_counter()
            warped = register_and_warp(img, pred, ref_img)
            secs = time.perf_counter() - t0
            dsc = float("nan") if warped is None else \
                warped_dice(warped.numpy(), ref_gt.numpy())
            cached[ref["case_id"]] = {"dsc": dsc, "reg_seconds": secs}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ref_case_id": k, **v} for k, v in cached.items()]).to_csv(
            cache_path, index=False)
    return {r["case_id"]: cached[r["case_id"]]["dsc"] for r in refs}


def score_site(site: str, cases: Sequence[dict], refs: Sequence[dict], cache_dir: Path,
               flip_right: bool) -> pd.DataFrame:
    """Atlas-RCA failure scores (``-max Dice``) of one query site."""
    rows = []
    for i, case in enumerate(cases):
        # Keep the image alive while its array view is read.
        pred = sitk.ReadImage(str(case["pred_path"]))
        if not sitk.GetArrayViewFromImage(pred).any():
            # Nothing to register: an empty mask has no Dice with any reference.
            print(f"  [{site}] {case['case_id']}: empty mask, skipped")
            continue
        dsc = case_dsc(case, refs, cache_dir / f"atlas_rca_{site}_{case['case_id']}.csv",
                       flip_right)
        finite = [d for d in dsc.values() if np.isfinite(d)]
        if not finite:
            print(f"  [{site}] {case['case_id']}: no valid registration, skipped")
            continue
        rows.append({"case_id": case["case_id"], "site": site, "method": "atlas_rca",
                     "score": -float(np.max(finite))})
    return pd.DataFrame(rows, columns=["case_id", "site", "method", "score"])
