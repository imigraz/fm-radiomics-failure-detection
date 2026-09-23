"""
Volume loading and the spatial preprocessing shared by every method.

  1. Resample to 1 mm isotropic spacing (B-spline for images, nearest
     neighbour for masks), unless all spacings are already within 10 %.
  2. Bilateral split: sites that store a left and a right structure (e.g.
     both hippocampi) in one volume are cut at the midplane of the left-right
     axis into two cases, ``<id>_L`` and ``<id>_R``.
  3. R->L flip (FM paths only): the R half is mirrored along the split axis so
     that it has the orientation of the (left-only) reference site.

Arrays are returned in (H, W, D) = (y, x, z) order: SimpleITK's (z, y, x)
array transposed with (1, 2, 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


# ─────────────────────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────────────────────

def strip_nnunet_suffix(filename: str) -> str:
    """``RUNMC_Case00_0000.nii.gz`` -> ``RUNMC_Case00.nii.gz`` (label file name)."""
    for old, new in (("_0000.nii.gz", ".nii.gz"), ("_0000.nii", ".nii")):
        if filename.endswith(old):
            return filename[: -len(old)] + new
    return filename


def case_base_id(image_path: Path) -> str:
    return strip_nnunet_suffix(image_path.name).replace(".nii.gz", "").replace(".nii", "")


def find_images(dataset_dir: Path) -> List[Path]:
    """Sorted channel-0 images in ``imagesTr/`` (or ``images/``)."""
    for sub in ("imagesTr", "images"):
        imgs = sorted((dataset_dir / sub).glob("*_0000.nii.gz"))
        if imgs:
            return imgs
    return []


def find_labels(dataset_dir: Path) -> List[Path]:
    """Sorted labels in ``labelsTr/`` (or ``labels/``)."""
    for sub in ("labelsTr", "labels"):
        labels = sorted((dataset_dir / sub).glob("*.nii.gz"))
        if labels:
            return labels
    return []


def site_case_ids(dataset_dir: Path, bilateral_axis: Optional[int] = None) -> List[str]:
    """Sorted case ids of a site's images; ``<id>_L``, ``<id>_R`` if split."""
    base_ids = [case_base_id(p) for p in find_images(dataset_dir)]
    if bilateral_axis is None:
        return sorted(base_ids)
    return sorted(f"{b}_{side}" for b in base_ids for side in ("L", "R"))


def gt_mask_path(dataset_dir: Path, image_path: Path) -> Optional[Path]:
    label = strip_nnunet_suffix(image_path.name)
    for sub in ("labelsTr", "labels"):
        p = dataset_dir / sub / label
        if p.exists():
            return p
    return None


def mv_mask_path(mv_dir: Path, site: str, base_id: str) -> Path:
    """Majority-vote mask of the 5 fold predictions (see ``masks.py``)."""
    return mv_dir / f"{site}_{base_id}_mv.nii.gz"


# ─────────────────────────────────────────────────────────────────────────────
# Resampling and loading
# ─────────────────────────────────────────────────────────────────────────────

def _is_close(img: sitk.Image, spacing: float) -> bool:
    """All spacings within 10 % of ``spacing``: resampling is skipped."""
    return all(abs(s - spacing) / max(spacing, 1e-6) < 0.10 for s in img.GetSpacing())


def _resample(volume: sitk.Image, grid: sitk.Image, spacing: float, interp: int) -> sitk.Image:
    """``volume`` resampled to ``spacing`` mm on the extent of ``grid``."""
    new_spacing = [spacing] * grid.GetDimension()
    new_size = [int(round(grid.GetSize()[i] * grid.GetSpacing()[i] / spacing))
                for i in range(grid.GetDimension())]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(new_spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(volume.GetDirection())
    r.SetOutputOrigin(volume.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(interp)
    return r.Execute(volume)


def resample_to_isotropic(
    img: sitk.Image,
    mask: Optional[sitk.Image] = None,
    spacing: float = 1.0,
    skip_if_close: bool = True,
) -> Tuple[sitk.Image, Optional[sitk.Image]]:
    """B-spline (image) / nearest-neighbour (mask) resampling to ``spacing`` mm."""
    if skip_if_close and _is_close(img, spacing):
        return img, mask
    img_out = _resample(img, img, spacing, sitk.sitkBSpline)
    mask_out = _resample(mask, img, spacing, sitk.sitkNearestNeighbor) if mask is not None else None
    return img_out, mask_out


def resample_mask_to_isotropic(mask: sitk.Image, spacing: float = 1.0) -> sitk.Image:
    """Nearest-neighbour resampling of a mask on its own, same rule as above."""
    if _is_close(mask, spacing):
        return mask
    return _resample(mask, mask, spacing, sitk.sitkNearestNeighbor)


def load_volume_pair(
    img_path: Path, mask_path: Path, spacing: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Image (float32) and binary mask (uint8), 1 mm isotropic, (H, W, D)."""
    img, mask = resample_to_isotropic(
        sitk.ReadImage(str(img_path)), sitk.ReadImage(str(mask_path)), spacing=spacing)
    img_arr = sitk.GetArrayFromImage(img).transpose(1, 2, 0).astype(np.float32)
    mask_arr = (sitk.GetArrayFromImage(mask).transpose(1, 2, 0) > 0).astype(np.uint8)
    return img_arr, mask_arr


# ─────────────────────────────────────────────────────────────────────────────
# Bilateral split and flip
# ─────────────────────────────────────────────────────────────────────────────

def yaml_axis_to_numpy_hwd(yaml_axis: int) -> int:
    """Benchmark YAML axis (0 = x = LR, 1 = y, 2 = z) -> (H, W, D) array axis."""
    return {0: 1, 1: 0, 2: 2}.get(yaml_axis, 1)


def yaml_axis_to_numpy_zyx(yaml_axis: int) -> int:
    """Benchmark YAML axis -> axis of SimpleITK's (z, y, x) array."""
    return 2 - yaml_axis


def split_array_half(arr: np.ndarray, axis: int, side: str) -> np.ndarray:
    """First (``"L"``) or second (``"R"``) half of ``arr`` along ``axis``."""
    mid = arr.shape[axis] // 2
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(0, mid) if side == "L" else slice(mid, None)
    return arr[tuple(sl)]


def flip(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.flip(arr, axis=axis).copy()


def split_nifti_half(
    src_path: Path, dst_path: Path, numpy_zyx_axis: int, side: str,
    resample_spacing: Optional[float] = 1.0, is_mask: bool = False,
) -> None:
    """
    Write one half of a NIfTI volume (optionally resampled first) to
    ``dst_path``, keeping spacing and direction and shifting the origin of the
    R half. Used by the stages that read volumes from disk (Dice, shape
    features, FRD, Atlas-RCA). Masks (``is_mask``) are resampled with nearest
    neighbour, images with B-spline, as in ``resample_to_isotropic``.
    """
    img = sitk.ReadImage(str(src_path))
    if resample_spacing is not None:
        img = resample_mask_to_isotropic(img, resample_spacing) if is_mask else \
            resample_to_isotropic(img, spacing=resample_spacing)[0]
    arr = sitk.GetArrayFromImage(img)
    half = sitk.GetImageFromArray(split_array_half(arr, numpy_zyx_axis, side))
    half.SetSpacing(img.GetSpacing())
    half.SetDirection(img.GetDirection())
    origin = list(img.GetOrigin())
    if side == "R":
        ax = (arr.ndim - 1) - numpy_zyx_axis
        origin[ax] += (arr.shape[numpy_zyx_axis] // 2) * img.GetSpacing()[ax]
    half.SetOrigin(tuple(origin))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(half, str(dst_path))


# ─────────────────────────────────────────────────────────────────────────────
# Case iteration for the FM paths
# ─────────────────────────────────────────────────────────────────────────────

def iter_site_cases(
    site: str,
    dataset_dir: Path,
    is_reference: bool,
    mv_dir: Path,
    bilateral_axis: Optional[int] = None,
    flip_right: bool = True,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Yield ``(case_id, img_hwd, mask_hwd)`` for every case of a site.

    The reference site uses its ground-truth masks, query sites the
    majority-vote mask of the ensemble. Cases without a mask are skipped.
    ``bilateral_axis`` (YAML convention) splits each volume into ``_L`` and
    ``_R`` cases; with ``flip_right`` the R half is mirrored to L orientation.
    """
    for img_path in find_images(dataset_dir):
        base_id = case_base_id(img_path)
        if is_reference:
            mask_path = gt_mask_path(dataset_dir, img_path)
        else:
            mask_path = mv_mask_path(mv_dir, site, base_id)
            mask_path = mask_path if mask_path.exists() else None
        if mask_path is None:
            print(f"      skip {site}/{base_id}: no mask")
            continue

        img, mask = load_volume_pair(img_path, mask_path)
        if bilateral_axis is None:
            yield base_id, img, mask
            continue
        axis = yaml_axis_to_numpy_hwd(bilateral_axis)
        for side in ("L", "R"):
            h_img, h_mask = split_array_half(img, axis, side), split_array_half(mask, axis, side)
            if side == "R" and flip_right:
                h_img, h_mask = flip(h_img, axis), flip(h_mask, axis)
            yield f"{base_id}_{side}", h_img, h_mask
