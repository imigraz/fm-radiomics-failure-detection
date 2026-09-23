"""
Token pooling: the CLS token, or a mask-weighted mean of the patch tokens.

2D: the slice mask is area-resized (cv2.INTER_AREA) to the patch grid, giving
each patch the fraction of its area inside the mask. Patches below
PATCH_COV_THR get weight 0; the rest are weighted by coverage.

3D: the mask, in the axis order of the volume given to the 3D ViT, is
trilinearly resized to the cubic patch grid and flattened in C order, the
order in which the ViT flattens its patches. All patches are weighted by
coverage (no threshold).

If no patch has weight, the plain mean of all patch tokens is used.
"""

from __future__ import annotations

import numpy as np

PATCH_COV_THR = 0.10


def patch_weights_2d(mask_hw: np.ndarray, n_side: int) -> np.ndarray:
    """Per-patch mask coverage, shape (n_side**2,), row-major."""
    import cv2
    resized = cv2.resize(mask_hw.astype(np.float32), (n_side, n_side),
                         interpolation=cv2.INTER_AREA)
    return resized.flatten()


def pool_patches_2d(patch_tokens: np.ndarray, coverage: np.ndarray,
                    thr: float = PATCH_COV_THR) -> np.ndarray:
    w = np.where(coverage >= thr, coverage, 0.0)
    if w.sum() > 1e-8:
        return (patch_tokens * (w / w.sum())[:, None]).sum(0)
    return patch_tokens.mean(0)


def patch_weights_3d(mask_3d: np.ndarray, n_side: int) -> np.ndarray:
    """Per-patch mask coverage on the cubic grid, shape (n_side**3,), C order.

    ``mask_3d`` has the axis order of the volume the ViT embeds.
    """
    import torch
    import torch.nn.functional as F
    m = torch.from_numpy(mask_3d.astype(np.float32))[None, None]
    resized = F.interpolate(m, size=(n_side, n_side, n_side), mode="trilinear",
                            align_corners=False)
    return resized.squeeze().numpy().flatten()


def pool_patches_3d(patch_tokens: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    s = float(coverage.sum())
    if s < 1e-8:
        return patch_tokens.mean(0)
    return (patch_tokens * (coverage / s)[:, None]).sum(0).astype(np.float32)
