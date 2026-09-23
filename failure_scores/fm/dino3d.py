"""
3DINO-ViT embeddings of a 3D case (native 3D ViT, 112³ input).

The volume is clipped to its 0.05/99.95 percentiles and scaled to [-1, 1]
(as in the 3DINO usage notebook), then trilinearly resized to 112³.

    img  CLS token of the volume
    ov   CLS token of the volume with mask voxels brightened by
         OVERLAY_STRENGTH (clipped to 1), the 3D analogue of the 2D overlay
    w    patch tokens of the volume, pooled with mask-coverage weights
    ovw  patch tokens of the overlay volume, pooled with mask-coverage weights

3DINO needs its own code (a clone whose path is in the environment variable
THREEDINO_CODE_DIR) and numpy<2, so it runs in a separate environment (README);
its embeddings are cached in ``fm_cache/`` like the others.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .pooling import patch_weights_3d, pool_patches_3d

THREEDINO_CKPT = "3dino_vit_weights.pth"
THREEDINO_CFG = "train/vit3d_highres"
VOL_SIZE = (112, 112, 112)
OVERLAY_STRENGTH = 0.5


def load_model(code_dir: Optional[str] = None):
    """(model, device) from a local 3DINO clone (default: $THREEDINO_CODE_DIR), in eval mode."""
    import sys
    import torch

    code_dir = code_dir or os.environ.get("THREEDINO_CODE_DIR")
    if not code_dir or not Path(code_dir).exists():
        raise RuntimeError(f"3DINO code not found (THREEDINO_CODE_DIR={code_dir!r}); clone "
                           "https://github.com/AICONSlab/3DINO and set THREEDINO_CODE_DIR "
                           "to its path (see the README)")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.eval.setup import build_model_for_eval

    model = build_model_for_eval(load_and_merge_config_3d(THREEDINO_CFG),
                                 str(Path(code_dir) / THREEDINO_CKPT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def normalize(vol: np.ndarray) -> np.ndarray:
    """Clip to the 0.05/99.95 percentiles and scale to [-1, 1] (float32)."""
    lo, hi = float(np.percentile(vol, 0.05)), float(np.percentile(vol, 99.95))
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    v = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (v * 2.0 - 1.0).astype(np.float32)


def overlay(vol_norm: np.ndarray, mask: np.ndarray,
            strength: float = OVERLAY_STRENGTH) -> np.ndarray:
    if not mask.any():
        return vol_norm
    out = vol_norm.copy()
    out[mask] = np.clip(vol_norm[mask] + strength, -1.0, 1.0)
    return out


def to_tensor(vol_norm: np.ndarray, device=None, size: Tuple[int, int, int] = VOL_SIZE):
    """(1, 1, *size) tensor of an already normalised volume."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.clip(vol_norm, -1.0, 1.0).astype(np.float32)).float()[None, None]
    if t.shape[2:] != size:
        t = F.interpolate(t, size=size, mode="trilinear", align_corners=False)
    return t.to(device) if device is not None else t


def cls_token(model, t) -> np.ndarray:
    out = model(t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 3:
        out = out[:, 0, :]
    return np.array(out[0].cpu().tolist(), dtype=np.float32)


def patch_tokens(model, t) -> np.ndarray:
    """(N, C) normalised patch tokens, N = n_side³ in C order."""
    return model.forward_features(t)["x_norm_patchtokens"][0].cpu().numpy()


def embed_case(img_hwd: np.ndarray, mask_hwd: np.ndarray, model, device) -> Dict[str, np.ndarray]:
    import torch

    vol = normalize(img_hwd)
    mask = mask_hwd > 0
    vol_ov = overlay(vol, mask)

    def pooled(v):
        tokens = patch_tokens(model, to_tensor(v, device))
        n_side = round(tokens.shape[0] ** (1 / 3))
        return pool_patches_3d(tokens, patch_weights_3d(mask, n_side))

    with torch.no_grad():
        return {
            "img": cls_token(model, to_tensor(vol, device)),
            "ov":  cls_token(model, to_tensor(vol_ov, device)),
            "w":   pooled(vol),
            "ovw": pooled(vol_ov),
        }
