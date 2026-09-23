"""
2D ViT embeddings (DINOv3 ViT-B/16, RadDINO ViT-B/14) of a 3D case.

Every axial slice whose mask covers at least MIN_MASK_AREA pixels is embedded
with two forward passes: the plain slice (-> img, w) and the overlay slice
(-> ov, ovw). The case embedding of each mode is the mask-area-weighted mean
over those slices. If no slice reaches MIN_MASK_AREA (a very small predicted
mask), every slice with any mask pixel is used instead, so that each case with
a non-empty mask gets an embedding.

Token layout of the HF models: [CLS, registers..., patches...]. DINOv3 has 4
register tokens, RadDINO none; they are skipped so that the patch tokens line
up with the patch grid.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .pooling import patch_weights_2d, pool_patches_2d
from .render import gray_rgb, overlay_rgb, slice_window

MODEL_IDS = {
    "dinov3":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "raddino": "microsoft/rad-dino",
}
MIN_MASK_AREA = 10   # min mask pixels for a slice to be embedded


def load_model(backbone: str):
    """(processor, model, device) for a 2D backbone, in eval mode."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    model_id = MODEL_IDS[backbone]
    model = AutoModel.from_pretrained(model_id)
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        # Checkpoint without a preprocessor config: ImageNet-normalised
        # square resize to the model's image size.
        from transformers import ViTImageProcessor
        size = getattr(model.config, "image_size", 224)
        processor = ViTImageProcessor(size={"height": size, "width": size},
                                      image_mean=[0.485, 0.456, 0.406],
                                      image_std=[0.229, 0.224, 0.225])
        print(f"  {model_id}: no preprocessor config, using ViTImageProcessor {size}x{size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model.to(device).eval(), device


def split_tokens(tokens: np.ndarray, n_reg: int):
    """(CLS token, patch tokens) from [CLS, n_reg registers, patches]."""
    return tokens[0], tokens[1 + n_reg:]


def _forward(image, processor, model, device) -> np.ndarray:
    import torch
    inp = {k: v.to(device) for k, v in processor(images=image, return_tensors="pt").items()}
    with torch.no_grad():
        return model(**inp).last_hidden_state[0].cpu().numpy()


def embed_slice(img_sl: np.ndarray, mask_sl: np.ndarray,
                processor, model, device) -> Dict[str, np.ndarray]:
    """The four mode embeddings of one slice."""
    n_reg = getattr(model.config, "num_register_tokens", 0)

    cls, patches = split_tokens(
        _forward(gray_rgb(img_sl, slice_window(img_sl)), processor, model, device), n_reg)
    n_side = int(round(patches.shape[0] ** 0.5))
    coverage = patch_weights_2d(mask_sl, n_side)
    out = {"img": cls, "w": pool_patches_2d(patches, coverage)}

    cls_ov, patches_ov = split_tokens(
        _forward(overlay_rgb(img_sl, mask_sl), processor, model, device), n_reg)
    n_side_ov = int(round(patches_ov.shape[0] ** 0.5))
    if n_side_ov != n_side:
        coverage = patch_weights_2d(mask_sl, n_side_ov)
    out["ov"] = cls_ov
    out["ovw"] = pool_patches_2d(patches_ov, coverage)
    return out


def embed_case(img_hwd: np.ndarray, mask_hwd: np.ndarray,
               processor, model, device) -> Optional[Dict[str, np.ndarray]]:
    """Area-weighted mean of the slice embeddings; None if the mask is empty."""
    slice_areas = [float(mask_hwd[:, :, d].sum()) for d in range(img_hwd.shape[2])]
    min_area = MIN_MASK_AREA if max(slice_areas, default=0.0) >= MIN_MASK_AREA else 1.0
    per_mode = {}
    areas = []
    for d, area in enumerate(slice_areas):
        if area < min_area:
            continue
        for mode, emb in embed_slice(img_hwd[:, :, d], mask_hwd[:, :, d],
                                     processor, model, device).items():
            per_mode.setdefault(mode, []).append(emb)
        areas.append(area)
    if not areas:
        return None
    w = np.array(areas)
    w = w / w.sum()
    return {mode: (np.stack(embs) * w[:, None]).sum(0) for mode, embs in per_mode.items()}
