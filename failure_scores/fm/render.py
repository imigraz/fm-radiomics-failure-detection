"""
2D slice -> RGB PIL image for the 2D ViTs.

Intensities are windowed to the slice's 1st/99th percentiles and mapped to
uint8. Non-square slices are zero-padded to a square (image in the top-left
corner) so that the processor's resize keeps the aspect ratio.

The plain image uses a window passed in by the caller (the full-slice
percentiles); the overlay computes the same percentiles itself.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

OVERLAY_ALPHA = 0.4
OVERLAY_RGB = (255, 80, 0)


def slice_window(img_sl: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(img_sl, 1)), float(np.percentile(img_sl, 99))


def _to_uint8(img_sl: np.ndarray, lo, hi) -> np.ndarray:
    if hi == lo:
        return np.zeros(img_sl.shape, dtype=np.uint8)
    return np.clip((img_sl - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _pad_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = max(h, w)
    square = np.zeros((side, side) + arr.shape[2:], dtype=np.uint8)
    square[:h, :w] = arr
    return square


def gray_rgb(img_sl: np.ndarray, window: Optional[Tuple[float, float]] = None) -> Image.Image:
    """Grayscale slice as RGB (mode ``img``, ``w``)."""
    lo, hi = window if window is not None else (np.percentile(img_sl, 1),
                                                np.percentile(img_sl, 99))
    gray = _pad_square(_to_uint8(img_sl, lo, hi))
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1))


def overlay_rgb(img_sl: np.ndarray, mask_sl: np.ndarray,
                alpha: float = OVERLAY_ALPHA, color=OVERLAY_RGB) -> Image.Image:
    """Slice with the mask alpha-blended in ``color`` (mode ``ov``, ``ovw``)."""
    gray = _to_uint8(img_sl, np.percentile(img_sl, 1), np.percentile(img_sl, 99))
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    inside = mask_sl.astype(bool)
    for c, value in enumerate(color):
        rgb[:, :, c] = np.where(inside, (1 - alpha) * rgb[:, :, c] + alpha * value,
                                rgb[:, :, c])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(_pad_square(rgb))
