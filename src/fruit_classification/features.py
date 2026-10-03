"""Feature extraction for the Fruits-360 image classification task.

The feature pipeline is deliberately dependency-light (numpy + Pillow only) so
that CI and a fresh clone can reproduce it without extra compiled packages.

Design notes
------------
* ``hog_features`` is a vectorised histogram-of-oriented-gradients descriptor.
  It captures fruit shape/texture, which is the dominant signal for this task.
* ``color_histogram`` captures peel colour, which separates otherwise similar
  classes (e.g. `Apple 1` vs `Apple 5`).
* ``extract_features`` concatenates the two and is the single reusable function
  used by the dataset builder, the training pipeline and the tests.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

EPS = 1e-6


def to_rgb_array(img: Image.Image, size: int) -> np.ndarray:
    """Return an ``(size, size, 3)`` float32 RGB array in [0, 1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma plane; float32 in [0, 1]."""
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _rgb_to_hsv_planes(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H, S, V planes of HSV, each float32 in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > EPS
    # Guard against division by zero: only compute where delta is non-zero.
    safe = np.where(mask, delta, 1.0)
    idx = mask & (maxc == r)
    hue[idx] = ((g[idx] - b[idx]) / safe[idx]) % 6.0
    idx = mask & (maxc == g)
    hue[idx] = (b[idx] - r[idx]) / safe[idx] + 2.0
    idx = mask & (maxc == b)
    hue[idx] = (r[idx] - g[idx]) / safe[idx] + 4.0
    hue = (hue / 6.0 + 1.0) % 1.0

    sat = np.where(maxc > EPS, delta / (maxc + EPS), 0.0)
    return (
        hue.astype(np.float32),
        sat.astype(np.float32),
        maxc.astype(np.float32),
    )


def hog_features(gray: np.ndarray, cells: int = 4, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients for one grayscale image.

    Parameters
    ----------
    gray : (H, W) float32 array in [0, 1].
    cells : number of spatial cells per axis.
    bins  : number of unsigned orientation bins.

    Returns
    -------
    (cells * cells * bins,) float32 vector, L2-normalised per block.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {gray.shape}")

    height, width = gray.shape
    if height % cells or width % cells:
        raise ValueError(f"image shape {gray.shape} is not divisible by {cells} cells")

    # Central differences keep the border well defined.
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)  # (-pi, pi]

    # Soft binning: linear assignment between the two neighbouring bins, which
    # is the standard unsigned-HOG formulation without the sign offset.
    angle_norm = ((angle + np.pi) / (2.0 * np.pi)) * bins
    angle_norm = np.clip(angle_norm, 0.0, bins - EPS)
    lower = np.floor(angle_norm).astype(np.int32)
    weight = (angle_norm - lower).astype(np.float32)

    h_cell = height // cells
    w_cell = width // cells
    block = magnitude.reshape(cells, h_cell, cells, w_cell)
    lo = lower.reshape(cells, h_cell, cells, w_cell)
    wt = weight.reshape(cells, h_cell, cells, w_cell)

    # Sum over each cell's pixels for every orientation bin. Two linear
    # assignments per pixel (offset 0/1) approximate the hard binning of the
    # classic Dalal-Triggs HOG and avoid orientation discontinuities.
    out = np.zeros((cells, cells, bins), dtype=np.float32)
    for offset in (0, 1):
        weight_slice = wt if offset == 0 else (1.0 - wt)
        bin_idx = lo + offset
        for c in range(bins):
            out[..., c] += np.where(bin_idx == c, block * weight_slice, 0.0).sum(axis=(1, 3))

    norm = np.linalg.norm(out, axis=-1, keepdims=True) + EPS
    return (out / norm).ravel().astype(np.float32)


def color_histogram(rgb: np.ndarray, hue_bins: int = 8, sat_bins: int = 4, val_bins: int = 4) -> np.ndarray:
    """Joint HSV histogram, L1-normalised, flattened.

    Returns
    -------
    (hue_bins * sat_bins * val_bins,) float32 vector.
    """
    hue, sat, val = _rgb_to_hsv_planes(rgb)
    hi = np.clip((hue * hue_bins).astype(np.int32), 0, hue_bins - 1)
    si = np.clip((sat * sat_bins).astype(np.int32), 0, sat_bins - 1)
    vi = np.clip((val * val_bins).astype(np.int32), 0, val_bins - 1)
    flat = (hi * sat_bins + si) * val_bins + vi
    hist = np.bincount(flat.ravel(), minlength=hue_bins * sat_bins * val_bins)
    total = hist.sum() + EPS
    return (hist / total).astype(np.float32)


def feature_dim(image_size: int = 32, cells: int = 4, bins: int = 9, use_color: bool = True) -> int:
    """Feature vector length produced by :func:`extract_features`."""
    dim = cells * cells * bins
    if use_color:
        dim += 8 * 4 * 4
    return dim


def extract_features(
    img: Image.Image,
    image_size: int = 32,
    cells: int = 4,
    bins: int = 9,
    use_color: bool = True,
) -> np.ndarray:
    """Extract the classification feature vector from one PIL image."""
    rgb = to_rgb_array(img, image_size)
    parts = [hog_features(_luminance(rgb), cells=cells, bins=bins)]
    if use_color:
        parts.append(color_histogram(rgb))
    return np.concatenate(parts).astype(np.float32)