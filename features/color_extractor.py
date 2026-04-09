"""
Color signature extractor.
Returns a fixed 12-dim vector: [L1,A1,B1,w1, L2,A2,B2,w2, L3,A3,B3,w3]
representing the top-3 dominant color clusters (GMM on superpixels).
"""
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from sklearn.mixture import GaussianMixture
from loguru import logger

from config.settings import (
    N_SEGMENTS, COMPACTNESS, SIGMA,
    MIN_AREA_RATIO, N_COLOR_CLUSTERS,
)

# ─── Internal helpers ─────────────────────────────────────────

def _to_rgb_np(image: Union[Image.Image, Path, str], target_size: int = 512) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        raise TypeError(f"Unsupported type: {type(image)}")

    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _foreground_mask(rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray > threshold


def _superpixel_colors(rgb: np.ndarray):
    """Return list of (mean_LAB, area_ratio) per valid superpixel."""
    lab   = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    fg    = _foreground_mask(rgb)
    total = fg.sum()
    if total == 0:
        return []

    segs = slic(rgb, n_segments=N_SEGMENTS, compactness=COMPACTNESS,
                sigma=SIGMA, start_label=0)
    out  = []
    for sid in np.unique(segs):
        mask = (segs == sid) & fg
        area = mask.sum()
        if area == 0:
            continue
        ratio = area / total
        if ratio < MIN_AREA_RATIO:
            continue
        out.append((lab[mask].mean(axis=0), ratio))
    return out


def _build_color_vector(sp_data: list) -> Optional[np.ndarray]:
    """
    Fit GMM on superpixel LAB colours → fixed 12-dim vector.
    Returns None if data is insufficient.
    """
    if not sp_data:
        return None

    colors  = np.array([c for c, _ in sp_data])
    weights = np.array([w for _, w in sp_data])

    n_comp = min(N_COLOR_CLUSTERS, len(colors))
    gmm    = GaussianMixture(n_components=n_comp, random_state=42)
    try:
        gmm.fit(colors, sample_weight=weights)
    except TypeError:
        expanded = np.repeat(colors, (weights * 100).astype(int) + 1, axis=0)
        gmm.fit(expanded)

    labels  = gmm.predict(colors)
    centers = gmm.means_
    cw      = np.zeros(n_comp)
    for i, (_, w) in enumerate(sp_data):
        cw[labels[i]] += w
    cw /= cw.sum()

    # Sort by weight desc, always produce N_COLOR_CLUSTERS slots
    order   = np.argsort(cw)[::-1]
    vec     = []
    for i in range(N_COLOR_CLUSTERS):
        if i < n_comp:
            idx = order[i]
            vec.extend([*centers[idx].tolist(), float(cw[idx])])
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(vec, dtype=np.float32)


def _normalize_color_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize each LAB channel to [0,1] and keep weights as-is.
    LAB: L∈[0,100], A∈[-128,127], B∈[-128,127]
    """
    norm = vec.copy()
    for i in range(N_COLOR_CLUSTERS):
        base = i * 4
        norm[base + 0] = vec[base + 0] / 100.0          # L
        norm[base + 1] = (vec[base + 1] + 128) / 255.0  # A
        norm[base + 2] = (vec[base + 2] + 128) / 255.0  # B
        # weight unchanged (already 0-1)
    return norm


# ─── Public API ───────────────────────────────────────────────

def extract_color_vector(image: Union[Image.Image, Path, str]) -> Optional[List[float]]:
    """
    Main entry point.
    Returns a 12-dim normalized float list, or None on failure.
    """
    try:
        rgb    = _to_rgb_np(image)
        sp     = _superpixel_colors(rgb)
        vec    = _build_color_vector(sp)
        if vec is None:
            logger.warning("⚠️  Could not build color vector")
            return None
        norm   = _normalize_color_vector(vec)
        return norm.tolist()
    except Exception:
        logger.exception("❌ Color extraction failed")
        return None


def color_distance(v1: List[float], v2: List[float]) -> float:
    """
    Weighted LAB distance between two 12-dim color vectors.
    Considers the cluster weights when comparing clusters.
    Lower = more similar.
    """
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)

    total_dist = 0.0
    for i in range(N_COLOR_CLUSTERS):
        base = i * 4
        wa   = a[base + 3]
        wb   = b[base + 3]
        w    = (wa + wb) / 2.0
        lab_dist = np.linalg.norm(a[base:base+3] - b[base:base+3])
        total_dist += w * lab_dist
    return float(total_dist)