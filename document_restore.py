"""
Adaptive document restoration and enhancement using classical computer vision.

The pipeline estimates simple quality signals (blur, brightness, contrast) and
adjusts denoising, CLAHE, and thresholding so one script works across scans,
phone photos, and faded prints.

Usage:
  python document_restore.py input.png -o output.png
  python document_restore.py input.jpg -o out.png --no-deskew --debug
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class QualityMetrics:
    blur_var: float
    brightness: float  # 0..1 mean luminance
    contrast: float  # std of luminance / 255


@dataclass
class AdaptiveParams:
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    clahe_clip: float
    clahe_tile: int
    adaptive_block: int
    adaptive_c: int
    use_adaptive_thresh: bool
    gamma: float


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_image(path: Path, bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext if ext else ".png", bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode: {path}")
    buf.tofile(str(path))


def compute_metrics(gray: np.ndarray) -> QualityMetrics:
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    g = gray.astype(np.float64) / 255.0
    return QualityMetrics(
        blur_var=blur_var,
        brightness=float(np.mean(g)),
        contrast=float(np.std(g)),
    )


def choose_params(m: QualityMetrics) -> AdaptiveParams:
    """
    Map quality metrics to OpenCV parameters. Thresholds are heuristics;
    tune on your own dataset for a report / demo.
    """
    # Blur: Laplacian variance often < 100 for blurry phone shots, > 300 for sharp scans
    if m.blur_var < 80:
        bd, bsc, bss = 9, 100, 100
    elif m.blur_var < 200:
        bd, bsc, bss = 7, 75, 75
    else:
        bd, bsc, bss = 5, 50, 50

    # CLAHE: weak contrast needs stronger local equalization
    if m.contrast < 0.08:
        clip, tile = 3.5, 8
    elif m.contrast < 0.15:
        clip, tile = 2.5, 8
    else:
        clip, tile = 2.0, 16

    # Very dark / bright: gamma nudge (applied in linear-light-ish way on gray)
    if m.brightness < 0.35:
        gamma = 0.85
    elif m.brightness > 0.72:
        gamma = 1.12
    else:
        gamma = 1.0

    # Adaptive threshold window scales mildly with resolution idea: use contrast
    block = 31 if m.contrast < 0.12 else 21
    c = 10 if m.brightness < 0.4 or m.brightness > 0.65 else 8

    # Uniform lighting + high contrast: Otsu can work; else prefer adaptive
    use_adaptive = m.contrast < 0.18 or m.blur_var < 150

    return AdaptiveParams(
        bilateral_d=bd,
        bilateral_sigma_color=float(bsc),
        bilateral_sigma_space=float(bss),
        clahe_clip=clip,
        clahe_tile=tile,
        adaptive_block=block | 1,
        adaptive_c=c,
        use_adaptive_thresh=use_adaptive,
        gamma=gamma,
    )


def reduce_shadows_bgr(bgr: np.ndarray) -> np.ndarray:
    """Homomorphic-style lighting normalization in LAB L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=30, sigmaY=30)
    corrected = np.clip((l_f / blur) * 128.0, 0, 255).astype(np.uint8)
    lab = cv2.merge([corrected, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma_bgr(bgr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return bgr
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


def deskew_grayscale(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Deskew using min-area rectangle of Otsu foreground (text-like regions)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.shape[0] < 100:
        return gray, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.25:
        return gray, 0.0
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)


def largest_quad_from_contours(
    gray: np.ndarray, min_area_ratio: float = 0.15
) -> np.ndarray | None:
    """Return 4x2 float32 corners of largest quadrilateral-like contour, or None."""
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area_img = float(h * w)
    best = None
    best_area = 0.0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri < 100:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        a = float(cv2.contourArea(approx))
        if a < min_area_ratio * area_img or a <= best_area:
            continue
        best_area = a
        best = approx.reshape(4, 2).astype(np.float32)
    return best


def warp_document(bgr: np.ndarray, padding: float = 0.02) -> np.ndarray:
    """
    Perspective unwrap if a large quadrilateral is found; otherwise return input.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = largest_quad_from_contours(gray)
    if quad is None:
        return bgr
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1)
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))
    if max_w < 100 or max_h < 100:
        return bgr
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr, m, (max_w, max_h))
    ph, pw = warped.shape[:2]
    px, py = int(pw * padding), int(ph * padding)
    if px > 0 and py > 0 and pw - 2 * px > 10 and ph - 2 * py > 10:
        warped = warped[py : ph - py, px : pw - px]
    return warped


def restore_pipeline(
    bgr: np.ndarray,
    *,
    shadow_removal: bool = True,
    deskew: bool = True,
    perspective: bool = False,
    debug: bool = False,
) -> Tuple[np.ndarray, dict]:
    meta: dict = {}
    work = bgr.copy()
    if perspective:
        work = warp_document(work)
        meta["perspective_warp"] = True
    if shadow_removal:
        work = reduce_shadows_bgr(work)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    metrics = compute_metrics(gray)
    params = choose_params(metrics)
    meta["metrics"] = metrics
    meta["params"] = params

    work = apply_gamma_bgr(work, params.gamma)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    if deskew:
        gray, angle = deskew_grayscale(gray)
        meta["deskew_deg"] = angle
        if abs(angle) > 0.25:
            h, w = work.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            work = cv2.warpAffine(
                work,
                m,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(
        gray,
        d=params.bilateral_d,
        sigmaColor=params.bilateral_sigma_color,
        sigmaSpace=params.bilateral_sigma_space,
    )

    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip, tileGridSize=(params.clahe_tile, params.clahe_tile)
    )
    enhanced = clahe.apply(denoised)

    if params.use_adaptive_thresh:
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            params.adaptive_block,
            params.adaptive_c,
        )
    else:
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Optional cleanup: remove pepper noise on white background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    meta["binary"] = binary

    if debug:
        meta["enhanced_gray"] = enhanced
        meta["denoised_gray"] = denoised

    # Output: 3-channel for easy viewing/saving alongside binary
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary_bgr, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Input image path")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output image path")
    p.add_argument("--no-shadow", action="store_true", help="Skip shadow/light normalization")
    p.add_argument("--no-deskew", action="store_true", help="Skip rotation deskew")
    p.add_argument(
        "--perspective",
        action="store_true",
        help="Try perspective unwrap (document boundary detection)",
    )
    p.add_argument(
        "--also-binary-gray",
        type=Path,
        default=None,
        help="Also save single-channel enhanced grayscale before threshold",
    )
    p.add_argument("--debug", action="store_true", help="Print metrics and params")
    args = p.parse_args(argv)

    img = read_image(args.input)
    out, meta = restore_pipeline(
        img,
        shadow_removal=not args.no_shadow,
        deskew=not args.no_deskew,
        perspective=args.perspective,
        debug=args.debug or args.also_binary_gray is not None,
    )
    save_image(args.output, out)

    if args.also_binary_gray is not None and "enhanced_gray" in meta:
        eg = meta["enhanced_gray"]
        save_image(args.also_binary_gray, cv2.cvtColor(eg, cv2.COLOR_GRAY2BGR))

    if args.debug:
        m: QualityMetrics = meta["metrics"]
        par: AdaptiveParams = meta["params"]
        print(f"blur_var={m.blur_var:.2f} brightness={m.brightness:.3f} contrast={m.contrast:.3f}")
        print(
            f"bilateral_d={par.bilateral_d} clahe_clip={par.clahe_clip} "
            f"adaptive_block={par.adaptive_block} gamma={par.gamma} "
            f"adaptive_thresh={par.use_adaptive_thresh}"
        )
        if "deskew_deg" in meta:
            print(f"deskew_deg={meta['deskew_deg']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
