import cv2
import numpy as np

# Only very small isolated black blobs are removed (true pepper); keep conservative for scripts like Tamil.
_SPECKLE_MAX_AREA_CAP = 28


def deskew(image):
    """Estimate skew from mostly horizontal edges (ignore vertical structure)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 180)

    h, w = gray.shape[:2]
    vote = max(min(h, w) // 5, 120)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, vote)

    angles = []
    if lines is not None and len(lines) > 0:
        for rho, theta in lines[:, 0]:
            if abs(theta - np.pi / 2) > np.pi / 4:
                continue
            a = (theta - np.pi / 2) * (180.0 / np.pi)
            if abs(a) <= 35:
                angles.append(a)

    angle = float(np.median(angles)) if angles else 0.0

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _illumination_flatten(gray):
    """Reduce uneven lighting before contrast / threshold (soft, stable)."""
    h, w = gray.shape[:2]
    k = int(round(min(h, w) / 25.0))
    k = max(k | 1, 31)
    k = min(k, 151)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    blur = np.maximum(blur.astype(np.float32), 1.0)
    flat = np.clip(255.0 * gray.astype(np.float32) / blur, 0, 255).astype(np.uint8)
    return flat


def _denoise_light(gray):
    return cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)


def _denoise_strong(gray):
    return cv2.fastNlMeansDenoising(gray, None, h=14, templateWindowSize=7, searchWindowSize=21)


def _unsharp(gray, sigma=1.0, amount=1.35):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    out = cv2.addWeighted(gray, amount, blur, 1.0 - amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _gray_for_threshold(gray):
    """
    Light blur only on a copy used for thresholding.
    (Does not change the pre_threshold image shown in the UI.)
    """
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _binarize_adaptive(gray):
    g = _gray_for_threshold(gray)
    m = min(gray.shape[:2])
    block = int(m // 14)
    block = max(block | 1, 15)
    block = min(block, 41)
    return cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        7,
    )


def _binarize_otsu(gray):
    g = _gray_for_threshold(gray)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def _normalize_binary_for_display(binary):
    if np.mean(binary) < 127:
        return cv2.bitwise_not(binary)
    return binary


def _remove_black_pepper(binary, max_area=None):
    h, w = binary.shape[:2]
    if max_area is None:
        px = h * w
        max_area = max(8, min(_SPECKLE_MAX_AREA_CAP, int(px / 180000)))

    inv = cv2.bitwise_not(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = binary.copy()
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
            out[labels == i] = 255
    return out


def _cleanup_binary(binary):
    """Minimal post-process: only drop tiny isolated specks (no median/thinning/heavy close)."""
    return _remove_black_pepper(binary)


def process_pipeline(image, decision):
    steps = {}
    decision = str(decision)

    steps["original"] = image.copy()

    image = deskew(image)
    steps["deskew"] = image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray

    gray = _illumination_flatten(gray)
    steps["flatten"] = gray

    if "strong_denoise" in decision:
        gray = _denoise_strong(gray)
        gray = _unsharp(gray, sigma=0.9, amount=1.25)
        steps["strong_denoise"] = gray
    elif "denoise" in decision:
        gray = _denoise_light(gray)
        steps["denoise"] = gray

    if "clahe" in decision:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(16, 16))
        gray = clahe.apply(gray)
        steps["clahe"] = gray

    if "adaptive" in decision:
        steps["pre_threshold"] = gray.copy()
        thresh = _binarize_adaptive(gray)
        thresh = _normalize_binary_for_display(thresh)
        thresh = _cleanup_binary(thresh)
        steps["threshold"] = thresh
    elif "otsu" in decision:
        steps["pre_threshold"] = gray.copy()
        thresh = _binarize_otsu(gray)
        thresh = _normalize_binary_for_display(thresh)
        thresh = _cleanup_binary(thresh)
        steps["threshold"] = thresh
    else:
        out = _unsharp(gray, sigma=0.8, amount=1.15)
        steps["threshold"] = out

    steps["final"] = steps["threshold"]

    return steps
