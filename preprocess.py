import cv2
import numpy as np
import re

PAN_REGEX = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')



def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_crop_card(image: np.ndarray) -> np.ndarray:
    """
    Detect the card's rectangular boundary and perspective-warp it
    to a flat, upright rectangle. Falls back to the original image
    if no confident card-shaped contour is found.
    """
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    card_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > (small.shape[0] * small.shape[1] * 0.15):
            card_contour = approx
            break

    if card_contour is None:
        return orig  # couldn't confidently find the card — use original

    pts = card_contour.reshape(4, 2) * ratio
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 100 or maxHeight < 100:
        return orig  # degenerate result — bail out to original

    aspect_ratio = float(maxWidth) / float(maxHeight)
    if aspect_ratio < 1.1 or aspect_ratio > 2.3:
        return orig  # abnormal aspect ratio — bail out to original uncropped image

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # PAN cards are landscape — if warp came out portrait, rotate 90°
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize, auto-crop/flatten the card, and optimize contrast for OCR."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Make sure the file is a valid image.")

    # Downscale huge phone photos (speeds up OCR, no real accuracy loss)
    h, w = img.shape[:2]
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Try to auto-detect and flatten the card — fixes rotation + skew + crop
    cropped = auto_crop_card(img)

    # Mild contrast enhancement (CLAHE) on L channel for sharper OCR text without heavy blur
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced


def find_best_rotation(image: np.ndarray, model) -> tuple[np.ndarray, any, str]:
    """
    Try 0/90/180/270 rotations and return (best_rotated_image, ocr_result, debug_info).
    Uses early exit if PAN pattern is found to minimize CPU overhead.
    """
    # 0 degrees (most common)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res_0 = model([rgb])
    text_0 = ""
    for block in res_0.pages[0].blocks:
        for line in block.lines:
            text_0 += " ".join(w.value for w in line.words) + "\n"
    if PAN_REGEX.search(text_0.replace(" ", "")):
        return image, res_0, "chosen_angle=0 (early exit)"

    # 180 degrees
    img_180 = cv2.rotate(image, cv2.ROTATE_180)
    rgb_180 = cv2.cvtColor(img_180, cv2.COLOR_BGR2RGB)
    res_180 = model([rgb_180])
    text_180 = ""
    for block in res_180.pages[0].blocks:
        for line in block.lines:
            text_180 += " ".join(w.value for w in line.words) + "\n"
    if PAN_REGEX.search(text_180.replace(" ", "")):
        return img_180, res_180, "chosen_angle=180 (early exit)"

    # Helper to calculate confidence score
    def get_score(result):
        word_count = 0
        conf_sum = 0.0
        for block in result.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    word_count += 1
                    conf_sum += word.confidence
        avg_conf = (conf_sum / word_count) if word_count else 0.0
        return word_count * (0.3 + avg_conf)

    score_0 = get_score(res_0)
    score_180 = get_score(res_180)

    best_img = image if score_0 >= score_180 else img_180
    best_res = res_0 if score_0 >= score_180 else res_180
    best_score = max(score_0, score_180)
    best_angle = 0 if score_0 >= score_180 else 180

    # 90 degrees clockwise
    img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rgb_90 = cv2.cvtColor(img_90, cv2.COLOR_BGR2RGB)
    res_90 = model([rgb_90])
    score_90 = get_score(res_90)
    if score_90 > best_score:
        best_score = score_90
        best_img = img_90
        best_res = res_90
        best_angle = 90

    # 90 degrees counterclockwise (270 degrees)
    img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rgb_270 = cv2.cvtColor(img_270, cv2.COLOR_BGR2RGB)
    res_270 = model([rgb_270])
    score_270 = get_score(res_270)
    if score_270 > best_score:
        best_score = score_270
        best_img = img_270
        best_res = res_270
        best_angle = 270

    return best_img, best_res, f"chosen_angle={best_angle} (score fallback)"
