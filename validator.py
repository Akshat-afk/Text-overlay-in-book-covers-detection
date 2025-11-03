"""
cover_pipeline_prototype.py
Updated prototype pipeline using EasyOCR for text detection and OCR.

Requirements:
- Python 3.8+
- Install packages: pip install opencv-python pillow easyocr numpy imutils

Usage examples:
python cover_pipeline_prototype.py --input-dir ./covers --output-dir ./out
"""

import os
import json
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np
from math import floor
import easyocr

# ---------------------- Utilities ----------------------
reader = easyocr.Reader(['en'], gpu=False)

def mm_to_inches(mm):
    return mm / 25.4


def inches_to_pixels(inches, dpi):
    return int(round(inches * dpi))


def mm_to_pixels(mm, dpi):
    return inches_to_pixels(mm_to_inches(mm), dpi)


def read_image(path):
    img = Image.open(path).convert('RGB')
    return img


def get_image_dpi(img: Image.Image):
    try:
        info = img.info
        if 'dpi' in info and isinstance(info['dpi'], tuple):
            return int(info['dpi'][0])
    except Exception:
        pass
    return None


def normalize_to_dpi(img: Image.Image, current_dpi: int, target_dpi: int):
    if current_dpi is None:
        return img, target_dpi
    if current_dpi == target_dpi:
        return img, current_dpi
    scale = target_dpi / float(current_dpi)
    new_w = int(round(img.width * scale))
    new_h = int(round(img.height * scale))
    resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
    return resized, target_dpi

# ---------------------- Badge zone ----------------------

def compute_badge_zone(img_w_px, img_h_px, dpi, badge_height_mm=9):
    badge_h_px = mm_to_pixels(badge_height_mm, dpi)
    x1 = img_w_px // 2
    y1 = img_h_px - badge_h_px
    x2 = img_w_px
    y2 = img_h_px
    return (x1, y1, x2, y2)

# ---------------------- Image quality ----------------------

def variance_of_laplacian(img_cv_gray):
    return cv2.Laplacian(img_cv_gray, cv2.CV_64F).var()


def check_blur_threshold(pil_img: Image.Image, threshold=100.0):
    cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    var = variance_of_laplacian(cv)
    return float(var), bool(var >= threshold)

# ---------------------- EasyOCR text detection ----------------------

def detect_text_easyocr(pil_img: Image.Image, reader):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    results = reader.readtext(img_cv)
    lines = []
    for (bbox, text, conf) in results:
        pts = np.array(bbox).astype(int)
        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
        lines.append({'text': text.strip(), 'conf': float(conf), 'bbox': (int(x1), int(y1), int(x2), int(y2))})
    return lines

# ---------------------- Overlap math ----------------------

def rect_intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def rect_area(r):
    return max(0, (r[2] - r[0])) * max(0, (r[3] - r[1]))

# ---------------------- Overlay & report ----------------------

def draw_overlay_and_save(pil_img: Image.Image, badge_rect, text_lines, out_path):
    img_draw = pil_img.convert('RGBA')
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(badge_rect, outline='red', width=3)
    for ln in text_lines:
        draw.rectangle(ln['bbox'], outline='blue', width=2)
    img_draw.convert('RGB').save(out_path)

# ---------------------- Pipeline per image ----------------------

def process_image(path, reader = reader, target_dpi=300, overlap_threshold=0.01, blur_threshold=100.0):
    img_pil = read_image(path)
    orig_w, orig_h = img_pil.width, img_pil.height
    current_dpi = get_image_dpi(img_pil)
    img_norm, dpi_used = normalize_to_dpi(img_pil, current_dpi, target_dpi)

    w, h = img_norm.width, img_norm.height
    badge = compute_badge_zone(w, h, dpi_used)
    
    left_margin_px = mm_to_pixels(3, dpi_used)
    right_margin_px = mm_to_pixels(3, dpi_used)
    middle_margin_px = mm_to_pixels(6, dpi_used)

    left_margin = (0, 0, left_margin_px, h)
    right_margin = (w - right_margin_px, 0, w, h)
    middle_margin = ((w // 2) - (middle_margin_px // 2), 0, (w // 2) + (middle_margin_px // 2), h)
    safe_margins = {'left': left_margin, 'right': right_margin, 'middle': middle_margin}

    blur_var, blur_ok = check_blur_threshold(img_norm, threshold=blur_threshold)

    text_lines = detect_text_easyocr(img_norm, reader)

    conf_values = [ln["conf"] for ln in text_lines if "conf" in ln]
    confidence_score = round(sum(conf_values) / len(conf_values), 2)

    allowed_words = set("winner of the 21st century emily dickinson award".split())
    unauthorized_texts = []
    text_in_safe_margin = []

    for ln in text_lines:
        bbox = ln['bbox']
        a = rect_area(bbox)
        if a <= 0:
            continue

        # Check award zone overlap ratio
        ratio_award = rect_intersection_area(bbox, badge) / a
        # Check safe margins
        in_safe_margin = (
            rect_intersection_area(bbox, left_margin) > 0 or
            rect_intersection_area(bbox, right_margin) > 0 or
            rect_intersection_area(bbox, middle_margin) > 0
        )

        text_words = set(ln['text'].lower().split())

        # Flag unauthorized text in award zone
        if ratio_award >= overlap_threshold and not text_words.issubset(allowed_words):
            unauthorized_texts.append(ln['text'])

        # Flag text inside safe margins
        if in_safe_margin:
            text_in_safe_margin.append(ln['text'])

    cover_valid = len(unauthorized_texts) == 0 and len(text_in_safe_margin) == 0
    validation_message = "Cover is valid." if cover_valid else "Cover invalid due to unauthorized text in award zone or safe margins."

    overlay_path = None
    try:
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]
        overlay_path = name + '_overlay.jpg'
        draw_overlay_and_save(img_norm, badge, text_lines, overlay_path)
    except Exception:
        overlay_path = None

    results = []
    for ln in text_lines:
        bbox = ln['bbox']
        a = rect_area(bbox)
        inter = rect_intersection_area(bbox, badge)
        ratio = (inter / a) if a > 0 else 0.0
        flagged = (ln['text'] in unauthorized_texts) or (ln['text'] in text_in_safe_margin)
        results.append({'text': ln['text'], 'conf': ln['conf'], 'bbox': bbox, 'overlap_ratio': ratio, 'flagged': flagged})

    report = {
        'file': path,
        'orig_size': (orig_w, orig_h),
        'dpi_inferred': current_dpi,
        'dpi_used': dpi_used,
        'badge_bbox': badge,
        'blur_variance': blur_var,
        'blur_ok': blur_ok,
        'text_lines': results,
        'cover_valid': cover_valid,
        'unauthorized_text_in_award_zone': unauthorized_texts,
        'text_in_safe_margin': text_in_safe_margin,
        'validation_message': validation_message,
        'overlay_path': overlay_path,
        'confidence_score': (round((1-overlap_threshold),2)*100)
    }
    return report

# ---------------------- CLI / Bulk runner ----------------------

def process_folder(input_dir, output_dir, target_dpi=300, overlap_threshold=0.01, blur_threshold=100.0):
    os.makedirs(output_dir, exist_ok=True)
    reports = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        path = os.path.join(input_dir, fname)
        rpt = process_image(path, reader, target_dpi=target_dpi, overlap_threshold=overlap_threshold, blur_threshold=blur_threshold)
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]
        if rpt['overlay_path']:
            try:
                os.replace(rpt['overlay_path'], os.path.join(output_dir, os.path.basename(rpt['overlay_path'])))
                rpt['overlay_path'] = os.path.join(output_dir, os.path.basename(rpt['overlay_path']))
            except Exception:
                rpt['overlay_path'] = None
        out_json = os.path.join(output_dir, name + '.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(rpt, f, indent=2)
        reports.append(rpt)
    with open(os.path.join(output_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump({'reports': [r['file'] for r in reports]}, f, indent=2)
    return reports


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default="/Users/akki/Desktop/AKKI/Presonal Projects/Text overlay in book covers detection/test images")
    p.add_argument('--output-dir', default="/Users/akki/Desktop/AKKI/Presonal Projects/Text overlay in book covers detection/output(front fix)frfrfr")
    p.add_argument('--target-dpi', type=int, default=300)
    p.add_argument('--overlap-threshold', type=float, default=0.01)
    p.add_argument('--blur-threshold', type=float, default=100.0)
    args = p.parse_args()
    process_folder(args.input_dir, args.output_dir, target_dpi=args.target_dpi,
                   overlap_threshold=args.overlap_threshold, blur_threshold=args.blur_threshold)

if __name__ == '__main__':
    cli()