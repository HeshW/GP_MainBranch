#!/usr/bin/env python3
"""
Synthetic pathology/lab report image generator.

Generates realistic-looking scanned document images (PNG) with:
 - Header metadata (title, patient name, referred by, sex/age, date, path code, vcode)
 - Labeled narrative sections (Clinical, Gross, Microscopic, Diagnosis, Notes)
 - Optional barcode (Code128) or text-only VCode area
 - Simple visual augmentations to mimic scanning/noise/printing artifacts
 - Per-image JSON annotations containing field bounding boxes and section blocks

Output:
 - images: <out_dir>/images/report_00001.png
 - annotations: <out_dir>/annotations/report_00001.json

Dependencies (install via pip):
 - Pillow
 - numpy
 - python-barcode[pillow] (optional, for barcode images)
 - qrcode (optional, for QR codes)
 - opencv-python (optional, for some augmentations; Pillow-only augmentations are included)

Usage:
    python scripts/generate_synthetic_reports.py --out-dir ./synthetic_dataset --count 100

See --help for more options.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import string
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

if TYPE_CHECKING:
    # Optional heavy deps are imported only for type-checkers to avoid linter
    # unresolved-import warnings in editors that don't use the project's venv.
    try:  # pragma: no cover - typing-only
        import barcode  # type: ignore
        import qrcode  # type: ignore
    except Exception:
        pass

# Optional imports
try:
    import barcode  # python-barcode
    from barcode.writer import ImageWriter

    BARCODE_AVAILABLE = True
except Exception:
    BARCODE_AVAILABLE = False

try:
    import qrcode

    QRCODE_AVAILABLE = True
except Exception:
    QRCODE_AVAILABLE = False

# ------------------------
# Constants & templates
# ------------------------
DEFAULT_WIDTH = 1654  # ~ A4 @ 150 DPI (approx)
DEFAULT_HEIGHT = 2339

SECTION_LABELS = ["Clinical", "Gross", "Microscopic", "Diagnosis", "Notes"]

# Example sentence fragments for synthetic paragraphs (domain-flavored but generic)
PARA_SNIPPETS = [
    "Mild elevated enzymes with negative viral markers. Liver biopsy.",
    "Cores of tissue measuring approximately 1.5 cm were prepared and stained.",
    "Examination revealed tissue sections with no evidence of significant fibrosis.",
    "Few portal areas showed minimal lymphocytic infiltrates without plasma cell predominance.",
    "No evidence of viral inclusions or granulomatous inflammation.",
    "Intact architecture with minimal inflammatory reaction and no portal fibrosis.",
    "This picture is suggestive of a minimal drug-induced reaction.",
    "No pathological features to suggest autoimmune hepatitis.",
]

LAB_TEMPLATES = [
    ("glucose", "Glucose", (70.0, 140.0), "mg/dL"),
    ("hemoglobin", "HGB", (11.0, 17.5), "g/dL"),
    ("wbc", "WBC", (3.5, 11.0), "10^3/L"),
    ("platelets", "PLT", (150.0, 450.0), "10^3/L"),
]

# ------------------------
# Helpers
# ------------------------


def rand_name() -> str:
    fn = random.choice(
        [
            "Ahmed",
            "Fatima",
            "Omar",
            "Sara",
            "Khaled",
            "Mona",
            "Youssef",
            "Layla",
            "Hassan",
            "Nadia",
        ]
    )
    ln = random.choice(
        [
            "Ali",
            "Hassan",
            "Ibrahim",
            "Zaki",
            "Kamal",
            "Abdullah",
            "Zalat",
            "Farah",
            "Khalid",
            "Nour",
        ]
    )
    # Add numeric suffix to diversify
    return f"{fn} {ln}"


def rand_date(start_year=2018, end_year=2026) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    d = start + timedelta(days=random.randint(0, delta))
    return d.strftime("%d/%m/%Y")


def rand_vcode(length: int = 8) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def make_paragraph(min_sent=2, max_sent=5) -> str:
    s = " ".join(random.choice(PARA_SNIPPETS) for _ in range(random.randint(min_sent, max_sent)))
    # Clean spacing/punctuation a bit
    s = s.replace(" .", ".")
    return s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _font_text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    """Pillow-version-safe text sizing helper."""
    try:
        l, t, r, b = font.getbbox(text)
        return max(1, int(r - l)), max(1, int(b - t))
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return max(1, len(text) * 8), 14


def make_lab_lines() -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    """Generate parse-friendly lab lines and their expected structured values."""
    lines: List[str] = []
    expected: Dict[str, Dict[str, object]] = {}
    for canonical_key, label, value_range, unit in LAB_TEMPLATES:
        lo, hi = value_range
        # Keep one decimal place so parsing expectations are stable and realistic.
        value = round(random.uniform(lo, hi), 1)
        if unit:
            lines.append(f"{label}: {value} {unit}")
        else:
            lines.append(f"{label}: {value}")
        expected[canonical_key] = {"value": float(value), "unit": unit}
    return lines, expected


# ------------------------
# Rendering functions
# ------------------------


def load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    fallback_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    except Exception:
        pass
    for candidate in fallback_candidates:
        try:
            if os.path.exists(candidate):
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    # Fallback to a default PIL font (may be small)
    return ImageFont.load_default()


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    spacing: int = 4,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """
    Draw wrapped text at position xy (left, top). Returns the drawn text and bbox (x, y, w, h).
    Uses textwrap to wrap at character level heuristics.
    """
    x, y = xy
    wrapper = textwrap.TextWrapper(width=100)
    # Estimate width-based wrap by measuring characters per line for font
    # Simple heuristic: measure average char width
    avg_char_width = sum(_font_text_size(font, c)[0] for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") / 52
    max_chars = max(20, int(max_width / (avg_char_width + 1)))
    wrapper.width = max_chars
    lines = wrapper.wrap(text)
    cur_y = y
    max_w = 0
    for i, line in enumerate(lines):
        draw.text((x, cur_y), line, font=font, fill=0)
        w, h = _font_text_size(font, line)
        max_w = max(max_w, w)
        cur_y += h + spacing
    h_total = cur_y - y
    return "\n".join(lines), (x, y, max_w, h_total)


def render_report_image(
    width: int,
    height: int,
    fonts: Dict[str, ImageFont.FreeTypeFont],
    include_barcode: bool,
    make_qr: bool,
    random_seed: Optional[int] = None,
) -> Tuple[Image.Image, Dict]:
    """
    Render a single synthetic report and return (PIL.Image, annotation_dict)
    annotation_dict contains:
      - fields: dict of named fields -> {"text": str, "bbox": [x,y,w,h]}
      - sections: list of {"label": str, "text": str, "bbox": [x,y,w,h]}
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    img = Image.new("RGB", (width, height), color=(245, 245, 240))  # slightly off-white
    draw = ImageDraw.Draw(img)

    # Margins and layout
    margin = int(width * 0.06)
    content_w = width - 2 * margin
    y = margin

    annotation = {"fields": {}, "sections": [], "expected_labs": {}}

    # Title
    title_font = fonts.get("title")
    title_text = "Pathology Report"
    tw, th = _font_text_size(title_font, title_text)
    draw.text((margin + (content_w - tw) // 2, y), title_text, font=title_font, fill=0)
    y += th + 10

    # Top metadata block (two-column)
    meta_font = fonts.get("meta")
    label_font = fonts.get("label")
    left_x = margin
    right_x = margin + int(content_w * 0.55)
    col_w = int(content_w * 0.43)

    # Patient name
    patient_name = rand_name()
    pn_label = "Patient's Name:"
    draw.text((left_x, y), pn_label, font=label_font, fill=0)
    draw.text((left_x + 150, y), patient_name, font=meta_font, fill=0)
    # measure bbox for field
    pn_bbox = (
        left_x + 150,
        y,
        int(col_w),
        _font_text_size(meta_font, patient_name)[1],
    )
    annotation["fields"]["patient_name"] = {"text": patient_name, "bbox": [pn_bbox[0], pn_bbox[1], pn_bbox[2], pn_bbox[3]]}

    # Sex/Age on right
    sex_age = f"{random.choice(['Male', 'Female'])} / {random.randint(1, 95)} year"
    draw.text((right_x, y), "Sex/Age:", font=label_font, fill=0)
    draw.text((right_x + 110, y), sex_age, font=meta_font, fill=0)
    sa_bbox = (right_x + 110, y, int(col_w), _font_text_size(meta_font, sex_age)[1])
    annotation["fields"]["sex_age"] = {"text": sex_age, "bbox": [sa_bbox[0], sa_bbox[1], sa_bbox[2], sa_bbox[3]]}
    y += _font_text_size(meta_font, patient_name)[1] + 8

    # Referred by (left)
    referred = f"Prof. Dr. {random.choice(['Khaled', 'Zein', 'Nabil', 'Ramy'])}"
    draw.text((left_x, y), "Referred by:", font=label_font, fill=0)
    draw.text((left_x + 110, y), referred, font=meta_font, fill=0)
    ref_bbox = (left_x + 110, y, int(col_w), _font_text_size(meta_font, referred)[1])
    annotation["fields"]["referred_by"] = {"text": referred, "bbox": [ref_bbox[0], ref_bbox[1], ref_bbox[2], ref_bbox[3]]}

    # Date and VCode / Path Code (right)
    date_text = rand_date()
    draw.text((right_x, y), "Date:", font=label_font, fill=0)
    draw.text((right_x + 60, y), date_text, font=meta_font, fill=0)
    date_bbox = (right_x + 60, y, int(col_w), _font_text_size(meta_font, date_text)[1])
    annotation["fields"]["date"] = {"text": date_text, "bbox": [date_bbox[0], date_bbox[1], date_bbox[2], date_bbox[3]]}

    y += _font_text_size(meta_font, referred)[1] + 12

    # Path Code (left)
    path_code = str(random.randint(1, 99))
    draw.text((left_x, y), "Path. Code:", font=label_font, fill=0)
    draw.text((left_x + 110, y), path_code, font=meta_font, fill=0)
    pc_bbox = (left_x + 110, y, int(col_w), _font_text_size(meta_font, path_code)[1])
    annotation["fields"]["path_code"] = {"text": path_code, "bbox": [pc_bbox[0], pc_bbox[1], pc_bbox[2], pc_bbox[3]]}

    # VCode & optional barcode (right)
    vcode = rand_vcode(9)
    draw.text((right_x, y), "VCode:", font=label_font, fill=0)
    draw.text((right_x + 60, y), vcode, font=meta_font, fill=0)
    vcode_bbox = (right_x + 60, y, int(col_w), _font_text_size(meta_font, vcode)[1])
    annotation["fields"]["vcode"] = {"text": vcode, "bbox": [vcode_bbox[0], vcode_bbox[1], vcode_bbox[2], vcode_bbox[3]]}

    # Reserve barcode area beneath VCode if requested
    bc_y = y + _font_text_size(meta_font, vcode)[1] + 6
    if include_barcode and BARCODE_AVAILABLE:
        try:
            bc = barcode.get("code128", vcode, writer=ImageWriter())
            # render barcode to memory
            bc_io = io.BytesIO()
            bc.write(bc_io, options={"module_height": 10.0, "quiet_zone": 1.0})
            bc_io.seek(0)
            bc_img = Image.open(bc_io).convert("RGB")
            # resize to fit area
            target_w = int(content_w * 0.35)
            ratio = target_w / bc_img.width
            bc_img = bc_img.resize((target_w, int(bc_img.height * ratio)), Image.LANCZOS)
            img.paste(bc_img, (right_x + 20, bc_y))
            annotation["fields"]["barcode_bbox"] = [right_x + 20, bc_y, bc_img.width, bc_img.height]
            bc_y += bc_img.height + 6
        except Exception:
            # barcode generation failed; fallback to drawing simple bars
            pass
    elif include_barcode and QRCODE_AVAILABLE and make_qr:
        qr = qrcode.make(vcode)
        qr = qr.resize((160, 160))
        img.paste(qr, (right_x + 20, bc_y))
        annotation["fields"]["qrcode_bbox"] = [right_x + 20, bc_y, 160, 160]
        bc_y += 160 + 6
    else:
        # draw a faint placeholder line for barcode area
        draw.line((right_x + 20, bc_y, right_x + 160, bc_y), fill=0)
        bc_y += 14

    y = max(bc_y + 8, y + 40)

    # Separator line
    draw.line((margin, y, width - margin, y), fill=0)
    y += 10

    section_x = margin + 10
    section_w = content_w - 20

    # Lab panel for OCR model validation.
    # This block creates deterministic parse-friendly lines and stores ground truth.
    lab_title_font = fonts.get("section_label")
    lab_text_font = fonts.get("section_text")
    lab_title = "Lab Results:"
    draw.text((section_x, y), lab_title, font=lab_title_font, fill=0)
    y += _font_text_size(lab_title_font, lab_title)[1] + 6
    lab_lines, expected_labs = make_lab_lines()
    lab_start_y = y
    lab_max_w = 0
    for line in lab_lines:
        draw.text((section_x, y), line, font=lab_text_font, fill=0)
        lw, lh = _font_text_size(lab_text_font, line)
        lab_max_w = max(lab_max_w, lw)
        y += lh + 5
    lab_block_text = "\n".join(lab_lines)
    lab_bbox = (section_x, lab_start_y, lab_max_w, y - lab_start_y)
    annotation["sections"].append(
        {
            "label": "Lab Results",
            "text": lab_block_text,
            "bbox": [lab_bbox[0], lab_bbox[1], lab_bbox[2], lab_bbox[3]],
        }
    )
    annotation["expected_labs"] = expected_labs
    y += 14

    # Sections: render each label and a paragraph, capture bbox
    for label in SECTION_LABELS:
        # label
        label_font_local = fonts.get("section_label")
        draw.text((section_x, y), f"{label}:", font=label_font_local, fill=0)
        label_h = _font_text_size(label_font_local, label)[1]
        y += label_h + 6
        # paragraph
        para = make_paragraph()
        wrapped, bbox = draw_multiline(draw, (section_x, y), para, fonts.get("section_text"), section_w, spacing=6)
        # bbox returned is (x, y, w, h)
        annotation["sections"].append({"label": label, "text": para, "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]]})
        y += bbox[3] + 12
        # small horizontal gap between sections
        if y > height - 200:
            break

    # Signature block bottom-right
    sig_text = "Pathologist\nProf. Dr. " + random.choice(["Khaled Zalata", "A. Shehab", "R. Mansour"])
    sig_font = fonts.get("meta")
    sx = width - margin - 300
    sy = height - margin - 80
    draw.multiline_text((sx, sy), sig_text, font=sig_font, fill=0, spacing=4, align="right")
    annotation["fields"]["signature"] = {"text": sig_text, "bbox": [sx, sy, 300, 60]}

    return img, annotation


# ------------------------
# Simple augmentations
# ------------------------


def apply_augmentations(img: Image.Image, augment_level: float = 0.5) -> Image.Image:
    """
    Apply light augmentations to mimic scanning artifacts.
    augment_level: 0.0 - 1.0 controlling strength
    """
    arr = np.array(img).astype(np.uint8)

    # Slight rotation
    angle = random.uniform(-1.2, 1.2) * augment_level
    if abs(angle) > 0.01:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(245, 245, 240))

    # Gaussian blur (mild)
    if random.random() < 0.6 * augment_level:
        radius = random.uniform(0.3, 1.6) * augment_level
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Add paper texture / noise overlay
    noise = np.random.normal(loc=0.0, scale=8.0 * augment_level, size=(img.height, img.width, 1)).astype(np.int16)
    img_arr = np.array(img).astype(np.int16)
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)

    # Contrast / brightness
    if random.random() < 0.6:
        from PIL import ImageEnhance

        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(1.0 + (random.uniform(-0.08, 0.12) * augment_level))
        enh = ImageEnhance.Brightness(img)
        img = enh.enhance(1.0 + (random.uniform(-0.03, 0.06) * augment_level))

    # JPEG compression artifacts simulated by saving to JPEG in memory and reloading
    if random.random() < 0.4 * augment_level:
        bio = io.BytesIO()
        q = int(85 - 30 * augment_level)
        img.save(bio, format="JPEG", quality=q)
        bio.seek(0)
        img = Image.open(bio).convert("RGB")

    return img


# ------------------------
# Command-line interface
# ------------------------


def build_fonts(font_dir: Optional[str]) -> Dict[str, ImageFont.ImageFont]:
    # Try to find some reasonable fonts; user may pass a directory with .ttf files
    fonts = {}
    # default sizes
    fonts["title"] = load_font(None, size=52)
    fonts["label"] = load_font(None, size=30)
    fonts["meta"] = load_font(None, size=30)
    fonts["section_label"] = load_font(None, size=34)
    fonts["section_text"] = load_font(None, size=28)

    if font_dir and os.path.isdir(font_dir):
        # pick first few .ttf files we find to customize sizes
        ttf_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
        if ttf_files:
            fonts["title"] = load_font(ttf_files[0], size=52)
            fonts["label"] = load_font(ttf_files[0], size=30)
            fonts["meta"] = load_font(ttf_files[0], size=30)
            fonts["section_label"] = load_font(ttf_files[0], size=34)
            fonts["section_text"] = load_font(ttf_files[0], size=28)
    return fonts


def main():
    p = argparse.ArgumentParser(description="Generate synthetic scanned pathology/lab report images with annotations.")
    p.add_argument("--out-dir", required=True, help="Output directory for images and annotations.")
    p.add_argument("--count", type=int, default=100, help="Number of images to generate.")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Image width in pixels.")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Image height in pixels.")
    p.add_argument("--font-dir", type=str, default=None, help="Optional directory containing .ttf fonts to use.")
    p.add_argument("--include-barcode", action="store_true", help="Attempt to render Code128 barcode (requires python-barcode).")
    p.add_argument("--make-qr", action="store_true", help="If barcode lib missing, optionally render QR code (requires qrcode).")
    p.add_argument("--augment", type=float, default=0.6, help="Augmentation strength 0.0-1.0.")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    args = p.parse_args()

    out_images = os.path.join(args.out_dir, "images")
    out_ann = os.path.join(args.out_dir, "annotations")
    ensure_dir(out_images)
    ensure_dir(out_ann)

    fonts = build_fonts(args.font_dir)

    for i in range(1, args.count + 1):
        if args.seed is not None:
            seed = args.seed + i
        else:
            seed = None
        img, ann = render_report_image(
            width=args.width,
            height=args.height,
            fonts=fonts,
            include_barcode=args.include_barcode,
            make_qr=args.make_qr,
            random_seed=seed,
        )

        if args.augment and args.augment > 0:
            img = apply_augmentations(img, augment_level=args.augment)

        name = f"report_{i:05d}.png"
        img_path = os.path.join(out_images, name)
        img.save(img_path, format="PNG")

        # Save annotation JSON (include image filename and metadata)
        ann_out = {
            "image": os.path.relpath(img_path, args.out_dir),
            "width": args.width,
            "height": args.height,
            "fields": ann["fields"],
            "sections": ann["sections"],
            "expected_labs": ann.get("expected_labs", {}),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(out_ann, f"report_{i:05d}.json"), "w", encoding="utf-8") as f:
            json.dump(ann_out, f, ensure_ascii=False, indent=2)

        if i % 10 == 0 or i == args.count:
            print(f"Generated {i}/{args.count} -> {img_path}")


if __name__ == "__main__":
    main()