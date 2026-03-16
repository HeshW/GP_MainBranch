"""Smoke utility: print raw_text, fields, sections, and labs for an input image.

Usage:
    python models/ocr/scripts/_print_raw_text_plus_fields.py path/to/image.png --raw

This script uses the project's OCREngine and prints JSON-ish output to stdout.
"""
import argparse
import json
import os
import sys

# make repo root importable when run from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.ocr.engine import OCREngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to image file")
    p.add_argument("--raw", action="store_true", help="print raw text and fields")
    args = p.parse_args()

    engine = OCREngine()
    out = engine.extract(args.image)

    # print concise output
    result = {
        "raw_text": out.get("raw_text"),
        "fields": out.get("fields"),
        "sections": out.get("sections"),
        "raw_ocr_count": len(out.get("raw_ocr", [])),
        "labs_count": len(out.get("labs", [])),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
