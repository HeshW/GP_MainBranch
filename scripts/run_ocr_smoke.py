from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from models.ocr import OCREngine


def parse_args():
    p = argparse.ArgumentParser(description='Run a quick OCR smoke test')
    p.add_argument('--image', '-i', default='data/labreport1test.png', help='Path to the test image')
    p.add_argument('--raw', action='store_true', help='Also print full raw_text')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing test image: {image_path}")

    engine = OCREngine()
    out = engine.extract(str(image_path))

    print("OK, result keys:", list(out.keys()))
    print("labs_count:", len(out.get("labs", {})))
    print("warnings_count:", len(out.get("warnings", [])))
    print("sample_labs:", list(out.get("labs", {}).keys())[:10])
    if args.raw:
        print('\n---RAW_TEXT_START---')
        print(out.get('raw_text') or '')
        print('---RAW_TEXT_END---')


if __name__ == "__main__":
    main()
