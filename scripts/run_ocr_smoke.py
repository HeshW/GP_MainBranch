from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ocr import OCREngine


def main() -> None:
    image_path = Path("data/labreport1test.png")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing test image: {image_path}")

    engine = OCREngine()
    out = engine.extract(str(image_path))

    print("OK, result keys:", list(out.keys()))
    print("labs_count:", len(out.get("labs", {})))
    print("warnings_count:", len(out.get("warnings", [])))
    print("sample_labs:", list(out.get("labs", {}).keys())[:10])


if __name__ == "__main__":
    main()
