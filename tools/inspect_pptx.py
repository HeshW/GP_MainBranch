import re
import sys
import zipfile
import xml.etree.ElementTree as ET


NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def slide_number(path: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", path)
    return int(match.group(1)) if match else 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pptx_path = sys.argv[1]
    with zipfile.ZipFile(pptx_path) as deck:
        slides = sorted(
            [
                name
                for name in deck.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ],
            key=slide_number,
        )
        print(f"slides: {len(slides)}")
        for index, slide in enumerate(slides, 1):
            root = ET.fromstring(deck.read(slide))
            text = " | ".join(t.text or "" for t in root.findall(".//a:t", NS))
            print(f"\n--- slide {index}: {slide}")
            print(text[:3000])


if __name__ == "__main__":
    main()
