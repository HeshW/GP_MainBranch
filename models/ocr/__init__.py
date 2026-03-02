"""
models.ocr – OCR Model A
========================

Provides PaddleOCR-powered text extraction and lab-value parsing for
medical report images.

Quick start::

    from models.ocr import OCREngine

    engine = OCREngine()
    result = engine.extract("path/to/report.png")
    print(result["labs"])

If you already have raw OCR text and just need the structured parser::

    from models.ocr import extract_from_text

    result = extract_from_text("Glucose: 95 mg/dL  Hgb: 13.5 g/dL")
    print(result["labs"])
"""

from .engine import OCREngine, extract_from_text

__all__ = ["OCREngine", "extract_from_text"]
