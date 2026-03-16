"""Test that per-lab confidence aggregation attaches a numeric `confidence`.

This test monkeypatches PaddleOCR to return a controlled set of OCR lines
so aggregation is deterministic and lightweight.
"""
from __future__ import annotations

import pytest

import models.ocr.engine as engine_mod


class FakePaddleOCR:
    def __init__(self, **kwargs):
        pass

    def ocr(self, img, cls=True):
        # Single page with two lines; glucose line carries a high confidence.
        return [
            [
                ([[0, 0], [10, 0], [10, 10], [0, 10]], ("Glucose: 95 mg/dL", 0.85)),
                ([[0, 11], [10, 11], [10, 21], [0, 21]], ("Other text", 0.60)),
            ]
        ]


def test_lab_confidence_attached(monkeypatch):
    monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)
    monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)

    from models.ocr.engine import OCREngine

    o = OCREngine()
    out = o.extract("<array>")

    assert "glucose" in out["labs"]
    conf = out["labs"]["glucose"].get("confidence")
    assert conf is not None
    assert pytest.approx(conf, rel=1e-2) == 0.85
