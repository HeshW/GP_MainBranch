"""Unit test validating `raw_ocr` integration in OCREngine.

This test monkeypatches a fake PaddleOCR object so the test remains
lightweight and does not require the full Paddle stack.
"""
from __future__ import annotations

import pytest

import models.ocr.engine as engine_mod


class FakePaddleOCR:
    def __init__(self, **kwargs):
        pass

    def ocr(self, img, cls=True):
        # Simulate PaddleOCR output: list of pages -> list of lines
        # Each line is [bbox, (text, confidence)]
        return [
            [
                ([[0, 0], [10, 0], [10, 10], [0, 10]], ("Line one", 0.98)),
                ([[0, 11], [10, 11], [10, 21], [0, 21]], ("Line two", 0.75)),
            ]
        ]


def test_ocengine_returns_raw_ocr(monkeypatch, tmp_path):
    # Arrange: monkeypatch PaddleOCR availability and implementation
    monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)
    monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)

    from models.ocr.engine import OCREngine

    # Act
    ocr = OCREngine()
    out = ocr.extract("<array>")  # input is not used by fake ocr in this test

    # Assert
    assert "raw_ocr" in out
    assert isinstance(out["raw_ocr"], list)
    assert len(out["raw_ocr"]) == 2
    first = out["raw_ocr"][0]
    assert first["text"] == "Line one"
    assert pytest.approx(first["confidence"], rel=1e-2) == 0.98
    assert isinstance(first["bbox"], list)
