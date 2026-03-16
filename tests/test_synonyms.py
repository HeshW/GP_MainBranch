"""Verify the optional synonyms file merges into the runtime synonym map."""
from __future__ import annotations

from models.ocr import patterns


def test_synonyms_presence():
    # These aliases should be present after loading `synonyms_v15.json`.
    assert patterns.SYNONYM_MAP.get("hgb") == "hemoglobin"
    assert patterns.SYNONYM_MAP.get("plt") == "platelets"
    assert patterns.SYNONYM_MAP.get("creatinine") == "creatinine"
