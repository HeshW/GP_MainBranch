"""
Lightweight tests for the OCR model (Model A).

These tests do **not** require PaddleOCR or a GPU; they exercise the
regex-based extraction layer and the preprocessing utilities in isolation.
"""

from __future__ import annotations

import pytest

from models.ocr.engine import extract_from_text, _normalise_text, _parse_labs
from models.ocr.patterns import LAB_PATTERNS, SYNONYM_MAP


# ---------------------------------------------------------------------------
# Pattern / synonym map sanity checks
# ---------------------------------------------------------------------------

class TestPatterns:
    def test_synonym_map_is_non_empty(self):
        assert len(SYNONYM_MAP) > 0

    def test_all_canonical_keys_have_pattern(self):
        canonical_keys = set(SYNONYM_MAP.values())
        for key in canonical_keys:
            assert key in LAB_PATTERNS, f"No pattern for canonical key '{key}'"

    def test_canonical_keys_include_core_labs(self):
        core = {"glucose", "hemoglobin", "iron"}
        missing = core - set(LAB_PATTERNS)
        assert not missing, f"Missing canonical keys: {missing}"


# ---------------------------------------------------------------------------
# _normalise_text
# ---------------------------------------------------------------------------

class TestNormaliseText:
    def test_collapses_whitespace(self):
        assert _normalise_text("a  b\tc") == "a b c"

    def test_strips_leading_trailing(self):
        assert _normalise_text("  hello  ") == "hello"

    def test_newlines_become_spaces(self):
        assert _normalise_text("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# _parse_labs – unit tests for the regex extraction layer
# ---------------------------------------------------------------------------

class TestParseLabs:
    def test_glucose_mg_dl(self):
        labs, warnings = _parse_labs("Glucose: 95 mg/dL")
        assert "glucose" in labs
        assert labs["glucose"]["value"] == 95.0
        assert labs["glucose"]["unit"] == "mg/dL"
        assert warnings == []

    def test_glucose_synonym_glu(self):
        labs, _ = _parse_labs("Glu: 5.2 mmol/L")
        assert "glucose" in labs
        assert labs["glucose"]["value"] == 5.2
        assert labs["glucose"]["unit"] == "mmol/L"

    def test_hemoglobin_abbreviation_hgb(self):
        labs, _ = _parse_labs("HGB 13.5 g/dL")
        assert "hemoglobin" in labs
        assert labs["hemoglobin"]["value"] == 13.5

    def test_hemoglobin_abbreviation_hb(self):
        labs, _ = _parse_labs("Hb - 11.0 g/dL")
        assert "hemoglobin" in labs
        assert labs["hemoglobin"]["value"] == 11.0

    def test_iron_with_colon(self):
        labs, _ = _parse_labs("Iron: 80 µg/dL")
        assert "iron" in labs
        assert labs["iron"]["value"] == 80.0

    def test_comma_decimal_separator(self):
        labs, _ = _parse_labs("Glucose: 5,4 mmol/L")
        assert labs["glucose"]["value"] == pytest.approx(5.4)

    def test_case_insensitive(self):
        labs, _ = _parse_labs("GLUCOSE 100 MG/DL")
        assert "glucose" in labs

    def test_multiple_labs_in_one_text(self):
        text = "Glucose: 95 mg/dL\nHgb: 14.0 g/dL\nIron: 75 µg/dL"
        labs, warnings = _parse_labs(text)
        assert "glucose" in labs
        assert "hemoglobin" in labs
        assert "iron" in labs
        assert warnings == []

    def test_missing_unit_returns_none(self):
        labs, _ = _parse_labs("Glucose 95")
        assert "glucose" in labs
        assert labs["glucose"]["unit"] is None

    def test_no_match_returns_empty(self):
        labs, warnings = _parse_labs("Patient name: John Doe")
        assert labs == {}
        assert warnings == []

    def test_extra_whitespace_between_label_and_value(self):
        labs, _ = _parse_labs("Glucose  :  95   mg/dL")
        assert labs["glucose"]["value"] == 95.0

    def test_wbc_synonym(self):
        labs, _ = _parse_labs("WBC: 6.5 10^3/µL")
        assert "wbc" in labs

    def test_platelets_synonym_plt(self):
        labs, _ = _parse_labs("PLT 250 10^3/µL")
        assert "platelets" in labs


# ---------------------------------------------------------------------------
# extract_from_text (public convenience function)
# ---------------------------------------------------------------------------

class TestExtractFromText:
    def test_returns_required_keys(self):
        result = extract_from_text("Glucose: 100 mg/dL")
        assert "labs" in result
        assert "raw_text" in result
        assert "warnings" in result

    def test_raw_text_is_normalised(self):
        result = extract_from_text("  Glucose :  100  mg/dL  ")
        assert result["raw_text"] == "Glucose : 100 mg/dL"

    def test_result_is_json_serialisable(self):
        import json
        result = extract_from_text("Hb: 13.0 g/dL\nGlucose: 95 mg/dL")
        # Should not raise
        serialised = json.dumps(result)
        roundtrip = json.loads(serialised)
        assert roundtrip["labs"]["hemoglobin"]["value"] == 13.0

    def test_empty_string(self):
        result = extract_from_text("")
        assert result["labs"] == {}
        assert result["raw_text"] == ""
        assert result["warnings"] == []
