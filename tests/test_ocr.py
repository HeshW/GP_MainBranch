"""
Lightweight tests for the OCR model (Model A).

These tests do **not** require PaddleOCR or a GPU; they exercise the
regex-based extraction layer and the preprocessing utilities in isolation.
"""

from __future__ import annotations

import json

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


# ---------------------------------------------------------------------------
# v2 – A) Multi-match extraction + duplicate policy
# ---------------------------------------------------------------------------

class TestMultiMatchPolicy:
    def test_multi_match_keeps_last(self):
        """When a lab key appears twice the last match is used."""
        labs, _ = _parse_labs("Glucose: 80 mg/dL ... Glucose: 95 mg/dL")
        assert labs["glucose"]["value"] == 95.0

    def test_multi_match_emits_warning(self):
        """A warning is emitted whenever more than one match is found."""
        _, warnings = _parse_labs("Glucose: 80 mg/dL ... Glucose: 95 mg/dL")
        assert any("Multiple matches" in w and "glucose" in w for w in warnings)

    def test_multi_match_warning_includes_count(self):
        """The warning reports the number of matches found."""
        _, warnings = _parse_labs("Glucose: 80 mg/dL ... Glucose: 95 mg/dL")
        assert any("(2)" in w for w in warnings)

    def test_single_match_no_warning(self):
        """No multi-match warning when the lab appears exactly once."""
        _, warnings = _parse_labs("Glucose: 95 mg/dL")
        assert not any("Multiple matches" in w for w in warnings)

    def test_multi_match_last_unit_used(self):
        """The unit of the last match is preserved alongside its value."""
        labs, _ = _parse_labs("Glucose: 80 mg/dL ... Glucose: 5.2 mmol/L")
        assert labs["glucose"]["unit"] == "mmol/L"


# ---------------------------------------------------------------------------
# v2 – B) Broadened unit capture
# ---------------------------------------------------------------------------

class TestBroadenedUnitCapture:
    def test_unit_10_power_3_per_ul(self):
        """Units of the form 10^3/µL are captured correctly."""
        labs, _ = _parse_labs("WBC: 6.5 10^3/µL")
        assert labs["wbc"]["unit"] == "10^3/µL"

    def test_unit_10_power_9_per_l(self):
        """Units of the form 10^9/L are captured correctly."""
        labs, _ = _parse_labs("RBC: 5.0 10^9/L")
        assert labs["rbc"]["unit"] == "10^9/L"

    def test_unit_x10_power_3_per_ul(self):
        """Units of the form x10^3/uL (with 'x' prefix) are captured."""
        labs, _ = _parse_labs("PLT: 250 x10^3/uL")
        assert labs["platelets"]["unit"] == "x10^3/uL"

    def test_unit_percent(self):
        """Percentage unit is captured correctly."""
        labs, _ = _parse_labs("HCT: 45 %")
        assert labs["hematocrit"]["unit"] == "%"

    def test_existing_unit_mg_dl_still_works(self):
        labs, _ = _parse_labs("Glucose: 95 mg/dL")
        assert labs["glucose"]["unit"] == "mg/dL"

    def test_existing_unit_g_dl_still_works(self):
        labs, _ = _parse_labs("HGB: 13.5 g/dL")
        assert labs["hemoglobin"]["unit"] == "g/dL"

    def test_existing_unit_mmol_l_still_works(self):
        labs, _ = _parse_labs("Glucose: 5.2 mmol/L")
        assert labs["glucose"]["unit"] == "mmol/L"

    def test_existing_unit_ug_dl_still_works(self):
        labs, _ = _parse_labs("Iron: 80 µg/dL")
        assert labs["iron"]["unit"] == "µg/dL"

    def test_bare_number_not_captured_as_unit(self):
        """A bare number following the value must not be treated as a unit."""
        labs, _ = _parse_labs("Glucose: 95 100")
        # 100 should NOT be captured as the unit
        assert labs["glucose"]["unit"] is None


# ---------------------------------------------------------------------------
# v2 – C) Line-by-line and cross-line fallback
# ---------------------------------------------------------------------------

class TestLineByLineParsing:
    def test_lab_found_on_isolated_line(self):
        """A lab on its own line is found even among unrelated lines."""
        text = "Patient: John Doe\nGlucose: 95 mg/dL\nAge: 30"
        labs, _ = _parse_labs(text)
        assert "glucose" in labs
        assert labs["glucose"]["value"] == 95.0

    def test_multiple_labs_on_separate_lines(self):
        text = "Glucose: 95 mg/dL\nHGB: 14.0 g/dL\nIron: 75 µg/dL"
        labs, _ = _parse_labs(text)
        assert labs["glucose"]["value"] == 95.0
        assert labs["hemoglobin"]["value"] == 14.0
        assert labs["iron"]["value"] == 75.0


class TestCrossLineFallback:
    def test_value_on_next_line_is_found(self):
        """When the full pattern fails the line but the next line carries the
        value, the cross-line fallback should extract it."""
        # "(fasting)" after the colon breaks the normal separator regex,
        # causing passes 1 and 2 to fail – the fallback then fires.
        text = "Glucose: (fasting)\n5.0 mmol/L"
        labs, _ = _parse_labs(text)
        assert "glucose" in labs
        assert labs["glucose"]["value"] == 5.0
        assert labs["glucose"]["unit"] == "mmol/L"

    def test_cross_line_emits_warning(self):
        """A warning is always emitted when the cross-line fallback is used."""
        text = "Glucose: (fasting)\n5.0 mmol/L"
        _, warnings = _parse_labs(text)
        assert any("cross-line fallback" in w.lower() for w in warnings)

    def test_cross_line_source_match_contains_both_lines(self):
        """source_match for a cross-line entry joins the two lines with →."""
        text = "Glucose: (fasting)\n5.0 mmol/L"
        labs, _ = _parse_labs(text)
        sm = labs["glucose"]["source_match"]
        assert "→" in sm
        assert "Glucose" in sm
        assert "5.0" in sm

    def test_cross_line_skips_when_earlier_pass_succeeds(self):
        """If a lab is found in pass 1 or 2, the cross-line fallback is
        not invoked for it (no spurious fallback warnings)."""
        text = "Glucose: 95 mg/dL"
        _, warnings = _parse_labs(text)
        assert not any("cross-line" in w.lower() for w in warnings)

    def test_cross_line_with_digit_unit(self):
        """Cross-line fallback also captures digit-containing units."""
        text = "WBC: (automated)\n6.5 10^3/µL"
        labs, warnings = _parse_labs(text)
        assert labs["wbc"]["value"] == 6.5
        assert labs["wbc"]["unit"] == "10^3/µL"
        assert any("cross-line" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# v2 – D) source_match field
# ---------------------------------------------------------------------------

class TestSourceMatch:
    def test_source_match_present_in_every_entry(self):
        text = "Glucose: 95 mg/dL\nHGB: 13.5 g/dL"
        labs, _ = _parse_labs(text)
        for key, entry in labs.items():
            assert "source_match" in entry, f"'source_match' missing for '{key}'"

    def test_source_match_is_string(self):
        labs, _ = _parse_labs("Glucose: 95 mg/dL")
        assert isinstance(labs["glucose"]["source_match"], str)

    def test_source_match_contains_value_string(self):
        labs, _ = _parse_labs("Glucose: 95 mg/dL")
        assert "95" in labs["glucose"]["source_match"]

    def test_source_match_contains_label(self):
        labs, _ = _parse_labs("Glucose: 95 mg/dL")
        # Match is case-insensitive; the original casing is preserved in group(0)
        assert "Glucose" in labs["glucose"]["source_match"]

    def test_source_match_is_json_serialisable(self):
        result = extract_from_text("Hb: 13.0 g/dL\nGlucose: 95 mg/dL")
        serialised = json.dumps(result)
        roundtrip = json.loads(serialised)
        assert isinstance(roundtrip["labs"]["hemoglobin"]["source_match"], str)
        assert isinstance(roundtrip["labs"]["glucose"]["source_match"], str)

