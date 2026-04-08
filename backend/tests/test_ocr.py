"""
Lightweight tests for the OCR model (Model A).

These tests do **not** require PaddleOCR or a GPU; they exercise the
regex-based extraction layer and the preprocessing utilities in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from models.ocr.engine import extract_from_text, _normalise_text, _parse_labs, _to_rgb_array, _ensure_hwc3_uint8
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


# ---------------------------------------------------------------------------
# OCREngine – init compatibility (show_log fallback)
# ---------------------------------------------------------------------------

class TestOCREngineInit:
    """Tests for OCREngine.__init__ robustness – no real PaddleOCR required."""

    def test_show_log_accepted(self, monkeypatch):
        """When PaddleOCR accepts show_log, it is constructed once with that kwarg."""
        calls = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        import models.ocr.engine as engine_mod
        monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)

        from models.ocr.engine import OCREngine
        ocr = OCREngine()

        assert len(calls) == 1
        assert calls[0].get("show_log") is False

    def test_show_log_fallback_on_unknown_argument(self, monkeypatch):
        """When PaddleOCR rejects show_log with a ValueError, init retries without it."""
        calls = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if "show_log" in kwargs:
                    raise ValueError("Unknown argument: show_log")

        import models.ocr.engine as engine_mod
        monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)

        from models.ocr.engine import OCREngine
        ocr = OCREngine()  # must not raise

        assert len(calls) == 2, "Expected two construction attempts"
        assert "show_log" in calls[0], "First attempt should include show_log"
        assert "show_log" not in calls[1], "Second attempt must omit show_log"

    def test_unrelated_value_error_is_reraised(self, monkeypatch):
        """A ValueError unrelated to show_log must propagate, not be swallowed."""

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                raise ValueError("Something else went wrong")

        import models.ocr.engine as engine_mod
        monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)

        from models.ocr.engine import OCREngine
        with pytest.raises(ValueError, match="Something else went wrong"):
            OCREngine()

    def test_import_error_when_paddle_unavailable(self, monkeypatch):
        """OCREngine raises ImportError when PaddleOCR is not installed."""
        import models.ocr.engine as engine_mod
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", False)

        from models.ocr.engine import OCREngine
        with pytest.raises(ImportError, match="PaddleOCR is not installed"):
            OCREngine()


# ---------------------------------------------------------------------------
# OCREngine – extract() cls kwarg compatibility
# ---------------------------------------------------------------------------

class TestOCREngineExtractClsCompat:
    """Tests for OCREngine.extract() cls-argument fallback – no real PaddleOCR required."""

    def _make_engine(self, monkeypatch, fake_ocr_instance):
        """Return an OCREngine whose internal _ocr is *fake_ocr_instance*.

        PIL is patched so that any path input returns a tiny 1×1 RGB array,
        allowing tests to pass path strings without needing valid image files.
        """
        import models.ocr.engine as engine_mod

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                pass

        # Fake PIL so _to_rgb_array can open any existing path without real PNG data.
        class _FakePILImg:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def convert(self, mode):
                return np.zeros((1, 1, 3), dtype=np.uint8)

        class _FakePIL:
            Image = _FakePILImg

            @staticmethod
            def open(path):
                return _FakePILImg()

        monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)
        monkeypatch.setattr(engine_mod, "_PILImage", _FakePIL)
        monkeypatch.setattr(engine_mod, "_PIL_AVAILABLE", True)

        from models.ocr.engine import OCREngine
        engine = OCREngine(preprocess_image=False)
        engine._ocr = fake_ocr_instance
        return engine

    def test_cls_accepted_called_with_cls_true(self, monkeypatch, tmp_path):
        """Case A: when .ocr() accepts cls, it is called with cls=True."""
        calls = []

        class FakeOCR:
            def ocr(self, img, cls=None):
                calls.append({"img": img, "cls": cls})
                return []  # empty result – no text

        img = tmp_path / "img.png"
        img.write_bytes(b"")

        engine = self._make_engine(monkeypatch, FakeOCR())
        engine.extract(str(img))

        assert len(calls) == 1
        assert calls[0]["cls"] is True

    def test_cls_rejected_fallback_without_cls(self, monkeypatch, tmp_path):
        """Case B: when .ocr() raises TypeError for cls, it is retried without cls."""
        calls = []

        class FakeOCR:
            def ocr(self, img, **kwargs):
                calls.append(kwargs)
                if "cls" in kwargs:
                    raise TypeError(
                        "PaddleOCR.predict() got an unexpected keyword argument 'cls'"
                    )
                return []  # fallback call succeeds

        img = tmp_path / "img.png"
        img.write_bytes(b"")

        engine = self._make_engine(monkeypatch, FakeOCR())
        engine.extract(str(img))

        assert len(calls) == 2, "Expected two .ocr() calls (first with cls, then without)"
        assert "cls" in calls[0], "First call should include cls"
        assert "cls" not in calls[1], "Second (fallback) call must omit cls"

    def test_unrelated_type_error_is_wrapped_as_runtime_error(self, monkeypatch, tmp_path):
        """Case C: a TypeError unrelated to cls propagates as RuntimeError."""

        class FakeOCR:
            def ocr(self, img, **kwargs):
                raise TypeError("some other problem")

        img = tmp_path / "img.png"
        img.write_bytes(b"")

        engine = self._make_engine(monkeypatch, FakeOCR())
        with pytest.raises(RuntimeError, match="PaddleOCR failed to process"):
            engine.extract(str(img))


# ---------------------------------------------------------------------------
# OCREngine – extract() input normalisation (ndarray / PIL / path → RGB array)
# ---------------------------------------------------------------------------

class TestOCREngineExtractInputNormalisation:
    """Regression tests for ndarray/PIL input and RGB normalisation.

    No real PaddleOCR or image files required – PaddleOCR and PIL are both
    monkeypatched.
    """

    def _make_engine(self, monkeypatch, fake_ocr_instance, *, preprocess_image=False):
        """Return an OCREngine with *fake_ocr_instance* wired in."""
        import models.ocr.engine as engine_mod

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)

        from models.ocr.engine import OCREngine
        engine = OCREngine(preprocess_image=preprocess_image)
        engine._ocr = fake_ocr_instance
        return engine

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _capturing_fake_ocr():
        """Return a FakeOCR that records every call to .ocr() and its first arg."""
        calls = []

        class FakeOCR:
            def ocr(self, img, **kwargs):
                calls.append(img)
                return []

        return FakeOCR(), calls

    # ------------------------------------------------------------------ tests

    def test_numpy_rgb_array_accepted_directly(self, monkeypatch):
        """extract() must accept a (H,W,3) uint8 ndarray without calling Path()."""
        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        engine.extract(arr)

        assert len(calls) == 1
        assert isinstance(calls[0], np.ndarray), "ocr() should receive a numpy array"

    def test_numpy_array_never_wrapped_in_path(self, monkeypatch):
        """Passing a numpy array must not trigger Path(image) construction."""
        import models.ocr.engine as engine_mod

        path_calls = []
        original_path = engine_mod.Path

        class SpyPath(original_path):
            def __new__(cls, *args, **kwargs):
                path_calls.append(args)
                return super().__new__(cls, *args, **kwargs)

        monkeypatch.setattr(engine_mod, "Path", SpyPath)

        fake_ocr, _ = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        engine.extract(arr)

        # Path() must not have been called with the numpy array.
        assert not any(
            isinstance(a[0], np.ndarray) for a in path_calls if a
        ), "Path() should not be called with a numpy array"

    def test_rgba_array_converted_to_rgb(self, monkeypatch):
        """A (H,W,4) RGBA array must be converted to (H,W,3) RGB before ocr()."""
        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255  # set alpha channel
        engine.extract(rgba)

        assert len(calls) == 1
        assert calls[0].shape[2] == 3, "Alpha channel should have been dropped"

    def test_grayscale_array_converted_to_rgb(self, monkeypatch):
        """A (H,W) grayscale array must be expanded to (H,W,3) RGB before ocr()."""
        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        gray = np.full((4, 4), 128, dtype=np.uint8)
        engine.extract(gray)

        assert len(calls) == 1
        assert calls[0].ndim == 3 and calls[0].shape[2] == 3, (
            "Grayscale array should have been expanded to 3-channel RGB"
        )

    def test_path_input_opened_via_pil_and_converted_to_rgb(self, monkeypatch, tmp_path):
        """Path inputs must be loaded via PIL.Image.open and converted to RGB."""
        import models.ocr.engine as engine_mod

        open_calls = []

        class _FakePILImg:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def convert(self, mode):
                open_calls.append(mode)
                return np.zeros((1, 1, 3), dtype=np.uint8)

        class _FakePIL:
            Image = _FakePILImg

            @staticmethod
            def open(path):
                return _FakePILImg()

        monkeypatch.setattr(engine_mod, "_PILImage", _FakePIL)
        monkeypatch.setattr(engine_mod, "_PIL_AVAILABLE", True)

        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        img_path = tmp_path / "report.png"
        img_path.write_bytes(b"")  # just needs to exist

        engine.extract(str(img_path))

        assert "RGB" in open_calls, "PIL convert('RGB') should have been called"
        assert len(calls) == 1
        assert isinstance(calls[0], np.ndarray), "ocr() should receive a numpy array"

    def test_ocr_called_with_ndarray_not_string(self, monkeypatch, tmp_path):
        """ocr() must always receive a numpy array, never a file-path string."""
        import models.ocr.engine as engine_mod

        class _FakePILImg:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def convert(self, mode): return np.zeros((2, 2, 3), dtype=np.uint8)

        class _FakePIL:
            Image = _FakePILImg
            @staticmethod
            def open(path): return _FakePILImg()

        monkeypatch.setattr(engine_mod, "_PILImage", _FakePIL)
        monkeypatch.setattr(engine_mod, "_PIL_AVAILABLE", True)

        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        img_path = tmp_path / "report.png"
        img_path.write_bytes(b"")

        engine.extract(str(img_path))

        assert len(calls) == 1
        assert isinstance(calls[0], np.ndarray), (
            "ocr() must be called with a numpy array, not a path string"
        )
        assert not isinstance(calls[0], str)

    def test_file_not_found_for_missing_path(self, monkeypatch):
        """extract() must raise FileNotFoundError for non-existent paths."""
        fake_ocr, _ = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        with pytest.raises(FileNotFoundError):
            engine.extract("/no/such/file.png")

    def test_non_uint8_array_is_cast_to_uint8(self, monkeypatch):
        """A float64 ndarray should be cast to uint8 before ocr()."""
        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr)

        arr = np.full((4, 4, 3), 0.5, dtype=np.float64)
        engine.extract(arr)

        assert calls[0].dtype == np.uint8, "Array should be cast to uint8"


# ---------------------------------------------------------------------------
# _to_rgb_array – unit tests for the normalisation helper
# ---------------------------------------------------------------------------

class TestToRgbArray:
    """Unit tests for the ``_to_rgb_array`` helper."""

    def test_rgb_array_passthrough(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        out = _to_rgb_array(arr)
        assert out.shape == (4, 4, 3)
        assert out.dtype == np.uint8

    def test_rgba_drops_alpha(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        out = _to_rgb_array(arr)
        assert out.shape == (4, 4, 3)

    def test_grayscale_2d_to_rgb(self):
        arr = np.full((4, 4), 200, dtype=np.uint8)
        out = _to_rgb_array(arr)
        assert out.shape == (4, 4, 3)
        # All three channels should equal the original value.
        assert np.all(out[:, :, 0] == 200)
        assert np.all(out[:, :, 1] == 200)
        assert np.all(out[:, :, 2] == 200)

    def test_single_channel_3d_to_rgb(self):
        arr = np.full((4, 4, 1), 100, dtype=np.uint8)
        out = _to_rgb_array(arr)
        assert out.shape == (4, 4, 3)

    def test_float_array_cast_to_uint8(self):
        arr = np.full((2, 2, 3), 128.0, dtype=np.float32)
        out = _to_rgb_array(arr)
        assert out.dtype == np.uint8

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported image type"):
            _to_rgb_array(12345)

    def test_missing_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _to_rgb_array("/definitely/does/not/exist.png")

    def test_pil_image_converted_to_rgb(self, monkeypatch):
        """PIL.Image.Image input is converted to an RGB ndarray."""
        import models.ocr.engine as engine_mod

        class FakePILImage:
            def convert(self, mode):
                return np.zeros((2, 2, 3), dtype=np.uint8)

        # Make isinstance check succeed by patching _PILImage.Image.
        class FakePIL:
            class Image(FakePILImage):
                pass

        monkeypatch.setattr(engine_mod, "_PILImage", FakePIL)
        monkeypatch.setattr(engine_mod, "_PIL_AVAILABLE", True)

        pil_img = FakePIL.Image()
        out = engine_mod._to_rgb_array(pil_img)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# _ensure_hwc3_uint8 – unit tests for the channel-normalisation helper
# ---------------------------------------------------------------------------

class TestEnsureHwc3Uint8:
    """Unit tests for :func:`_ensure_hwc3_uint8`."""

    def test_rgb_passthrough(self):
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        out = _ensure_hwc3_uint8(arr)
        assert out.shape == (5, 5, 3)
        assert out.dtype == np.uint8

    def test_grayscale_2d_expanded(self):
        arr = np.full((5, 5), 128, dtype=np.uint8)
        out = _ensure_hwc3_uint8(arr)
        assert out.shape == (5, 5, 3)
        assert out.dtype == np.uint8
        assert np.all(out[:, :, 0] == 128)
        assert np.all(out[:, :, 1] == 128)
        assert np.all(out[:, :, 2] == 128)

    def test_rgba_alpha_dropped(self):
        arr = np.zeros((5, 5, 4), dtype=np.uint8)
        arr[:, :, :3] = 42
        out = _ensure_hwc3_uint8(arr)
        assert out.shape == (5, 5, 3)
        assert np.all(out == 42)

    def test_float_cast_to_uint8(self):
        arr = np.full((3, 3, 3), 200.0, dtype=np.float32)
        out = _ensure_hwc3_uint8(arr)
        assert out.dtype == np.uint8

    def test_unsupported_channel_count_raises(self):
        arr = np.zeros((3, 3, 2), dtype=np.uint8)
        with pytest.raises(TypeError, match="Unsupported channel count"):
            _ensure_hwc3_uint8(arr)

    def test_unsupported_ndim_raises(self):
        arr = np.zeros((3,), dtype=np.uint8)
        with pytest.raises(TypeError, match="Unsupported array shape"):
            _ensure_hwc3_uint8(arr)

    def test_non_array_raises(self):
        with pytest.raises(TypeError, match="Expected a numpy.ndarray"):
            _ensure_hwc3_uint8("not an array")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OCREngine – preprocess path produces 3-channel output (regression for #6)
# ---------------------------------------------------------------------------

class TestOCREnginePreprocessChannelNormalisation:
    """Regression tests: engine must call .ocr() with a (H,W,3) array even
    when the preprocessing pipeline returns a 2D grayscale array.

    No real PaddleOCR required – everything is monkeypatched.
    """

    def _make_engine(self, monkeypatch, fake_ocr_instance, *, preprocess_image=True):
        import models.ocr.engine as engine_mod

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(engine_mod, "_PaddleOCR", FakePaddleOCR)
        monkeypatch.setattr(engine_mod, "_PADDLEOCR_AVAILABLE", True)

        from models.ocr.engine import OCREngine
        engine = OCREngine(preprocess_image=preprocess_image)
        engine._ocr = fake_ocr_instance
        return engine

    @staticmethod
    def _capturing_fake_ocr():
        calls = []

        class FakeOCR:
            def ocr(self, img, **kwargs):
                calls.append(img)
                return []

        return FakeOCR(), calls

    def test_preprocess_grayscale_output_expanded_to_3ch(self, monkeypatch):
        """When preprocess() returns a 2D array, engine must convert to (H,W,3)."""
        import models.ocr.engine as engine_mod

        # Stub preprocess to return a 2D grayscale array (as the real pipeline does).
        monkeypatch.setattr(engine_mod, "preprocess", lambda img, **kw: np.full((8, 8), 200, dtype=np.uint8))

        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr, preprocess_image=True)

        engine.extract(np.zeros((8, 8, 3), dtype=np.uint8))

        assert len(calls) == 1
        assert calls[0].ndim == 3, "ocr() must receive a 3D array"
        assert calls[0].shape[2] == 3, "ocr() must receive a 3-channel array"
        assert calls[0].dtype == np.uint8

    def test_preprocess_rgba_output_converted_to_3ch(self, monkeypatch):
        """When preprocess() returns a (H,W,4) RGBA array, engine drops alpha."""
        import models.ocr.engine as engine_mod

        monkeypatch.setattr(engine_mod, "preprocess", lambda img, **kw: np.zeros((8, 8, 4), dtype=np.uint8))

        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr, preprocess_image=True)

        engine.extract(np.zeros((8, 8, 3), dtype=np.uint8))

        assert len(calls) == 1
        assert calls[0].shape == (8, 8, 3)

    def test_ocr_always_receives_3ch_ndarray(self, monkeypatch):
        """Regardless of preprocess output, ocr() must be called with (H,W,3)."""
        import models.ocr.engine as engine_mod

        # preprocess returns binary grayscale (the real behaviour)
        monkeypatch.setattr(engine_mod, "preprocess", lambda img, **kw: np.zeros((6, 6), dtype=np.uint8))

        fake_ocr, calls = self._capturing_fake_ocr()
        engine = self._make_engine(monkeypatch, fake_ocr, preprocess_image=True)

        engine.extract(np.zeros((6, 6, 3), dtype=np.uint8))

        assert isinstance(calls[0], np.ndarray)
        assert calls[0].ndim == 3
        assert calls[0].shape[2] == 3

