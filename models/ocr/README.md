# OCR Engine – Model A (v2)

PaddleOCR-based text extraction and regex-driven lab-value parsing for
medical report images.

## v2 Improvements

| Feature | Detail |
|---------|--------|
| **Multi-match extraction** | `finditer` replaces `search`; all occurrences of a synonym are examined |
| **Duplicate policy** | Last match wins; a warning is emitted with the match count |
| **Broadened unit capture** | Digit-containing units (`10^3/µL`, `10^9/L`, `x10^3/uL`, `%`) are now parsed |
| **Multi-pass parsing** | Full text → line-by-line → cross-line fallback |
| **`source_match` field** | Every lab entry records the exact matched substring |

## Module Layout

```
models/ocr/
├── __init__.py   – package exports (OCREngine, extract_from_text)
├── engine.py     – OCREngine class and extract_from_text() helper
├── patterns.py   – regex patterns, synonym map, and new label/value patterns
├── utils.py      – OpenCV pre-processing utilities
└── README.md     – this file
```

## Usage

### Full image pipeline

```python
from models.ocr import OCREngine

engine = OCREngine(
    lang="en",             # PaddleOCR language (default: "en")
    use_angle_cls=True,    # rotate detection (default: True)
    preprocess_image=True, # apply OpenCV pipeline (default: True)
)

result = engine.extract("lab_report.png")
```

**Return value** (`result`) is a plain `dict`:

| Key | Type | Description |
|-----|------|-------------|
| `labs` | `dict` | Canonical lab name → `{"value": float, "unit": str \| None, "source_match": str}` |
| `raw_text` | `str` | Full concatenated OCR text |
| `warnings` | `list[str]` | Parsing issues (empty if everything is clean) |

### Lab entry schema (v2)

Each entry in `labs` is a `dict` with three keys:

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | Parsed numeric value |
| `unit` | `str \| None` | Unit string, or `None` when absent |
| `source_match` | `str` | Exact matched substring (or `"label → value"` for cross-line matches) |

Example:

```python
result["labs"]["glucose"]
# {'value': 95.0, 'unit': 'mg/dL', 'source_match': 'Glucose: 95 mg/dL'}
```

### Duplicate policy

When the same canonical lab key is matched more than once (e.g. a lab
value repeated in a report header and footer), the **last match** is kept
as the final value.  A warning string is added to `result["warnings"]`
that includes the match count:

```
"Multiple matches (2) for 'glucose'; keeping last match."
```

### Text-only parsing (no OCR)

```python
from models.ocr import extract_from_text

result = extract_from_text("Glucose: 95 mg/dL\nHgb: 13.5 g/dL\nIron: 75 µg/dL")
```

Useful for testing or when OCR is handled externally.

## Multi-pass Parsing

`_parse_labs` applies three passes in order:

1. **Full normalised text** – the entire OCR text is collapsed to a single
   line and searched.  All occurrences are found with `finditer`.
2. **Line-by-line** – for any lab not found in pass 1, each individual line
   is searched.  Useful when unrelated numbers would otherwise confuse a
   cross-line regex.
3. **Cross-line fallback** – for any lab still missing, a heuristic looks
   for a label-only line immediately followed by a value-only line (e.g.
   `"Glucose: (fasting)"\n"5.0 mmol/L"`).  A warning is always emitted
   when this path fires.

## Supported Lab Values

| Canonical key | Recognised synonyms |
|---------------|---------------------|
| `glucose`     | Glucose, Gluc, Glu, Blood Sugar, Blood Glucose |
| `hemoglobin`  | Hemoglobin, Haemoglobin, Hgb, Hb |
| `iron`        | Iron, Fe, Serum Iron |
| `wbc`         | WBC, White Blood Cells, White Blood Count, Leukocytes |
| `rbc`         | RBC, Red Blood Cells, Red Blood Count, Erythrocytes |
| `platelets`   | Platelets, PLT, Thrombocytes |
| `hematocrit`  | Hematocrit, Haematocrit, HCT, PCV |
| `cholesterol` | Cholesterol, Chol, Total Cholesterol |
| `creatinine`  | Creatinine, Creat, Cr |
| `urea`        | Urea, BUN, Blood Urea Nitrogen |
| `sodium`      | Sodium, Na |
| `potassium`   | Potassium, K |
| `calcium`     | Calcium, Ca |

## Recognised Unit Formats

| Format | Examples |
|--------|---------|
| Letter-first | `mg/dL`, `g/dL`, `mmol/L`, `µg/dL`, `mEq/L` |
| Percentage | `%` |
| Digit-prefix (must contain `^`) | `10^3/µL`, `10^9/L`, `x10^3/uL` |

Bare numbers immediately after the value are **not** captured as a unit,
preventing false positives from reference ranges printed nearby.

## Image Preprocessing (`utils.py`)

| Function | Description |
|----------|-------------|
| `to_grayscale(image)` | Convert BGR image to grayscale |
| `apply_threshold(image, method="otsu"\|"adaptive")` | Binarise image |
| `denoise(image, h=10)` | Non-local-means noise removal |
| `preprocess(image, method="otsu")` | Full pipeline: denoise → grayscale → threshold |

All functions accept a file path (`str`/`Path`) **or** a NumPy array so
they can be chained.

## Dependencies

| Package | Purpose |
|---------|---------|
| `paddleocr` | OCR engine |
| `paddlepaddle` | PaddlePaddle deep-learning framework |
| `opencv-python-headless` | Image preprocessing |
| `numpy` | Array operations |
| `Pillow` | Image I/O |

Install everything with:

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/test_ocr.py -v
```

