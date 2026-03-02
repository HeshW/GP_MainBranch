# OCR Engine – Model A

PaddleOCR-based text extraction and regex-driven lab-value parsing for
medical report images.

## Module Layout

```
models/ocr/
├── __init__.py   – package exports (OCREngine, extract_from_text)
├── engine.py     – OCREngine class and extract_from_text() helper
├── patterns.py   – regex patterns and synonym map
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
| `labs` | `dict` | Canonical lab name → `{"value": float, "unit": str \| None}` |
| `raw_text` | `str` | Full concatenated OCR text |
| `warnings` | `list[str]` | Parsing issues (empty if everything is clean) |

### Text-only parsing (no OCR)

```python
from models.ocr import extract_from_text

result = extract_from_text("Glucose: 95 mg/dL\nHgb: 13.5 g/dL\nIron: 75 µg/dL")
```

Useful for testing or when OCR is handled externally.

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
