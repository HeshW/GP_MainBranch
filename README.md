# GP_MainBranch

Modular medical-report analysis system.

## Architecture

```
models/
  ocr/        – Model A: OCR-based lab-value extraction (PaddleOCR)
  diagnosis/  – Model B: diagnosis inference (TBD)
  therapy/    – Model C: therapy recommendation (TBD)
manager/      – Model D: orchestrator / manager
tests/        – unit tests
```

## Quick Start

### Run manager tester (console prototype)

- `python -m manager.manager_tester --labs '{"glucose": 145.0, "hemoglobin": 11.0}'`
- `python -m manager.manager_tester --symptoms "fatigue and thirst" --labs '{"glucose": 185.0}'`
- `python -m manager.manager_tester --image path/to/report.png`

(Use `--rag --faiss-index-dir /path --gemini-key $GEMINI_API_KEY` to enable RAG once configured.)

### Installation

```bash
pip install -r requirements.txt
```

> **Note**: `paddlepaddle` requires Python 3.8–3.10.  
> For GPU support replace `paddlepaddle` with `paddlepaddle-gpu`.

### Using the OCR Engine (Model A)

```python
from models.ocr import OCREngine

engine = OCREngine()
result = engine.extract("path/to/lab_report.png")

# Structured lab values
print(result["labs"])
# e.g. {'glucose': {'value': 95.0, 'unit': 'mg/dL'}, 'hemoglobin': ...}

# Full raw text for debugging
print(result["raw_text"])

# Any extraction warnings
print(result["warnings"])
```

If you already have raw OCR text and only need the regex parser:

```python
from models.ocr import extract_from_text

result = extract_from_text("Glucose: 95 mg/dL  Hgb: 13.5 g/dL")
print(result["labs"])
```

### Running Tests

```bash
pytest tests/
```

Tests do **not** require PaddleOCR to be installed; they exercise only the
regex extraction layer.

## OCR Engine Details

See [`models/ocr/README.md`](models/ocr/README.md) for full documentation.
