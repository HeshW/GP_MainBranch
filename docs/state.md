# GP Project State (moved to /docs)

Last updated: 2026-03-16

This file was moved from the repository root into `docs/` for clarity and now documents the current verified state and next steps.

Environment
- Active project virtual environment: `.venv` (Python 3.11.9)
- Dependency manifests: `requirements-runtime.txt`, `requirements-test.txt`, `requirements.txt` (delegates)

Project status summary (2026-03-16)
- Focus: Dataset validation and OCR pipeline verification.
- Unit tests: `pytest` run locally → `85 passed, 101 warnings` (Paddle/proto deprecation warnings non-blocking).

Key implemented items
- OCR pipeline
  - `OCREngine` (models/ocr/engine.py): multi-pass parsing, `raw_ocr` output, per-lab `confidence`, `warnings` aggregation.
  - `fields.py`: header/section extraction utilities.
  - `patterns.py` + `synonyms_v15.json`: lab label patterns and synonym mappings.
  - Smoke utilities: `models/ocr/scripts/_print_raw_text_plus_fields.py` and `scripts/run_ocr_smoke.py`.

- Synthetic dataset
  - `models/ocr/scripts/scripts_generate_synthetic_reports.py` updated to produce OCR-ready samples with `expected_labs` ground truth.
  - `data/ocrdata/` generated locally (20 samples) for OCR validation (not committed to remote).

VCS & CI notes
- `.gitignore` updated to ignore `data/kaggleocrset/` and `data/ocrdata/`.
- Sensitive dataset files that were previously tracked were purged from git history and removed from remote.
- GitHub Actions workflows were added, iterated, and then removed temporarily to resolve editor diagnostics; workspace now suppresses stale diagnostics for `.github/workflows/**`.

Verification
- Ran smoke extraction on `data/labreport1test.png` → `fields` and `sections` parsed; `labs_count` was 0.
- Ran smoke tests on selected images from `data/kaggleocrset/` → sample run passed.

Known constraints & recommendations
- Use non-conda Python 3.11 `.venv` to avoid ABI issues.
- Install runtime pins before running heavy OCR scripts:
  ```bash
  source .venv/Scripts/activate
  python -m pip install -r requirements-runtime.txt
  python -m pip install --no-deps paddleocr==2.7.0.3
  ```
- Guard heavy/integration OCR tests with `RUN_OCR_INTEGRATION=1`.

Next immediate actions
1. Add integration test for `data/ocrdata/` vs `expected_labs` (opt-in).
2. Manually review `models/ocr/synonyms_v15.json` and extend lab synonyms.
3. Add barcode validation CLI and batch JSON output.

If anything was moved or removed inadvertently, restore from git history (contact me for restore commands).
