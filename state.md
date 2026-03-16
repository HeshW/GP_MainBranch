# GP Project State

Last updated: 2026-03-16

## Environment

- Installed interpreters detected via `py -0p`:
  - Python 3.14 (global default)
  - Python 3.12
  - Python 3.11
- Active project virtual environment:
  - `.venv` exists and was created from Python 3.11.9
  - `.venv/pyvenv.cfg` shows:
    - `version = 3.11.9`
    - `home = C:\Users\Hesh\AppData\Local\Python\pythoncore-3.11-64`
  - This is a non-conda base (good for avoiding prior Anaconda ABI path issues)
- Legacy environments:
  - `.venv`, `.venv311`, and `.venv311_clean` were archived under `.old_venvs/`
- Dependency profiles now in repo:
  - `requirements.txt`: unified entrypoint (delegates to runtime + test requirements)
  - `requirements-test.txt`: test-only dependencies without runtime binary overrides
  - `requirements-runtime.txt`: pinned OCR runtime/inference profile
    - Note: install `paddleocr==2.7.0.3` separately with `--no-deps`

## Project Architecture

- Repository root: `GP_MainBranch`
- Main modules:
  - `models/ocr/`: OCR pipeline (Model A)
    - `engine.py`: `OCREngine` and text extraction/parsing orchestration
    - `patterns.py`: synonym map and regex patterns for lab value extraction
    - `utils.py`: preprocessing utilities (OpenCV)
  - `models/diagnosis/`: placeholder (Model B)
  - `models/therapy/`: placeholder (Model C)
  - `manager/`: orchestrator package placeholder (Model D)
- Effective current entry points:
  - `from models.ocr import OCREngine`
  - `from models.ocr import extract_from_text`
- Key data files:
  - `data/labreport1test.png` (primary OCR test image currently visible)
- Tests:
  - `tests/test_ocr.py` (regex/parser-focused unit tests)

## Current State

- Current focus: stabilizing environment and dependency compatibility for OCR runtime.
- Completed in this session:
  - Consolidated old virtual environments into `.old_venvs/`.
  - Recreated clean `.venv` using Python 3.11.9.
  - Added dependency split files:
    - `requirements-test.txt`
    - `requirements-runtime.txt`
  - Added mismatch diagnostic script:
    - `models/ocr/scripts/check_env_mismatch.py`
  - Installed and validated fresh `.venv` environment end-to-end:
    - `pytest tests/test_ocr.py -q --disable-warnings` -> `82 passed`
    - Runtime import check passed for Python 3.11.9, `numpy 1.23.5`, `cv2 4.6.0`, `paddle 2.6.2`
    - PaddleOCR initialized successfully after installing runtime extras
    - OCR smoke run on `data/labreport1test.png` completed (`result keys: labs/raw_text/warnings`)
  - Added OCR smoke runner:
    - `models/ocr/scripts/run_ocr_smoke.py`
  - Moved runtime logs into `logs/` and removed redundant helper `scripts/_print_raw_text.py`.
  - Moved OCR helper scripts into `models/ocr/scripts/` and added editable `models/ocr/synonyms_v15.json` template.
  - Populated `models/ocr/synonyms_v15.json` with an initial curated alias set to improve lab coverage (glucose, hemoglobin/Hgb, WBC, RBC, platelets, creatinine, BUN/urea, electrolytes, AST/ALT, ALP, lipids).
- OCR logic currently implemented:
  - Multi-pass lab extraction strategy in `models/ocr/engine.py`:
    1. Full normalized-text pass
    2. Line-by-line pass
    3. Cross-line fallback (label line + value-only next line)
  - Duplicate handling: last match wins with warning.
  - Unit capture includes forms like `%`, `mg/dL`, `10^3/µL`, `x10^3/uL`.
  - Output includes `labs`, `raw_text`, and `warnings`, with `source_match` per lab item.
  - `labs` entries now include a `confidence` float derived from `raw_ocr` when available; low/critical confidence values append warnings to `warnings`.

## Known Constraints

- Python/runtime compatibility:
  - Global Python is 3.14, but project should run in `.venv` on Python 3.11.9.
  - Historical Anaconda-based venvs caused ABI/path issues; must keep non-conda venv.
- OCR dependency constraints:
  - Paddle stack can be version-sensitive on Windows.
  - Prior validated profile uses:
    - `paddlepaddle==2.6.2`
    - `paddleocr==2.7.0.3` (commonly installed with `--no-deps` in this project context)
    - `numpy==1.23.5`
    - `opencv-python==4.6.0.66`
- Data/input constraints:
  - Input expected as report images (`.png`, etc.) for OCR.
  - Parser expects lab-like lines/synonyms and numeric value formats with optional units.
- Hardware/acceleration:
  - No GPU requirement is currently assumed; default flow targets CPU compatibility unless explicitly configured otherwise.

## Universal Rules

- **Commits:** Keep all commits local until you explicitly request a push to remote.
- **State & Roadmap:** After completing any task, update `state.md` and `roadmap.md` with a short verification note and commit locally.
- **Session Memory:** Append a short note to `/memories/session/ocr_finalize_todos.md` for traceability between sessions.
- **Logs & Artifacts:** Store runtime logs and temporary outputs under `logs/`; never commit large model files or raw fixtures containing PHI.
- **Environment:** Use a non-conda Python 3.11 venv and the pinned runtime deps (`numpy`, `opencv-python`, `paddlepaddle`, `paddleocr`) documented in `requirements-*`.
- **Integration Tests:** Guard heavy/integration tests behind `RUN_OCR_INTEGRATION=1` (or equivalent) to avoid CI flakiness.
- **Synonyms & Ingestion:** Treat `GPProject_OC_Version15.txt` as guidance only; only apply synonyms to `models/ocr/synonyms_v15.json` after manual review.
- **Data Privacy:** Sanitize test fixtures to remove PHI before adding to the repo; redact logs when sharing externally.

## Next Steps Checklist

- [x] Activate new `.venv` and upgrade base tooling (`pip`, `setuptools`, `wheel`).
- [x] Install and validate `requirements-test.txt`.
- [x] Run `pytest tests/test_ocr.py` in `.venv` and confirm baseline green.
- [x] Install pinned OCR runtime profile from `requirements-runtime.txt`.
- [ ] Run quick runtime checks:
  - [x] `python -c "import numpy, cv2; ..."`
  - [x] `from paddleocr import PaddleOCR` init
  - [x] `from models.ocr import OCREngine` + `engine.extract(...)`
- [x] Run `scripts/check_env_mismatch.py --requirements requirements-runtime.txt` and resolve any missing/mismatch entries.
- [x] If Paddle issues persist, install only missing runtime extras incrementally and rerun checks.
- [ ] Once stable, optionally remove `.old_venvs/` to reclaim disk space.

## Post-verification status & next actions (2026-03-16)

- Verification run: executed `models/ocr/scripts/_print_raw_text_plus_fields.py` against `data/labreport1test.png` using the project's `.venv`.
  - Outcome: successful OCR + parsing. See `logs/ocr_smoke_fields.json` for full JSON output.
  - Observations: `fields` and `sections` are extracted correctly; `labs_count` is 0 for this image (no lab-value matches found).

- Immediate next actions:
  1. Add a small integration test that asserts `fields` and `sections` keys exist for the sample image (tests/test_ocr_integration.py).
  2. Manually review and populate `models/ocr/synonyms_v15.json` to improve lab label coverage before automated ingestion.
  3. Extend the engine to optionally return `raw_ocr` (bboxes + confidences) so downstream confidence-driven filters can run.
  4. Add CI guard `RUN_OCR_INTEGRATION=1` to avoid running heavy OCR in standard CI runs.

- How to re-run verification locally (quick):

```bash
# Activate venv (Git Bash)
source .venv/Scripts/activate

# (recommended) install full runtime pins if not already
python -m pip install -r requirements-runtime.txt

# Run smoke script and capture output
python -m models.ocr.scripts._print_raw_text_plus_fields data/labreport1test.png --raw > logs/ocr_smoke_fields.json 2>&1

# Inspect results
sed -n '1,200p' logs/ocr_smoke_fields.json
```

Add a short test and a quick PR for `synonyms_v15.json` updates once you approve the manual alias mappings.

### Raw OCR added (2026-03-16)

- Status: implemented — `OCREngine.extract()` now includes a `raw_ocr` list of items `{"text", "bbox", "confidence"}` when PaddleOCR provides them.
- Tests: Added `tests/test_ocr_raw_ocr.py` (monkeypatched PaddleOCR) which verifies `raw_ocr` structure; the test completed successfully in the project `.venv`.
- Smoke: `models/ocr/scripts/_print_raw_text_plus_fields.py` now reports `raw_ocr_count` for quick verification.
 - Tests (delta): Added `tests/test_synonyms.py` (verifies curated synonyms merged into `SYNONYM_MAP`) and `tests/test_lab_confidence.py` (monkeypatched PaddleOCR verifies per-lab `confidence` aggregation).

Next verification steps: run the unit test suite and, if desired, an integration smoke run against `data/labreport1test.png` once runtime deps are installed.

## Useful Commands

```bash
# Activate venv (Git Bash)
source .venv/Scripts/activate

# Activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# Test profile install
python -m pip install -r requirements-test.txt

# Runtime pinned profile install
python -m pip install -r requirements-runtime.txt
python -m pip install --no-deps paddleocr==2.7.0.3

# Mismatch check
python scripts/check_env_mismatch.py --requirements requirements-runtime.txt
```

## Finalization Delta (2026-03-16)

- Rebuilt `.venv` from scratch and removed ABI mismatches by re-pinning the binary stack:
  - `numpy==1.23.5`
  - `scipy==1.15.3`
  - `scikit-image==0.20.0`
  - `opencv-python==4.6.0.66`
  - `opencv-contrib-python==4.6.0.66`
- Aligned dependency files to avoid future drift:
  - `requirements.txt` now delegates to `requirements-runtime.txt` + `requirements-test.txt`.
  - `requirements-test.txt` no longer upgrades NumPy/OpenCV.
- Full test suite result after rebuild and OCR finalization:
  - `85 passed, 101 warnings in 2.30s`
- OCR production caveat:
  - `pdf2docx` and `PyMuPDF<1.21.0` are optional PDF-path dependencies and can require heavy native builds on Windows; they are not required for current image-based OCR flows/tests.

## Synthetic Dataset Verification Delta (2026-03-16)

- Updated `models/ocr/scripts/scripts_generate_synthetic_reports.py` to generate OCR-validation-ready lab panels and ground truth labels:
  - Added deterministic `Lab Results` block with parse-friendly lines for `glucose`, `hemoglobin`, `wbc`, and `platelets`.
  - Added per-sample `expected_labs` payload into each annotation JSON.
  - Switched lab panel rendering to line-by-line drawing (instead of wrapped paragraph rendering) to improve OCR reliability.
  - Added Pillow compatibility sizing helper to support environments where `textsize`/`getsize` APIs differ.
  - Improved default font behavior with TrueType fallback candidates and larger default font sizes for clearer OCR text.
- Generated a new synthetic dataset under `data/ocrdata/`:
  - Images: `data/ocrdata/images/report_00001.png` ... `report_00020.png`
  - Annotations: `data/ocrdata/annotations/report_00001.json` ... `report_00020.json`
- Random-sample OCR verification (4 generated samples) passed against `expected_labs` with value tolerance checks.
- Full test suite re-run after generator changes:
  - `85 passed, 101 warnings in 2.12s`

