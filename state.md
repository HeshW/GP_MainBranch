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
  - `requirements.txt`: broad/default project dependencies
  - `requirements-test.txt`: lightweight test dependencies (no Paddle runtime)
  - `requirements-runtime.txt`: pinned OCR runtime profile based on prior validated handoff
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
    - `scripts/check_env_mismatch.py`
  - Installed and validated fresh `.venv` environment end-to-end:
    - `pytest tests/test_ocr.py -q --disable-warnings` -> `82 passed`
    - Runtime import check passed for Python 3.11.9, `numpy 1.23.5`, `cv2 4.6.0`, `paddle 2.6.2`
    - PaddleOCR initialized successfully after installing runtime extras
    - OCR smoke run on `data/labreport1test.png` completed (`result keys: labs/raw_text/warnings`)
  - Added OCR smoke runner:
    - `scripts/run_ocr_smoke.py`
  - Moved runtime logs into `logs/` and removed redundant helper `scripts/_print_raw_text.py`.
- OCR logic currently implemented:
  - Multi-pass lab extraction strategy in `models/ocr/engine.py`:
    1. Full normalized-text pass
    2. Line-by-line pass
    3. Cross-line fallback (label line + value-only next line)
  - Duplicate handling: last match wins with warning.
  - Unit capture includes forms like `%`, `mg/dL`, `10^3/µL`, `x10^3/uL`.
  - Output includes `labs`, `raw_text`, and `warnings`, with `source_match` per lab item.

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

# Mismatch check
python scripts/check_env_mismatch.py --requirements requirements-runtime.txt
```
