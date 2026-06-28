# Project: OCR & EOG Diagnosis Tool
# Status: Initializing Agent Context (updated based on codebase analysis)

## Environment & Stack
- **Python Version (venv):** 3.11.9 (Strict)
- **Key Folders:**
    - "/models"
    - "/manager"
    - "/tests"
    - "/docs"
    - "/scripts"
    - "/.vscode"
    - "/static"
    - "/README.md"
- **Core Dependencies (from requirements-runtime.txt):**
    - numpy==1.23.5
    - scipy==1.15.3
    - opencv-python==4.6.0.66
    - opencv-contrib-python==4.6.0.66
    - scikit-image==0.20.0
    - paddlepaddle==2.6.2
    - Pillow>=10.0.0
    - requests>=2.31.0
    - PyYAML>=6.0
    - shapely>=2.0.0
    - pyclipper>=1.3.0
    - attrdict>=2.0.1
    - beautifulsoup4>=4.12.0
    - fire>=0.7.0
    - fonttools>=4.24.0
    - imgaug==0.4.0
    - lmdb>=1.4.0
    - lxml>=4.9.0
    - openpyxl>=3.1.0
    - premailer>=3.10.0
    - python-docx>=1.1.0
    - rapidfuzz>=3.0.0
    - tqdm>=4.65.0
- **Test Dependencies (from requirements-test.txt):**
    - pytest>=8.1.1

## Development Rules
1. **Version Control:** We are locked to Python 3.11.9. Do not use Python 3.12+ syntax (like PEP 695 type aliases).
2. **Virtual Environments:** Use non-conda Python 3.11 `.venv`. Keep virtual environments out of the repository (listed in `.gitignore`).
3. **Testing:** `pytest` is used for unit tests. Heavy OCR integration tests should be guarded with `RUN_OCR_INTEGRATION=1`.
4. **Code Hygiene:** Heavy ML/LLM dependencies are lazy-imported. Keep PHI out of committed fixtures; use `logs/` for temporary outputs and never commit raw PHI.
5. **Installation Note:** PaddleOCR might require installation with `--no-deps` on Windows for binary compatibility (see `requirements-runtime.txt` and `docs/ENV_SETUP.md`).