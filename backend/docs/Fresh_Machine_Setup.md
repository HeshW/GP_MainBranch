# Fresh Machine Setup (Windows-focused)

This runbook is the safe path for reproducing a working environment on a new machine.

## 1) Create virtual environment

From repository root:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

## 2) Install pinned dependencies

```bash
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps paddleocr==2.7.0.3
```

Why the extra PaddleOCR command:
- We intentionally avoid PaddleOCR optional PDF extras (`pdf2docx`, `PyMuPDF<1.21.0`) because they can fail to build on Windows/Python 3.11.

## 3) Configure environment file

```bash
copy backend\.env.example backend\.env
```

Set at least:
- `USE_RAG=true` (optional)
- `FAISS_INDEX_DIR=backend/artifacts/artifacts/faiss_data_natural`
- `CLINICALBERT_MODEL_DIR=backend/artifacts/artifacts/clinicalbert_classifier_natural`
- `USE_FINETUNED_CLASSIFIER=true` (optional)
- `FINETUNED_MODEL_DIR=backend/artifacts/artifacts/clinicalbert_classifier_natural`
- `GEMINI_API_KEY=...` (required for chat and full AI responses)

## 4) Start backend from repo root

```bash
uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
```

Important:
- Start from repo root, not from `backend/`.
- Relative paths in `.env` are resolved from the current working directory.

## 5) Start frontend

```bash
cd frontend
npm install
npm run dev
```

Open:
- Frontend: http://127.0.0.1:5173
- API docs: http://127.0.0.1:8000/api/docs
- API meta: http://127.0.0.1:8000/api/v1/meta

## 6) Validate environment quickly

From repo root:

```bash
.venv\Scripts\python backend/scripts/check_env_mismatch.py --requirements requirements.txt
.venv\Scripts\python backend/scripts/check_env_mismatch.py --requirements backend/requirements-ai.txt
.venv\Scripts\python backend/scripts/validate_ai_setup.py
.venv\Scripts\python -m pytest -q
```

## Known traps and how this setup avoids them

1. NumPy/OpenCV ABI break
- Cause: installing newer unpinned `faiss-cpu`/`datasets` may force NumPy 2.x.
- Fix: use pinned files (`requirements-runtime.txt` + `requirements-ai.txt`).

2. Chat returns fallback text only
- Cause: missing/invalid `GEMINI_API_KEY`.
- Fix: set valid key in `backend/.env` and restart backend.

3. RAG/classifier look disabled even when flags are true
- Cause: wrong working directory or stale artifact paths cause relative model/index paths to fail.
- Fix: always run backend from repo root and point `.env` to the actual artifact directories under `backend/artifacts/artifacts/...`.

4. PaddleOCR optional deps fail on fresh Windows box
- Cause: optional PDF conversion stack may require native builds.
- Fix: install `paddleocr` with `--no-deps` as documented above.
