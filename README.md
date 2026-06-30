# GP_MainBranch

Modular medical-report analysis system with a FastAPI backend, a Next.js frontend, optional RAG/classifier artifacts, and reproducible evaluation scripts.

## Project Structure

```text
backend/
  app/                 FastAPI app, routers, config, auth, and database setup
  manager/             Pipeline orchestration and chat/session support
  models/              OCR, diagnosis, RAG, mental-health, and therapy modules
  scripts/             Evaluation, health-check, ingestion, and reproducibility tools
  tests/               Backend unit and API contract tests
  docs/                Backend evaluation and artifact documentation

frontend_next/         Active Next.js application
notebooks/             Reproducibility notebooks for final model/artifact work
data/
  raw/                 Local raw datasets, ignored except .gitkeep
  processed/           Local processed datasets, ignored except .gitkeep
  evaluation/          Generated evaluation outputs, ignored except .gitkeep
archive/               Legacy code and research notes retained for review
docs/                  Repository-level reports
requirements.txt       Unified Python install entrypoint
```

Large datasets, model weights, FAISS indexes, local databases, virtual environments, Node installs, and generated build outputs are intentionally ignored.

## Quick Start

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Create `backend/.env` from `backend/.env.example`, then start the API:

```bash
uvicorn app.main:app --reload --app-dir backend --port 8000
```

Start the active frontend:

```bash
cd frontend_next
npm install
npm run dev
```

Open `http://localhost:3000`. FastAPI docs are available at `http://127.0.0.1:8000/api/docs`.

## Runtime Notes

- Optional AI modules are safe-off by default in `backend/.env.example`.
- If RAG/classifier flags are enabled but required artifacts are missing, startup degrades gracefully.
- Therapy generation is feature-flagged with `ENABLE_THERAPY=false` by default.
- RAG metadata loading prefers JSON metadata. Use unsafe pickle loading only for trusted local artifacts.
- Service API-key protection can be enabled with `REQUIRE_SERVICE_API_KEY=true` and `SERVICE_API_KEY=...`.

## Development And Evaluation

Run backend tests:

```bash
pytest backend/tests/
```

Run the API integration subset:

```bash
pytest -q backend/tests/test_api_integration.py
```

Run the frontend build:

```bash
cd frontend_next
npm run build
```

Build the discussion/evaluation pack:

```bash
python backend/scripts/build_discussion_evaluation.py
```

Evaluation outputs are generated under `data/evaluation/` and are not committed by default.

## Archive Policy

`archive/` contains files kept for manual review or historical context, not active runtime code. The archived Vite frontend is retained under `archive/frontend-vite/`; the active frontend is `frontend_next/`.

See `docs/REPOSITORY_CLEANUP_REPORT.md` for the latest cleanup classification and validation notes.
