# GP_MainBranch

Modular medical-report analysis system.

## Architecture

```text
backend/app/                  FastAPI entrypoints, routers, and config
backend/manager/              Orchestration layer split by responsibility
backend/manager/chat_manager.py    Public facade used by API and scripts
backend/manager/chat_support.py    Chat prompt/messages helpers
backend/manager/pipeline_support.py Pipeline report assembly helpers
backend/manager/session_store.py   In-memory chat session storage
backend/models/               OCR, diagnosis, and therapy engines
backend/tests/                Backend test suite

frontend/src/features/        Feature-first UI modules
frontend/src/features/analysis/ Analysis tabs + request hook
frontend/src/features/results/  Result rendering
frontend/src/features/chat/     Chat UI
frontend/src/shared/            Shared API client, layout, hooks, and types
```

## Web application

1. Install API deps: `pip install -r backend/requirements.txt`.
2. Start API from repo root: `uvicorn app.main:app --reload --app-dir backend --port 8000`.
3. Start UI: `cd frontend && npm install && npm run dev`.
4. Open `http://127.0.0.1:5173`.

Notes:
- Frontend dev server is pinned to host `127.0.0.1` and port `5173` in `frontend/vite.config.ts`.
- If port `5173` is occupied, Vite will fail fast (strict port) instead of silently switching ports.
- Optional service-level API protection (no user accounts required):
	- Backend `.env`: set `REQUIRE_SERVICE_API_KEY=true` and `SERVICE_API_KEY=<shared-secret>`.
	- Frontend `.env` (inside `frontend/`): set `VITE_API_KEY=<shared-secret>`.
	- Health/meta endpoints remain accessible without key for diagnostics.
- Optional advanced AI modules are safe-off by default in `backend/.env.example` (`USE_RAG=false`, `USE_FINETUNED_CLASSIFIER=false`).
	- If enabled but required assets are missing, backend startup degrades gracefully and keeps core endpoints available.
- RAG metadata loading is hardened by default:
	- Prefer `metadata_mapping.json` for FAISS metadata.
	- If using `metadata_mapping.pkl`, provide `metadata_mapping.pkl.sha256` for hash verification.
	- Use `ALLOW_UNSAFE_PICKLE_METADATA=true` only for trusted local artifacts.

Swagger docs: `http://127.0.0.1:8000/api/docs`.

## Quick start

### Run manager tester

- `python -m manager.manager_tester --labs '{"glucose": 145.0, "hemoglobin": 11.0}'`
- `python -m manager.manager_tester --symptoms "fatigue and thirst" --labs '{"glucose": 185.0}'`
- `python -m manager.manager_tester --image path/to/report.png`

### Installation

```bash
pip install -r requirements.txt
```

### Run tests

```bash
pytest backend/tests/
```

Frontend smoke test:

```bash
cd frontend && npm run test:smoke
```

Backend API integration subset:

```bash
pytest -q backend/tests/test_api_integration.py
```
