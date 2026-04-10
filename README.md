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
