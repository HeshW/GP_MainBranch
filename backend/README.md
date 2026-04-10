# GP Medical Analysis - HTTP API

FastAPI layer over `manager.ChatManager` for OCR, AI-first diagnosis fusion, and therapy generation.

## Run (from repository root)

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/api/docs` for Swagger UI.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness |
| GET | `/api/v1/meta` | Version and AI configuration flags |
| POST | `/api/v1/pipeline/labs` | JSON body: `labs`, optional `symptoms` |
| POST | `/api/v1/pipeline/image` | Multipart form: `file` (report image) |
| POST | `/api/v1/pipeline/symptoms` | JSON: `text`, optional `use_symptom_parser` |

The React app proxies `/api` to this server in development.
