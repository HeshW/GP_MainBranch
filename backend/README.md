# GP Medical Analysis - HTTP API

FastAPI layer over `manager.ChatManager` for OCR and AI-first diagnosis fusion.

Therapy generation is feature-flagged and disabled by default (`ENABLE_THERAPY=false`).
When disabled, pipeline responses keep the `therapy` key with a compatibility placeholder payload.

## Run (from repository root)

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/api/docs` for Swagger UI.

## LLM Provider Configuration

The backend supports provider selection using environment variables:

- `LLM_PROVIDER=gemini|openrouter`
- `LLM_API_KEY=...`
- `LLM_MODEL_NAME=...`

OpenRouter-specific optional fields:

- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_SITE_URL=...`
- `OPENROUTER_APP_NAME=GP Medical Analysis`

Legacy `GEMINI_API_KEY` and `GEMINI_MODEL_NAME` remain supported for backward compatibility.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness |
| GET | `/api/v1/meta` | Version and AI configuration flags |
| POST | `/api/v1/pipeline/labs` | JSON body: `labs`, optional `symptoms` |
| POST | `/api/v1/pipeline/image` | Multipart form: `file` (report image) |
| POST | `/api/v1/pipeline/symptoms` | JSON: `text`, optional `use_symptom_parser` |

The React app proxies `/api` to this server in development.

## Evaluation For Discussion

Build the no-retrain graduation discussion pack from the saved classifier and RAG artifacts:

```bash
python backend/scripts/build_discussion_evaluation.py
```

Outputs are written to `data/evaluation/archive/discussion/`, including a Markdown summary,
classifier metrics from saved test predictions, RAG artifact inventory, and a
retraining recommendation.

Generate realistic free-text chat cases before running end-to-end evaluation:

```bash
python backend/scripts/generate_real_chat_cases.py
python backend/scripts/evaluate_pipeline_end_to_end.py --cases data/evaluation/real_chat_cases.json --output data/evaluation/archive/real_chat/real_chat_summary.json --use-rag --use-finetuned-classifier --disable-llm-synthesis --faiss-index-dir backend/artifacts/artifacts/faiss_data_natural --clinicalbert-model-dir backend/artifacts/artifacts/clinicalbert_classifier_natural --finetuned-model-dir backend/artifacts/artifacts/clinicalbert_classifier_natural
```

For full RAG leave-one-out retrieval metrics, use an environment with `faiss-cpu`
installed:

```bash
python backend/scripts/evaluate_rag_confusion.py --index-dir backend/artifacts/artifacts/faiss_data_natural --output-dir data/evaluation/archive/rag_natural
```

Final reproducible RAG diagnostics are kept in `data/evaluation/rag_diagnostics/`.

Retrain with the notebooks only if new data/labels are added or these evaluation
metrics are below the target needed for the project discussion.
