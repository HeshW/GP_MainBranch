# GP Project State

Last updated: 2026-03-24

Environment
- Active project virtual environment: `.venv` (Python 3.11.9)
- Dependency manifests: `requirements-runtime.txt`, `requirements-test.txt`, `requirements.txt` (delegates)

Project status summary (2026-03-24)
- Focus: Manager-level orchestration and integrated testing across OCR + Diagnosis, with manual labs and symptom input support.
- Unit tests: `pytest` run locally → 98+ passed across OCR, diagnosis, and manager tests; no failures in current branch.

Recent engineering updates (2026-03-24)
- `manager/chat_manager.py` implemented (run_ocr, _build_report, run_diagnosis, run_pipeline, run_chat prototype).
- `manager/diagnosis_adapter.py` updated to delegate to `ChatManager` and preserve legacy model function signatures.
- `manager/manager_tester.py` implemented interactive and one-shot CLI with args for image/labs/symptoms/RAG.
- `tests/test_manager.py` and `tests/test_manager_tester.py` added and validated.
- RAG support paths tested for default disable and parameter handling (see `test_rag_disabled_by_default`).
- Manual input support added (symptoms + labs merge path) and tested (`test_manual_input_only_path`).
- README updated with manager tester quickstart.

Key implemented items (complete)
- OCR layer: `models/ocr` pipeline, synonyms, section extraction, and smoke test scripts.
- Diagnosis layer: `models/diagnosis/diagnosisengine.py` with full rule set and opt-in RAG path.
- Orchestration layer: `manager/chat_manager.py` and adapter/test harness.
- Tests: expanded integration + unit tests, stable execution with all tests passing.

VCS & data status
- `.gitignore` ensures `data/kaggleocrset/` and `data/ocrdata/` are not committed; `.idea` local workspace metadata is present but can be ignored if undesired.
- Example dataset: smoke-test harness uses `data/kaggleocrset`; test results saved in `test_results_kaggleocrset.json`.
- No FAISS index or Gemini credentials in repo; RAG path is environment-optional.

Open items and blockers
- Free-text symptom parser/validator implemented and reviewed (complete).
- FastAPI/web UI not implemented.
- Therapy module integration pending.
- RAG index data must still be provided externally for full RAG path validation.

Next immediate actions
1. Build FastAPI service (`api/app.py`) with endpoints for /v1/pipeline and /v1/symptoms.
2. Add UI workflow for symptom review, manual lab correction, and approval.
3. Add therapy suggestion state in `models/therapy` (and `ChatManager.run_therapy`).
4. Add CI checks for RAG-path optional dependencies and dataset availability gating.
5. Add schema-based request validation with Pydantic models.

Quick status
- All required manager implementation tasks (PR#1–PR#7) are complete and code pushed.
- Current blocker: one downstream feature (free-text symptom NLP) pending implementation.



