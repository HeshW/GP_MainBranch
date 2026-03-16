# GP_MainBranch — Roadmap
Date: 2026-03-16

Overview
- Purpose: End-to-end system to ingest pathology/lab documents, extract structured data (OCR), reason about findings (Diagnosis), recommend therapies (Therapy), and expose a human-friendly UI. A Manager (LLM) orchestrates model calls, validation, and outputs.

High-level components
- Model A — OCR Engine: image ingestion, layout-aware OCR, header/section extraction, lab/value parsing, barcode decode, confidence/warnings.
- Model B — Diagnosis: rule-based scaffold and ML prototype for generating diagnoses and differentials.
- Model C — Therapy: map diagnoses + context to therapy suggestions with safety checks and audit trail.
- Model D — Manager (LLM / Orchestrator): coordinates models, persistence, APIs, and report synthesis.
- Frontend & Backend: minimal UI for review + backend services (API, auth, job queue, persistence).

Milestones (current focus)
- M1 — OCR stabilization (COMPLETE): core OCR parsing, lab extraction, `OCREngine.extract()` shape stabilised.
- M2 — Parsing & lab extraction (COMPLETE): pattern/synonym work and confidence aggregation.
- M3 — CLI & metadata validation (IN PROGRESS): batch CLI, barcode checks and structured JSON output.
- M4 — Diagnosis (IN PROGRESS → PARTIAL): rule-based `DiagnosisEngine` implemented, unit tests added; next: broaden test coverage and config externalisation.
- M5 — Therapy (PLANNED): scaffold therapy mappings and safety checks.
- M6 — Manager orchestrator (PLANNED): define adapter contract, APIs, and orchestration flows.
- M7 — Integration & CI (PLANNED): opt-in heavy integration jobs (FAISS/LLM) and separation of lightweight CI.

Recent progress (delta — 2026-03-16)
- Synthetic generator hardened: deterministic `expected_labs`, improved font sizing, and dataset generation (`data/ocrdata/`).
- `OCREngine.extract()` now returns `{raw_text, fields, sections, labs, warnings, raw_ocr}` and includes per-lab `confidence` values.
- Smoke scripts and tests added: `models/ocr/scripts/_print_raw_text_plus_fields.py`, `scripts/run_ocr_smoke.py`, and unit tests covering raw_ocr and synonyms.
- Verified: smoke run against `data/labreport1test.png` produced fields/sections and `labs_count: 0` (no lab values in that image).
- Repository hygiene: large dataset directories were added to `.gitignore`; tracked sensitive dataset files were purged from history.

Additional recent progress (delta — 2026-03-16, continued)
- Diagnosis refactor: prototype notebook refactored into `models/diagnosis/diagnosisengine.py` implementing a lightweight rule-based engine and optional RAG components (lazy-loaded).  The module exposes a `DiagnosisEngine` class and a convenience `diagnose()` function.
- Manager adapter: `manager/diagnosis_adapter.py` was added to map `OCREngine.extract()` output into the diagnosis engine and provide a small CLI for image-driven diagnosis.
- Tests & validation: unit tests for the diagnosis engine were added at `tests/test_diagnosis_engine.py` and the full test suite passed locally (`89 passed, 101 warnings`).
- Input validation & docs: `DiagnosisEngine.diagnose()` now validates `report` and `report['labs']` types; short usage notes added at `docs/DIAGNOSIS_ENGINE.md`.
- Editor ergonomics: heavy ML/LLM dependencies (torch/transformers/faiss/google-generativeai) are now lazy-imported to avoid forcing installs during lightweight use and to reduce editor/linter diagnostics.

Cross-cutting notes
- Use non-conda Python 3.11 `.venv` with pinned runtime deps (see `docs/ENV_SETUP.md`).
- Guard heavy OCR integration tests with `RUN_OCR_INTEGRATION=1` to avoid CI flakiness.
- Keep PHI out of committed fixtures; use `logs/` for temporary outputs and never commit raw PHI.

Next actions (priority)
1. Expand `DiagnosisEngine` unit tests (rule boundaries, unit/scale normalization, malformed inputs).
2. Input normalisation: canonicalise units and numeric scales in `report['labs']`.
3. Externalise `_RULES` into a config (YAML/JSON) + add loader and schema validation.
4. Add CLI barcode validation and batch JSON output.
5. Prepare opt-in RAG integration (FAISS index build + gated CI job).
