# GP_MainBranch — One‑Page Roadmap (updated)
Date: 2026-03-16

Purpose
- Short, actionable roadmap covering the full GP project: the four models (OCR, Diagnosis, Therapy, Manager/LLM), plus frontend and backend responsibilities. Designed to hand to developers or the VSCode Copilot agent.

Vision
- End-to-end system that ingests pathology/lab documents, extracts structured data (OCR), reasons about findings (Diagnosis), recommends/records therapies (Therapy), and exposes a human-friendly UI. A Manager (LLM) orchestrates model calls, validation, and final outputs; the backend hosts services and persists state; the frontend provides clinician/analyst workflows.

High-level components
- Model A — OCR Engine
  - Responsibilities: image ingestion, layout-aware OCR, header/section extraction, lab/value parsing, barcode decode, confidence/warnings.
  - Deliverables: stable OCREngine API, smoke tests, parsing rules, field/section outputs, barcode validation.

- Model B — Diagnosis
  - Responsibilities: consume structured findings (labs, narrative), apply clinical rules/ML models to generate diagnoses and differential considerations.
  - Deliverables: rule set + lightweight model or classifier, explainable outputs, confidence scores, unit tests.

- Model C — Therapy
  - Responsibilities: map diagnoses and patient context to suggested therapy actions / follow-up steps; include constraints and safety checks.
  - Deliverables: therapy suggestion module (rule-driven initially), audit trail, recommended actions with rationale.

- Model D — Manager (LLM / Orchestrator / Backend)
  - Responsibilities: coordinate data flow across models; present a unified reasoning layer (LLM) to synthesize outputs, resolve conflicts, generate human-readable reports; provide API endpoints; handle persistence, queuing, job orchestration, authentication.
  - Deliverables: service API, orchestration logic, LLM prompts + safety guards, logs/auditing, background job processing.

- Frontend
  - Responsibilities: UI for uploading documents, reviewing extracted fields, manual corrections, viewing diagnosis & therapy recommendations, and exporting reports.
  - Deliverables: minimal React or similar SPA, upload/review workflows, edit/save corrections, basic user/session management.

- Backend (service layer)
  - Relationship to Manager: the Manager (Model D) implements high-level orchestration and decision logic; the Backend implements the runtime services:
    - REST/gRPC API, authentication, data storage, job queue, model-serving endpoints, monitoring.
    - Note: treat Manager as the orchestration layer running on top of the Backend; Backend is the runtime/infrastructure for Manager and models.

Milestones (priority order, short timeline suggested)
- M1 — Stabilize OCR ingestion (1–2 weeks)
  - Create clean Python 3.11 environment and pinned deps.
  - Implement header/section extraction (fields.py) and integrate into OCREngine.
  - Add smoke tests and scripts; ensure reproducible outputs for sample images.
  - Deliverable: OCREngine returns {raw_text, fields, sections, labs, warnings}.

- M2 — Parsing & lab extraction tuning (1 week)
  - Tune LAB_PATTERNS, add raw_ocr with bboxes, add confidence-based warnings.
  - Add unit tests and fixture images.
  - Deliverable: reliable structured lab extraction (high precision on fixtures).

- M3 — Barcode & metadata validation + CLI (1 week)
  - Add barcode decoding, cross-validate VCode, CLI to batch-process images, and output JSON.
  - Deliverable: `scripts/run_ocr_smoke.py` and per-image JSON outputs.

- M4 — Diagnosis model scaffold (2–3 weeks)
  - Implement rule-based diagnosis engine + small ML prototype (if data available).
  - Define input schema and expected outputs (explanations, confidence).
  - Deliverable: diagnosis module with tests and example results.

- M5 — Therapy recommendations scaffold (2 weeks)
  - Implement rule-based mapping from diagnoses to therapy suggestions with safety checks and audit trail.
  - Deliverable: therapy module with sample mappings and tests.

- M6 — Manager & Backend orchestration (3–4 weeks)
  - Build REST API, job queue, persistence (DB), and orchestrator logic that invokes OCR → Diagnosis → Therapy.
  - Integrate a lightweight LLM (prompt templates + guardrails) to synthesize/format final reports.
  - Deliverable: backend service with endpoints to submit document, check job status, retrieve report.

- M7 — Frontend (2–3 weeks)
  - Basic upload/review UI, edit extracted fields, view final report, and trigger re-run or accept corrections.
  - Deliverable: functional UI that connects to backend endpoints.

- M8 — Integration, CI, and documentation (2 weeks)
  - Add smoke CI job (Python 3.11), tests, and developer docs/handbooks for environment & deployment.
  - Deliverable: reproducible dev environment, CI smoke tests, README and onboarding docs.

Cross-cutting priorities & constraints
- Use non-conda Python 3.11 venvs for reproducibility (we discovered conda-tainted venv issues).
- Pin critical binaries: numpy, opencv, paddlepaddle/paddleocr versions to avoid ABI conflicts.
- Prefer rule-based modules first (Diagnosis/Therapy) for explainability; augment with ML later.
- Keep privacy in mind: redact sensitive fields in logs/fixtures for public repos.

Risks & mitigations
- Dependency/ABI conflicts (NumPy/OpenCV/Paddle): mitigate with pinned deps and non-conda venvs.
- LLM hallucination in Manager: mitigate with prompt engineering, verification checks (structured outputs), and conservative wording.
- Data scarcity for ML diagnosis/therapy: start rule-based and collect labeled data for later model training.

Next immediate actions (for today / next session)
1. Finalize OCREngine changes (fields.py, engine integration, smoke script).
2. Add unit smoke test and fixture(s) (sanitised).
3. Start Diagnosis module scaffold (interface + basic rules).

Post-verification follow-ups (2026-03-16):

- Verification run completed: `models/ocr/scripts/_print_raw_text_plus_fields.py` executed successfully and produced `fields` and `sections` for the sample image. `logs/ocr_smoke_fields.json` contains the full output; `labs_count` was 0 for this image.

- Updated short-term priorities (next sprint):
    1. Add an integration test that asserts `fields` and `sections` exist for `data/labreport1test.png`.
    2. Manually populate and review `models/ocr/synonyms_v15.json` to cover missing lab synonyms; iterate `LAB_PATTERNS` afterwards.
    3. Add optional `raw_ocr` (bounding boxes + confidences) to `OCREngine.extract()` output.
      - Status: implemented (2026-03-16) — `raw_ocr` added and smoke script updated to report `raw_ocr_count`.
      - Tests: added `tests/test_ocr_raw_ocr.py` verifying structure; unit test executed successfully in the project venv.
    4. Add CI guard (`RUN_OCR_INTEGRATION=1`) to keep heavy OCR runs opt-in.

These items are high priority to move from smoke validation to repeatable integration tests and to improve lab detection coverage.
Verification note (2026-03-16):

- A quick verification run was attempted using `models/ocr/scripts/_print_raw_text_plus_fields.py` but the run failed at import time due to a missing runtime dependency (`numpy`).
- Recommendation: ensure the project's runtime profile is installed into the active `.venv` before running integration smoke scripts. Example:

```bash
source .venv/Scripts/activate
python -m pip install -r requirements-runtime.txt
```

Retry the smoke script after installing dependencies to confirm that `OCREngine.extract()` returns the expected keys: `raw_text`, `fields`, `sections`, `labs`.

Recent repository changes (2026-03-16):
- `models/ocr/engine.py` — added `_collect_raw_ocr()` and now includes `raw_ocr` in the `OCREngine.extract()` output.
- `models/ocr/fields.py` — header/section extraction utilities used by the parsing pipeline.
- `models/ocr/patterns.py` — updated to optionally merge `models/ocr/synonyms_v15.json` into canonical lab mappings.
- `models/ocr/synonyms_v15.json` — template for manual population of lab synonyms/aliases.
- `models/ocr/scripts/_print_raw_text_plus_fields.py` — smoke script updated to report `raw_ocr_count` alongside `raw_text`, `fields`, and `sections`.
- `tests/test_ocr_raw_ocr.py` — unit test added; monkeypatches the OCR backend and asserts the `raw_ocr` structure (list of {text, bbox, confidence}).
- `logs/ocr_smoke_fields.json` — smoke run output (for `data/labreport1test.png`) captured and saved during verification runs.

Delta updates (2026-03-16):
- `models/ocr/synonyms_v15.json` — populated with an initial curated alias set (glucose, hemoglobin/Hgb, WBC, RBC, platelets, creatinine, BUN/urea, electrolytes, AST/ALT, ALP, lipids).
- `models/ocr/engine.py` — added per-lab confidence aggregation derived from `raw_ocr` and emitted low/critical confidence warnings; labs now include a numeric `confidence` field.
- `tests/test_synonyms.py` — new unit test asserting curated aliases are present in the runtime `SYNONYM_MAP`.
- `tests/test_lab_confidence.py` — new unit test (monkeypatched PaddleOCR) verifying per-lab `confidence` attachment.

If you want, I can generate:
- A one-page milestones checklist as a kanban-style task list (To Do / In Progress / Done),
- Or a minimal GitHub Projects board outline (columns and sample cards) to paste into your VSCode Copilot chat.

Which of those would you like next?

Final verification update (2026-03-16):

- Environment was rebuilt and stabilized in `.venv` with pinned binary/runtime versions.
- Dependency manifests were aligned to prevent future NumPy/OpenCV drift:
  - `requirements.txt` now delegates to runtime + test requirement files.
  - `requirements-test.txt` no longer upgrades runtime binary packages.
- Full pytest run completed successfully after rebuild:
  - `85 passed, 101 warnings`.
- OCR finalization status:
  - `raw_ocr` output, synonym enrichment, and per-lab confidence aggregation are implemented and covered by unit tests.

Synthetic OCR dataset validation update (2026-03-16):

- Completed: synthetic sample generation pipeline hardened for OCR validation:
  - `models/ocr/scripts/scripts_generate_synthetic_reports.py` now emits a deterministic `Lab Results` block and annotation-level `expected_labs` ground truth.
  - Rendering reliability improved via line-by-line lab drawing, larger default fonts, and Pillow API compatibility helpers.
- Completed: generated dataset at `data/ocrdata/` (20 image/annotation pairs) for repeatable OCR checks.
- Completed: random verification on 4 generated samples using `OCREngine.extract()` versus `expected_labs` passed.
- Completed: full project regression run after generator updates passed (`85 passed, 101 warnings`).

Near-term follow-up:
1. Add an opt-in integration test (guarded by `RUN_OCR_INTEGRATION=1`) that samples generated files in `data/ocrdata/` and validates extracted labs against `expected_labs`.
2. Keep synthetic generator defaults conservative for OCR validation (`--augment 0`) and use stronger augmentation only for stress testing.