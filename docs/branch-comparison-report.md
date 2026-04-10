# Branch Comparison Report: `master` vs `mego-edits`

**Repository:** `HeshW/GP_MainBranch`  
**Report Date:** 2026-04-10  
**Branches compared:** `origin/master` (HEAD `28cbd4d`) vs `origin/mego-edits` (HEAD `b656b61`)  
**Note:** The two branches share **no common ancestor** — they represent parallel development tracks, not a fork/divergence from a shared commit.

---

## Table of Contents

1. [Side-by-Side Architecture Summary](#1-side-by-side-architecture-summary)  
2. [End-to-End Pipeline Flows](#2-end-to-end-pipeline-flows)  
3. [File-Level Change Inventory](#3-file-level-change-inventory)  
4. [Behavioral Differences](#4-behavioral-differences)  
5. [Data Contracts and Schema Changes](#5-data-contracts-and-schema-changes)  
6. [Dependency and Environment Changes](#6-dependency-and-environment-changes)  
7. [Quality and Risk Evaluation](#7-quality-and-risk-evaluation)  
8. [Actionable Merge Recommendations](#8-actionable-merge-recommendations)

---

## 1. Side-by-Side Architecture Summary

| Dimension | `master` | `mego-edits` |
|---|---|---|
| **Entry point** | CLI / `manager.manager_tester` | FastAPI server (`backend/app/main.py`) + React UI |
| **Project layout** | Flat root: `manager/`, `models/`, `tests/`, `scripts/`, `docs/` | Monorepo with `backend/` sub-package + `frontend/` |
| **Concurrency model** | **Synchronous** Python throughout | **Async/await** throughout backend |
| **Orchestrator** | `manager/chat_manager.py` — single-file, sync | `backend/manager/chat_manager.py` — async, split across `chat_support.py`, `pipeline_support.py`, `session_store.py`, `runtime.py` |
| **Diagnosis engine** | Single file `models/diagnosis/diagnosisengine.py` with inline `_RULES` list (Python dataclasses) | Split into `diagnosisengine.py`, `rules.py`, `synthesis.py`, `text.py`, `rag.py`; clinical rules live in `clinical_rules.yaml` |
| **Diagnosis approach** | Rule-based only (lightweight) + optional RAG (ClinicalBERT + FAISS + Gemini) | **AI-first with rule safety checks**: fine-tuned ClinicalBERT classifier (primary), RAG (support), rule engine (safety gating), Gemini LLM synthesis |
| **Therapy engine** | Placeholder (`models/therapy/__init__.py` — empty) | Fully implemented `TherapyEngine` using Gemini LLM; produces structured Arabic therapy plans |
| **Chat** | Stub response only (`run_chat` returns static string) | Full session-backed chat with rolling history, SSE streaming, Gemini LLM |
| **Web API** | None | FastAPI: `/api/v1/pipeline/labs`, `/image`, `/ocr`, `/symptoms`, `/diagnosis`, `/chat` |
| **Frontend** | None | React 19 + TypeScript + Vite; three analysis panels (Labs, Image, Symptoms) + Result view + Chat |
| **Arabic language support** | None | Arabic → English translation for RAG queries and classifier input |
| **Configuration** | Environment variables (ad-hoc in scripts) | `pydantic-settings`-based `Settings` class loaded from `backend/.env` |
| **LLM provider abstraction** | Direct Gemini calls in diagnosis engine | `BaseModelProvider` / `GeminiProvider` async abstraction in `models/common/ai_provider.py` |
| **Test runner** | `pytest tests/` from repo root | `pytest backend/tests/` via `pytest.ini` at root; `conftest.py` injects `backend/` into `sys.path` |
| **Notebooks** | `models/diagnosis/diagnosisproto.ipynb` only | Full set in `notebooks/`: ClinicalBERT fine-tuning, end-to-end natural pipeline, Colab runbook |

### Architecture Diagram — `master`

```
CLI / scripts
     │
     ▼
manager/
  chat_manager.py          ← sync orchestrator
  ├─ run_ocr()
  ├─ _build_report()
  ├─ run_diagnosis()
  └─ run_pipeline()
         │
         ├──► models/ocr/engine.py  (PaddleOCR → structured labs)
         └──► models/diagnosis/
                  diagnosisengine.py  (inline _RULES + optional RAG)
```

### Architecture Diagram — `mego-edits`

```
React Frontend (Vite)
  features/analysis → POST /api/v1/pipeline/*
  features/chat     → POST /api/v1/chat (SSE stream)
       │
       ▼
FastAPI (backend/app/main.py)
  ├─ /api/v1/health
  ├─ /api/v1/pipeline/labs|image|ocr|symptoms|diagnosis
  └─ /api/v1/chat  (non-streaming + GET /stream SSE)
       │
       ▼
manager/chat_manager.py  ← async facade
  ├─ pipeline_support.build_report()
  ├─ run_ocr_only()
  ├─ run_diagnosis_only()    ─────► models/diagnosis/
  │                                   diagnosisengine.py
  │                                   ├─ rules.py + clinical_rules.yaml
  │                                   ├─ rag.py  (ClinicalBERT + FAISS)
  │                                   ├─ synthesis.py  (Gemini response)
  │                                   └─ text.py  (build_combined_text)
  ├─ run_pipeline()
  ├─ run_from_symptoms()      ─────► symptom_normalizer.py
  ├─ run_chat()               ─────► session_store.py + GeminiProvider
  └─ stream_chat()            ──────────────────┘
         │
         └──► models/therapy/engine.py  ← Gemini therapy plan
                  models/common/ai_provider.py  ← async LLM abstraction
```

---

## 2. End-to-End Pipeline Flows

### 2.1 Image (OCR) path

**`master`**
1. Caller invokes `ChatManager.run_pipeline(image="path")` (sync).
2. `run_ocr()` → `OCREngine.extract()` → PaddleOCR → returns `{labs, raw_text, fields, sections}`.
3. `_build_report()` merges OCR output + any `manual_input`.
4. `run_diagnosis(report)` → `DiagnosisEngine.diagnose(report)` — iterates `_RULES` list, matches lab thresholds.
5. Returns `{status, ocr, diagnosis, therapy: None, warnings, elapsed_ms}`.

**`mego-edits`**
1. Client `POST /api/v1/pipeline/image` (multipart form).
2. Router saves upload to temp file; calls `await manager.run_pipeline(image=tmpfile)`.
3. `prepare_report()` → `pipeline_support.build_report()` — awaits `run_ocr()` → OCR engine.
4. `run_diagnosis_only(report)` → `await DiagnosisEngine.diagnose(report)`.
   - Lab rules evaluated from `clinical_rules.yaml`.
   - Symptom rules evaluated if `symptoms` field present.
   - Optional: RAG query to ClinicalBERT + FAISS (if configured).
   - Optional: Fine-tuned classifier prediction (if configured).
   - Decision fusion: picks highest-confidence source.
   - Optional: Gemini `DiagnosisResponseSynthesizer` produces `gemini_response`.
5. `TherapyEngine.generate_therapy(diagnosis, patient_info)` → Gemini LLM → structured therapy plan.
6. Returns `{status, ocr, diagnosis, therapy, warnings, elapsed_ms}` — temp file cleaned up.
7. React `ResultView` renders findings, therapy plan, and summary.

### 2.2 Manual labs path

**`master`** — `run_pipeline(labs={"glucose": 145})` skips OCR, merges labs dict directly into report.

**`mego-edits`** — `POST /api/v1/pipeline/labs` with JSON `{labs: {...}, symptoms?: "..."}` → same async pipeline; also routes through optional therapy generation.

### 2.3 Symptoms (free text) path

**`master`**
1. `run_from_symptoms(text)` → `parse_symptoms(text)` → `validate_parsed()`.
2. Converts validated output to `manual_input`, calls `run_pipeline(manual_input=...)`.
3. Returns pipeline result plus `parsed`, `validated`, `review_required`.

**`mego-edits`**
1. `POST /api/v1/pipeline/symptoms` with `{text, use_symptom_parser}`.
2. `run_from_symptoms(text)` → `parse_symptoms()` → `validate_parsed()` → **`build_normalized_symptom_text()`** (new).
3. `build_manual_input_from_validated()` — passes `symptom_list` (structured list) AND `raw_text` separately (richer report struct than `master`).
4. Full async pipeline including therapy.
5. Result includes extra `normalized_text` key.

### 2.4 RAG / AI diagnosis path

**`master`** — RAG is opt-in via `use_rag=True`, handled inside the monolithic `DiagnosisEngine`. ClinicalBERT embeds the combined text, FAISS finds top-K cases, Gemini generates narrative.

**`mego-edits`** — Same RAG components but:
- Moved to dedicated `rag.py` module.
- Fine-tuned classifier (`FineTunedDiagnosisClassifier`) can run in parallel with RAG.
- Decision fusion logic (`_build_final_diagnosis`, `_build_decision_fusion`) explicitly selects winning source (classifier > RAG > rules).
- Arabic translation layer wraps both classifier input and RAG queries.
- Gemini synthesis is now a separate `DiagnosisResponseSynthesizer` class returning structured JSON validated by `AIClinicalResponse` Pydantic model.

### 2.5 Chat path

**`master`** — `run_chat()` returns a static stub string with no memory.

**`mego-edits`**
1. `POST /api/v1/chat` or `GET /api/v1/chat/stream?session_id=...&message=...`.
2. `ChatSessionStore.append()` adds message to rolling history (max 12, keeps 10 recent).
3. `build_chat_prompt()` formats Arabic role-tagged history.
4. `GeminiProvider.generate_content()` (non-streaming) or `generate_stream()` (SSE).
5. Model reply stored back in session history.

---

## 3. File-Level Change Inventory

### 3.1 Files only on `master` (deleted / not ported)

| Path | Notes |
|---|---|
| `docs/DIAGNOSIS_ENGINE.md` | Diagnosis engine docs (superseded by `backend/docs/`) |
| `docs/ENV_SETUP.md` | Env setup guide (superseded) |
| `docs/state.md` | Project state tracker |
| `models/therapy/__init__.py` | Empty placeholder module |
| `tests/__init__.py` | Old test package init |

### 3.2 Files only on `mego-edits` (new additions)

| Path | Description |
|---|---|
| `backend/app/` | Full FastAPI application (`main.py`, `config.py`, `deps.py`, routers, schemas) |
| `backend/.env.example` | Template for environment config |
| `backend/faiss_data/medical_cases.index` | **Binary FAISS index committed to git** — large artifact |
| `backend/manager/chat_support.py` | Chat prompt helpers + Arabic system strings |
| `backend/manager/pipeline_support.py` | `build_report()` async helper extracted from ChatManager |
| `backend/manager/runtime.py` | `run_async()` / `sync_from_async()` bridges |
| `backend/manager/session_store.py` | `ChatSessionStore` in-memory rolling history |
| `backend/manager/symptom_normalizer.py` | DDX-style symptom normalizer with 50+ canonical templates |
| `backend/models/common/ai_provider.py` | Async `GeminiProvider` + `BaseModelProvider` ABC |
| `backend/models/diagnosis/clinical_rules.yaml` | Externalised clinical threshold rules |
| `backend/models/diagnosis/rag.py` | `ClinicalBERTEmbedder`, `MedicalCaseSearcher`, `MedicalRAGAssistant`, `FineTunedDiagnosisClassifier`, `ArabicToEnglishTranslator` |
| `backend/models/diagnosis/rules.py` | Rule loading from YAML + `diagnose_from_labs()` + `diagnose_from_symptoms()` |
| `backend/models/diagnosis/synthesis.py` | `DiagnosisResponseSynthesizer` — Gemini JSON synthesis |
| `backend/models/diagnosis/text.py` | `build_combined_text()` extracted helper |
| `backend/models/therapy/engine.py` | Full `TherapyEngine` with Gemini LLM |
| `backend/models/ocr/image_io.py` | OCR image I/O helpers |
| `backend/models/ocr/parsing.py` | OCR parsing helpers |
| `backend/models/ocr/raw_ocr.py` | Raw OCR layer |
| `backend/models/ocr/types.py` | OCR type definitions |
| `backend/scripts/train_clinicalbert_classifier.py` | ClinicalBERT fine-tuning script |
| `backend/scripts/build_ddxplus_natural_csvs.py` | DDX-Plus dataset builder |
| `backend/scripts/evaluate_pipeline_end_to_end.py` | End-to-end evaluation |
| `backend/scripts/evaluate_rag_confusion.py` | RAG confusion matrix evaluation |
| `backend/scripts/rebuild_faiss_from_ddx.py` | FAISS index rebuild script |
| `backend/scripts/validate_ai_setup.py` | AI setup validator |
| `backend/tests/conftest.py` | `sys.path` fixture for tests |
| `backend/tests/test_evaluate_pipeline_metrics.py` | Pipeline metrics tests |
| `backend/tests/test_finetuned_classifier_integration.py` | Classifier integration tests |
| `backend/tests/test_rag.py` | RAG unit tests |
| `backend/tests/test_symptom_normalizer.py` | Normalizer tests |
| `backend/tests/test_therapy_engine.py` | TherapyEngine tests |
| `frontend/` | Full React 19 + TypeScript + Vite frontend |
| `implementation_plan.md` | Professionalisation plan (local planning artifact) |
| `notebooks/` | ClinicalBERT fine-tuning notebook, Colab all-in-one, runbook |
| `pytest.ini` | Points pytest at `backend/tests` |
| `GP_(1) (1).ipynb` | Prototype notebook |

### 3.3 Files renamed / moved (equivalent content, path changed)

| `master` path | `mego-edits` path | Similarity |
|---|---|---|
| `manager/__init__.py` | `backend/manager/__init__.py` | 100% |
| `manager/symptom_parser.py` | `backend/manager/symptom_parser.py` | ~60% |
| `manager/symptom_validator.py` | `backend/manager/symptom_validator.py` | ~98% |
| `models/__init__.py` | `backend/models/__init__.py` | 100% |
| `models/ocr/` (all files) | `backend/models/ocr/` | 100% |
| `requirements-runtime.txt` | `backend/requirements-runtime.txt` | 100% |
| `requirements-test.txt` | `backend/requirements-test.txt` | identical |
| `scripts/check_env_mismatch.py` | `backend/scripts/check_env_mismatch.py` | 100% |
| `scripts/ingest_oc_version15.py` | `backend/scripts/ingest_oc_version15.py` | 100% |
| `scripts/run_all_tests.py` | `backend/scripts/run_all_tests.py` | 100% |
| `scripts/smoke_test_kaggleocrset.py` | `backend/scripts/smoke_test_kaggleocrset.py` | 100% |

### 3.4 Files modified in place (same name, different content)

| File | Key changes |
|---|---|
| `manager/chat_manager.py` | Async rewrite; adds therapy, streaming, session store, Arabic support; all methods now `async def` |
| `manager/manager_tester.py` | Minor updates to reflect async API |
| `manager/diagnosis_adapter.py` | Simplified to thin wrapper; removes inline logic; adds `run_async` bridge |
| `models/diagnosis/diagnosisengine.py` | Major AI-first refactor: modular, async `diagnose()`, decision fusion, safety layer |
| `models/diagnosis/__init__.py` | Now exports `async diagnose()` |
| `tests/test_manager.py` | All tests now wrapped in `run_async()` |
| `tests/test_diagnosis_engine.py` | Tests wrapped in `run_async()`; 2 new test cases added |
| `.gitignore` | Minor additions |
| `README.md` | Completely rewritten for backend+frontend monorepo |

---

## 4. Behavioral Differences

### 4.1 Synchronous vs. Asynchronous

`master` uses synchronous Python everywhere. `mego-edits` converts every public method of `ChatManager` and `DiagnosisEngine` to `async def`. This is a **breaking change at every call site**. The `runtime.py` module provides `run_async()` and `sync_from_async()` adapters so CLI scripts and synchronous tests can still invoke async code without an active event loop.

### 4.2 Diagnosis logic: rules-first vs AI-first

`master`: Rule engine is the sole decision-maker. If a lab matches a threshold, that's the diagnosis.

`mego-edits`: AI models (classifier and/or RAG) are the primary source; rules act as a safety net. The `_build_final_diagnosis()` method applies a multi-tier priority:
1. Fine-tuned classifier (if confidence ≥ `CLASSIFIER_PRIMARY_THRESHOLD` = 0.55) and not overridden by a "critical" rule finding.
2. RAG result (if classifier not available or low confidence, and not overridden by rules).
3. Rules fallback (always taken when classifier/RAG absent, or when rules detect critical severity and override AI).

A dedicated `_should_prefer_rules_over_ai()` method gates the AI output against safety criteria.

### 4.3 Clinical rules: inline code vs YAML

`master` encodes rules as Python dataclasses in `_RULES: List[_Rule]`. Changing any threshold requires editing Python code.

`mego-edits` externalises rules to `clinical_rules.yaml`. Any clinician can update thresholds without touching Python. The YAML `Rule` model adds `operator` (lt, le, gt, ge, eq, range) and `min/max` fields replacing the lambda-based `check` callable.

### 4.4 Therapy engine

`master`: `models/therapy/__init__.py` is empty — the therapy pipeline is a stub.

`mego-edits`: `TherapyEngine` sends diagnosis findings to Gemini LLM and parses structured `AITherapyPlanResponse` (clinical analysis, recommendations, lifestyle advice, emergency signs, disclaimer). If no abnormal findings are detected, a safe "no urgent therapy needed" message is returned without calling the LLM.

### 4.5 Chat

`master`: `run_chat()` always returns the same static string regardless of input.

`mego-edits`: Full multi-turn Arabic chat with rolling window history (max 12 messages), non-streaming and SSE streaming endpoints. The system prompt instructs Gemini to act as a medical assistant and not to reveal its underlying identity.

### 4.6 Symptom processing

`master`: `run_from_symptoms()` parses, validates, and concatenates symptoms into `manual_input["symptoms"]` as a space-joined string.

`mego-edits`: Adds `build_normalized_symptom_text()` which maps raw symptom tokens to canonical DDX-style question strings (50+ templates in `symptom_normalizer.py`). The validated result now carries both `raw_text` (normalised) and `symptom_list` (structured list) — the latter is passed to `diagnose_from_symptoms()` for symptom-rule matching.

### 4.7 Import path and module resolution

`master`: Runs from repo root; Python finds `manager/`, `models/`, `tests/` directly on `sys.path`.

`mego-edits`: Runs from `backend/` sub-directory. `app/main.py` explicitly inserts the repo root on `sys.path`. `pytest.ini` sets `testpaths = backend/tests`; `conftest.py` inserts `backend/` into `sys.path`. The CLI scripts still expect to be run from the repo root with `backend/` importable.

### 4.8 OCR module

Both branches share identical OCR source files after the rename from `models/ocr/` → `backend/models/ocr/`. `mego-edits` adds four new OCR helper files (`image_io.py`, `parsing.py`, `raw_ocr.py`, `types.py`) which split OCR sub-concerns that were previously inline.

---

## 5. Data Contracts and Schema Changes

### 5.1 Pipeline response shape

Both branches return a dict with the same top-level keys:

```json
{
  "status": "ok",
  "id": null,
  "ocr": { ... } | null,
  "diagnosis": { ... },
  "therapy": null | { ... },
  "warnings": [ ... ],
  "elapsed_ms": 123.4
}
```

**`master`**: `therapy` is always `null`.

**`mego-edits`**: `therapy` is populated with `{therapy_plan, structured_therapy, metadata}`.

### 5.2 Diagnosis sub-object

**`master`** keys: `findings`, `summary`, `disclaimer`, and optionally `rag_response`, `retrieved_cases`.

**`mego-edits`** adds: `decision_fusion`, `safety`, `final_diagnosis`, `classifier_prediction`, `gemini_response`, `gemini_response_metadata`, `structured_gemini_response`.

This is an **additive, backward-compatible extension** for consumers that accept extra keys.

### 5.3 `symptoms` path response

**`master`** adds top-level keys: `parsed`, `validated`, `review_required`.

**`mego-edits`** adds: `parsed`, `validated`, `normalized_text`, `review_required`.

### 5.4 Pydantic API schemas (new in `mego-edits`)

`backend/app/schemas/pipeline.py` formalises request bodies:
- `LabsPipelineRequest` — `labs: Dict[str, Any]` + optional `symptoms: str`
- `SymptomsPipelineRequest` — `text`, `use_symptom_parser`, `low_confidence_threshold`
- `DiagnosisOnlyRequest` — `report: Dict[str, Any]`
- `DiagnosisFromSymptomsRequest` — `text`, `low_confidence_threshold`

`backend/app/schemas/ai.py` adds structured AI output models:
- `AIDiagnosisResponse`, `AITherapyPlanResponse`, `AIClinicalResponse`

None of these schemas exist on `master`.

### 5.5 `run_ocr` return type change

`master`: `run_ocr()` is synchronous, returns `dict` directly.

`mego-edits`: `run_ocr()` is `async def`, returns an awaitable. Any code calling `run_ocr()` without `await` will silently get a coroutine object instead of a dict.

---

## 6. Dependency and Environment Changes

### 6.1 New Python packages (`mego-edits` only)

| Package | Use |
|---|---|
| `fastapi>=0.115.0` | Web API framework |
| `uvicorn[standard]>=0.32.0` | ASGI server |
| `python-multipart>=0.0.9` | File upload support |
| `pydantic-settings>=2.6.0` | `Settings` class loading from `.env` |
| `google-genai>=1.71.0` | Gemini async SDK (`google.genai.Client`) |

These are listed in `backend/requirements.txt` (API-only layer). They do not appear anywhere on `master`.

### 6.2 OCR / runtime requirements

`requirements-runtime.txt` is **identical** on both branches (same OCR stack: `paddlepaddle`, `numpy`, `opencv`, etc.).

### 6.3 Test requirements

`requirements-test.txt` is **identical** on both branches (`pytest`, `pytest-mock`, etc.).

### 6.4 Key environment variables (new in `mego-edits`)

From `backend/.env.example`:

```
USE_FINETUNED_CLASSIFIER=true
FINETUNED_MODEL_DIR=backend/artifacts/clinicalbert_classifier_natural
CLASSIFIER_MAX_LENGTH=256
CLASSIFIER_TRANSLATE_ARABIC=true
CLINICALBERT_MODEL_DIR=backend/artifacts/bio_clinicalbert
RAG_TRANSLATE_ARABIC=true
```

`master` has no equivalent `.env.example`; configuration is passed directly as CLI flags.

### 6.5 Frontend dependencies

Entirely new in `mego-edits`:
- React 19, react-dom 19
- TypeScript 5.6, Vite 6, `@vitejs/plugin-react`

No frontend existed on `master`.

### 6.6 Binary artifact committed to git

`backend/faiss_data/medical_cases.index` is a binary FAISS index (~several MB) committed directly to the `mego-edits` branch. This should not be tracked in git — see recommendations below.

---

## 7. Quality and Risk Evaluation

### 7.1 Strengths of `mego-edits`

1. **Production-ready API layer.** FastAPI with proper CORS, lifespan management, Pydantic validation, and Swagger docs is a significant upgrade from a CLI-only prototype.

2. **Async throughout.** All I/O-bound operations (LLM calls, OCR, future DB) are non-blocking. The server can serve concurrent requests without blocking on Gemini latency.

3. **Modular diagnosis engine.** Splitting `diagnosisengine.py` into `rules.py`, `rag.py`, `synthesis.py`, `text.py`, and externalising rules to YAML dramatically improves maintainability and testability.

4. **YAML-driven clinical rules.** Clinicians can update thresholds without Python knowledge. No redeployment needed if rules are hot-loaded (current implementation loads at module import time, but a reload hook is trivially added).

5. **Decision fusion with safety gating.** The AI-first approach with deterministic rules as a safety override is a sound medical AI design pattern. Critical rule findings can block an overconfident AI output.

6. **Full-stack React frontend.** Provides a usable interface for demonstrations and end-user testing without requiring CLI access.

7. **Arabic language support.** Translation layer for both RAG and classifier input; Arabic system prompt for chat — suitable for the target user demographic.

8. **Structured Pydantic AI responses.** `AIClinicalResponse`, `AITherapyPlanResponse` enforce schema at the LLM output boundary, preventing unstructured free-text leaking into the API response.

9. **Session-backed streaming chat.** In-memory session store with SSE streaming is a minimal but functional chat experience.

10. **Expanded test coverage.** New test files for RAG, therapy engine, symptom normalizer, pipeline metrics, and fine-tuned classifier.

### 7.2 Regressions and incompatibilities vs `master`

| Issue | Severity | Details |
|---|---|---|
| **Async-only `diagnose()` breaks sync callers** | High | Any script or test that calls `diagnose(report)` synchronously will receive a coroutine. Must wrap with `run_async()`. Some existing tests already updated; others may be missed. |
| **`run_ocr()` is now async** | High | `monkeypatch.setattr(manager, "run_ocr", lambda image: fake_ocr)` patches in a sync callable, but `pipeline_support.build_report()` uses `isawaitable()` to decide whether to `await` — so the monkeypatch still works. However, any production code path that calls `run_ocr()` synchronously will silently fail. |
| **Module paths changed** | Medium | All code in `manager/` and `models/` now lives under `backend/`. Absolute imports like `from models.diagnosis import DiagnosisEngine` break if run from repo root without `backend/` on `sys.path`. |
| **`chat_support.py` contains mojibake** | Medium | The file contains Arabic strings rendered as garbled question marks (`??? ?????? ???`). This is a UTF-8 encoding or git storage issue. At runtime these strings will not display correctly to Arabic-speaking users. |
| **Binary FAISS index in git** | Medium | `backend/faiss_data/medical_cases.index` is committed. This bloats the repository, may trigger LFS warnings, and means the index is difficult to update separately from code. |
| **In-memory session store** | Low-Medium | `ChatSessionStore` is in-process only. Every restart loses all session history. Concurrent workers (Gunicorn multi-process) will not share sessions. Not a regression vs `master` (master had no sessions at all), but creates a reliability expectation mismatch. |
| **`implementation_plan.md` committed to repo root** | Low | A developer planning artifact that should not be in source control. Contains local Windows file paths (`file:///c:/Users/10/Downloads/...`). |
| **`diagnose_from_symptoms()` relies on `symptom_list` key** | Low | If `symptom_list` is absent from the report (e.g., labs-only input), the function is simply not called. This is safe but the code path is only reachable via the normalised symptom flow; direct `raw_text` input without `symptom_list` will not trigger symptom rules. |
| **No authentication or rate limiting on API** | Low | The FastAPI app has no auth layer. For a medical prototype this is acceptable, but any public deployment without auth is a privacy risk. |
| **Settings cached with `@lru_cache`** | Low | `get_settings()` is cached for the process lifetime. Changes to `.env` after startup are not picked up. This can cause confusing behaviour in tests that set environment variables. |

### 7.3 Potential merge conflicts / hotspots

1. **`manager/chat_manager.py`** — Completely rewritten; will conflict on every line with `master`.
2. **`models/diagnosis/diagnosisengine.py`** — Refactored from monolith to orchestrator; conflicts inevitable.
3. **`models/diagnosis/__init__.py`** — Module API changed from sync to async.
4. **`tests/test_manager.py`** and **`tests/test_diagnosis_engine.py`** — Modified on both branches; manual reconciliation required.
5. **`.gitignore`** — Minor differences; easy to resolve.
6. **`README.md`** — Completely rewritten; `master` version is CLI-focused, `mego-edits` is API/frontend-focused.

### 7.4 Security and privacy concerns

1. **No input sanitisation on `raw_text`.** Free-text symptoms are passed directly to Gemini LLM. A malicious user could attempt prompt injection via the `/pipeline/symptoms` endpoint.  
   *Mitigation:* The system instruction is in the Gemini config, not in the user prompt, which limits injection surface. But `raw_text` itself reaches the LLM without sanitisation.

2. **`GEMINI_API_KEY` validation is weak.** `GeminiProvider` and `TherapyEngine` check `"AIza" in api_key` to determine validity. This is a formatting heuristic, not a cryptographic check.

3. **Temp files from image upload are unlinked in a `finally` block** — correctly implemented; no temp file leak.

4. **Binary FAISS index committed to git.** If the index contains patient data from real medical records, this is a serious HIPAA/privacy concern.

5. **SSE `stream_chat` endpoint uses `GET` with query-string parameters.** The user's chat message is visible in server logs, proxy logs, and browser history. Consider `POST`-based SSE or WebSocket instead.

---

## 8. Actionable Merge Recommendations

### 8.1 What to cherry-pick first (high value, low risk)

| Item | Action |
|---|---|
| `backend/models/diagnosis/clinical_rules.yaml` + `rules.py` | Adopt immediately. Externalising rules from Python to YAML is a pure improvement with no breaking changes to the rules themselves. Update `master`'s `diagnosisengine.py` to load from YAML. |
| `backend/models/common/ai_provider.py` | Clean async LLM abstraction. Cherry-pick into master's diagnosis engine to replace direct Gemini calls. |
| `backend/models/diagnosis/synthesis.py` + `backend/app/schemas/ai.py` | Structured Pydantic response models. No side effects — purely additive. |
| `backend/manager/session_store.py` | Self-contained; no external dependencies. Replaces the stub `run_chat()` with real session memory. |
| `backend/manager/symptom_normalizer.py` | Additive improvement to symptom parsing; backwards-compatible. |
| `backend/manager/runtime.py` | Needed immediately if any async migration begins. Zero dependencies. |
| `backend/models/therapy/engine.py` | Drop-in replacement for the empty therapy stub. |

### 8.2 What to avoid merging as-is

| Item | Reason |
|---|---|
| `backend/faiss_data/medical_cases.index` | Binary artifact should not be in git. Move to artifact storage / model registry. |
| `implementation_plan.md` | Developer planning doc with local file paths; delete before merging. |
| `backend/manager/chat_support.py` (as-is) | Contains mojibake Arabic strings. Fix encoding first (ensure file is saved as UTF-8 without BOM and re-committed). |
| `backend/app/routers/chat.py` `GET /stream` endpoint | SSE over `GET` exposes user messages in logs. Redesign as `POST`-based or WebSocket before merging. |

### 8.3 Suggested merge strategy

**Phase 1 — Foundation (no breaking changes to `master`)**

1. Add `backend/requirements.txt` (FastAPI/uvicorn/pydantic-settings/google-genai) to `master`'s root `requirements.txt` as optional extras.
2. Add `backend/models/common/ai_provider.py` and `backend/manager/runtime.py`.
3. Migrate clinical rules to YAML (`clinical_rules.yaml` + `rules.py`) without touching the `DiagnosisEngine` public API. Keep sync `diagnose()` on `master` as a thin `run_async()` wrapper.
4. Add `TherapyEngine` and wire it into `ChatManager.run_pipeline()` (returns `None` when no Gemini key is configured).

**Phase 2 — Async migration**

5. Convert `DiagnosisEngine.diagnose()` to `async def`. Update all call sites to use `run_async()` in sync contexts.
6. Convert `ChatManager` public methods to `async def`.
7. Update all tests to use `run_async()` or `pytest-asyncio` markers.
8. Run full test suite after each file change.

**Phase 3 — Web layer**

9. Merge the `backend/app/` FastAPI structure. Keep backward-compatible CLI entry points.
10. Add `.env.example`, update `README.md`.
11. Move FAISS index out of git; add `backend/faiss_data/` to `.gitignore`.

**Phase 4 — Frontend**

12. Merge `frontend/` directory entirely — no conflicts with `master` (it has no frontend).
13. Verify API contract with integration tests against the FastAPI server.

**Phase 5 — Quality hardening (before production)**

14. Fix `chat_support.py` encoding; verify Arabic strings display correctly.
15. Add input sanitisation for LLM-bound text fields.
16. Add a `POST /api/v1/chat/stream` endpoint to replace the `GET` variant.
17. Replace `@lru_cache` on `get_settings()` with a lazy-init pattern that can be reset in tests.
18. Delete `implementation_plan.md` from the `mego-edits` branch before merging.

### 8.4 Verification checklist after merge

```
Backend unit tests
[ ] pytest backend/tests/ — all tests pass without PaddleOCR installed
[ ] test_rule_based_detects_diabetes (rule engine regression)
[ ] test_accepts_plain_float_value
[ ] test_no_labs_returns_empty_summary
[ ] test_symptoms_only_returns_symptom_rule_finding
[ ] test_diagnosis_returns_fusion_and_safety_metadata
[ ] test_run_pipeline_from_labs
[ ] test_run_pipeline_from_image (monkeypatched OCR)
[ ] test_run_pipeline_manual_symptoms
[ ] test_therapy_engine (fallback when no API key)

API smoke tests (requires uvicorn running)
[ ] GET /api/v1/health → {"status": "ok"}
[ ] GET /api/v1/meta → includes rag_enabled, finetuned_classifier_enabled
[ ] POST /api/v1/pipeline/labs {"labs": {"glucose": 145}} → diagnosis.findings non-empty
[ ] POST /api/v1/pipeline/symptoms {"text": "fatigue and thirst"} → review_required field present
[ ] POST /api/v1/chat {"session_id": "s1", "message": "Hello"} → response string

Frontend tests
[ ] npm run build (no TypeScript errors)
[ ] Open http://127.0.0.1:5173 — Labs tab loads
[ ] Submit sample labs JSON → result panel renders findings
[ ] Submit symptom text → result panel renders
[ ] Navigate to Image tab, upload a PNG → pipeline runs (or graceful error)

Environment and config
[ ] Server starts with no .env file (all features disabled gracefully)
[ ] Server starts with USE_RAG=false, USE_FINETUNED_CLASSIFIER=false
[ ] GEMINI_API_KEY missing → therapy and chat return fallback messages, not 500
[ ] Binary FAISS index is NOT committed to git

Security
[ ] Confirm .gitignore includes backend/faiss_data/ and backend/artifacts/
[ ] Verify temp image files are cleaned up after /pipeline/image
[ ] Confirm no API keys or secrets in committed files
```

---

*Report generated by static code and diff analysis of `origin/master` (`28cbd4d`) and `origin/mego-edits` (`b656b61`). No execution of either branch was performed.*
