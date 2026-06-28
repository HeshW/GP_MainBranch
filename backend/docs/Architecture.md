# Backend Architecture

This backend is a FastAPI service around a hybrid medical analysis pipeline. It supports
manual labs, report-image OCR, free-text symptoms, diagnosis fusion, clarification,
therapy placeholders, and chat.

## Main Runtime Flow

```text
Frontend / API client
 -> FastAPI routers in app/routers
 -> ChatManager orchestration
 -> report preparation
    -> OCR path: models/ocr/OCREngine
    -> symptom path: parser -> validator -> normalizer
    -> labs path: direct structured labs
 -> DiagnosisEngine
    -> lab rules and symptom rules
    -> optional RAG retrieval
    -> optional fine-tuned classifier
    -> candidate collection and reranking
    -> clarification planning when evidence is weak or conflicting
    -> safety metadata and patient-facing synthesis
 -> optional TherapyEngine response
 -> API response
```

## Key Components

- `app/main.py`: FastAPI application factory, startup guardrails, middleware, and router registration.
- `app/routers/pipeline.py`: HTTP endpoints for labs, image OCR, symptoms, diagnosis-only, and clarification.
- `app/routers/chat.py`: non-streaming and SSE chat endpoints.
- `manager/chat_manager.py`: public facade coordinating OCR, diagnosis, therapy, and chat.
- `manager/symptom_parser.py`: extracts symptom/lab mentions, negation, duration, and context from free text.
- `manager/symptom_validator.py`: canonicalizes parsed symptoms/labs and marks review-required cases.
- `manager/symptom_normalizer.py`: builds diagnosis-ready natural text from parsed evidence.
- `models/ocr/*`: PaddleOCR integration plus regex parsing for labs, fields, sections, and raw OCR confidence.
- `models/diagnosis/rules.py`: deterministic lab and symptom rules.
- `models/diagnosis/diagnosisengine.py`: diagnosis fusion, candidate generation, reranking, clarification, safety, and summary.
- `models/diagnosis/rag.py`: optional ClinicalBERT/FAISS retrieval and fine-tuned classifier helpers.
- `models/diagnosis/synthesis.py`: optional LLM patient-facing clinical response synthesis.
- `models/therapy/engine.py`: feature-flagged therapy generation/fallback.

## Diagnosis Flow

```text
Prepared report
 -> diagnose_from_labs
 -> diagnose_from_symptoms
 -> optional RAG query
 -> optional classifier prediction
 -> _collect_diagnostic_candidates
 -> _expand_base_diagnostic_candidates
 -> _rerank_base_candidates
 -> _build_final_diagnosis
 -> _build_clarification
 -> _build_safety
 -> optional DiagnosisResponseSynthesizer
```

The current design is conservative but still one-shot oriented: it can show a `final_diagnosis`
before clarification is complete. Future diagnosis work should make `assessment_state`,
differential diagnosis ranking, and serious-condition support gating explicit.

## Operational Notes

- RAG and classifier are optional and are disabled if required assets are unavailable.
- Therapy is feature-flagged and defaults to disabled.
- Chat session memory is in-process only and is lost on restart.
- Local model/index artifacts should live under ignored artifact directories, not as partially tracked files.

## Active Setup Documentation

Keep `Fresh_Machine_Setup.md` for environment onboarding. Model training and artifact generation are covered
by the notebooks runbook under `notebooks/COLAB_RUNBOOK.md`.
