# AI Architecture Evaluation

## 1. Overview

The AI part of this project is not a single model. It is a hybrid medical-analysis pipeline composed of:

1. OCR extraction for medical report images.
2. Rule-based diagnosis from structured lab values.
3. Optional AI enhancement through:
   - RAG over similar medical cases using ClinicalBERT + FAISS + Gemini.
   - Fine-tuned text classifier for diagnosis prediction.
4. LLM-based therapy recommendation generation in Arabic.
5. LLM-based chat assistant with short session memory.

This makes the system a **hybrid AI architecture** rather than a pure end-to-end LLM system.

## 2. Main AI Components

### A. API / Orchestration Layer

- `backend/app/main.py`
- `backend/app/routers/pipeline.py`
- `backend/app/routers/chat.py`
- `backend/manager/chat_manager.py`

Role:
- Receives input from frontend or API clients.
- Routes requests into OCR, diagnosis, therapy, and chat flows.
- Keeps the main orchestration logic in `ChatManager`.

### B. OCR Engine

- `backend/models/ocr/engine.py`

Role:
- Accepts image input.
- Uses `PaddleOCR` to extract raw text.
- Parses lab values and sections from OCR output.
- Produces structured report fields such as:
  - `labs`
  - `raw_text`
  - `fields`
  - `sections`

### C. Rule-Based Diagnosis Engine

- `backend/models/diagnosis/diagnosisengine.py`
- `backend/models/diagnosis/rules.py`
- `backend/models/diagnosis/clinical_rules.yaml`

Role:
- Core deterministic diagnosis path.
- Reads lab values from the structured report.
- Compares them with threshold-based clinical rules from YAML.
- Returns findings, confidence, severity, and evidence.

This is the current stable diagnostic backbone of the system.

### D. RAG Layer

- `backend/models/diagnosis/rag.py`
- `backend/faiss_data/medical_cases.index`
- `backend/faiss_data/metadata_mapping.pkl`

Role:
- Converts patient text into embeddings using Bio_ClinicalBERT.
- Searches similar cases from a FAISS index.
- Sends retrieved cases to Gemini to produce a structured preliminary assessment.

Pipeline:
- Patient text -> embedding -> FAISS retrieval -> Gemini reasoning with retrieved context.

### E. Fine-Tuned Classifier

- `backend/models/diagnosis/rag.py`
- `backend/scripts/train_clinicalbert_classifier.py`

Role:
- Loads a fine-tuned sequence-classification model.
- Predicts the most likely diagnosis label from patient text.
- Can translate Arabic into English before inference if Gemini is configured.

### F. LLM Provider Abstraction

- `backend/models/common/ai_provider.py`

Role:
- Provides one abstraction layer for Gemini.
- Supports:
  - async generation
  - streaming
  - structured JSON output using Pydantic schemas

This is a strong architectural decision because it reduces direct coupling between business logic and the external model API.

### G. Therapy Generation

- `backend/models/therapy/engine.py`

Role:
- Uses diagnosis findings as input.
- Requests Gemini to generate a structured treatment / recommendation plan in Arabic.
- Returns both structured JSON and markdown-like text.

### H. Chat Assistant

- `backend/manager/chat_manager.py`
- `backend/manager/session_store.py`
- `backend/manager/chat_support.py`

Role:
- Offers conversational medical chat.
- Maintains lightweight in-memory conversation history.
- Uses Gemini streaming for live chat responses.

## 3. End-to-End AI Data Flow

### Image flow

1. User uploads image.
2. API stores temp image.
3. `ChatManager.run_pipeline()` calls OCR.
4. OCR extracts structured labs and report text.
5. `DiagnosisEngine` runs rule-based diagnosis.
6. Optional RAG / classifier runs if enabled.
7. `TherapyEngine` generates treatment recommendations.
8. API returns diagnosis + therapy + warnings.

### Text / symptoms flow

1. User enters symptoms.
2. Symptom parser extracts symptoms and any lab-like values.
3. Validator normalizes them and marks low-confidence cases.
4. Structured manual input is merged into report format.
5. Diagnosis + therapy pipeline runs as above.

## 4. Architectural Strengths

### Strength 1: Hybrid design

The system does not depend only on a generative model. It combines:
- deterministic clinical rules
- retrieval-based reasoning
- optional classifier
- LLM-generated explanations / therapy

This is better for medical systems than using an LLM alone.

### Strength 2: Clear separation of responsibilities

The project separates:
- API layer
- orchestration layer
- OCR layer
- diagnosis layer
- therapy layer
- provider abstraction

This makes the code easier to explain and evolve.

### Strength 3: Optional advanced AI paths

RAG and the fine-tuned classifier are optional, not mandatory.
This gives graceful architectural scaling:
- basic mode: OCR + rules
- advanced mode: OCR + rules + RAG / classifier + Gemini

### Strength 4: Structured AI outputs

The project uses Pydantic schemas for Gemini outputs.
This is excellent for production-style design because it makes responses more predictable.

### Strength 5: Some evaluation tooling already exists

There is already a script for RAG evaluation:
- `backend/scripts/evaluate_rag_confusion.py`

This is useful in discussion because it shows the team already thought about measurable AI quality.

## 5. Current Weaknesses / Risks

### Weakness 1: Therapy engine is currently broken at runtime

`backend/models/therapy/engine.py` is missing required imports such as:
- `Optional`
- `Dict`
- `Any`
- `GeminiProvider`
- `logging`

Also `logger` is used without being defined.

This means the therapy component can crash during `ChatManager` initialization when no Gemini key is provided.

### Weakness 2: The real diagnostic decision is still mostly rule-based

Although the project contains RAG and classifier components, the primary diagnosis result is still built from rule matching.
So in the discussion, it is more accurate to say:

> "The deployed diagnostic core is rule-based, with AI-enhancement modules available for retrieval and classification."

Not:

> "The whole diagnosis is fully AI-driven."

### Weakness 3: No unified evaluation for the whole pipeline

There is evaluation for RAG retrieval, but there is no single benchmark covering:
- OCR extraction accuracy
- diagnosis correctness
- therapy quality
- Arabic translation quality
- end-to-end clinical usefulness

### Weakness 4: Chat memory is in-memory only

`session_store.py` keeps sessions only in RAM.
So chat history is lost after restart and is not suitable for real deployment.

### Weakness 5: Medical safety controls are still limited

The project has disclaimers, but it still lacks stronger safety controls such as:
- confidence gating before showing AI advice
- mandatory escalation for critical findings
- audit logging for model decisions
- explicit unsafe-output handling

### Weakness 6: No evidence fusion layer

Rule-based findings, classifier prediction, and RAG output are returned side by side.
But there is no fusion logic that decides:
- which one is primary
- when one source overrides another
- how conflicts are resolved

### Weakness 7: Dependence on external Gemini for multiple roles

Gemini is used for:
- RAG explanation
- Arabic translation
- therapy generation
- chat

This creates operational risk:
- API-key dependency
- latency dependency
- cost dependency
- partial failure propagation

## 6. What Is Missing To Be Ready For Discussion

### Must-fix before discussion

1. Fix the therapy engine runtime issue.
2. Make sure the backend can start successfully without crashing.
3. Prepare one clear architecture diagram showing the AI flow.
4. Be explicit that diagnosis is hybrid, not pure LLM diagnosis.

### High-value additions

1. Add an evaluation table:
   - OCR accuracy
   - diagnosis accuracy
   - RAG top-1 / top-3 / top-5
   - classifier accuracy
2. Add one slide or section called "Safety and Limitations".
3. Add one slide or section called "Future Improvements".
4. Add persistent storage for sessions if chat is part of the demo.
5. Add fallback behavior when Gemini is unavailable.

### Best next AI improvements

1. Build a decision-fusion layer combining:
   - rule findings
   - classifier output
   - RAG evidence
2. Add confidence thresholds to suppress weak AI outputs.
3. Add evaluation datasets for Arabic symptom input.
4. Add tests for therapy generation and chat.
5. Add monitoring / logging around model failures and latency.

## 7. Suggested Defense Narrative

You can explain the AI architecture like this:

> "Our project uses a hybrid AI architecture for medical-report analysis. We begin with OCR to transform report images into structured medical data. Then a rule-based diagnosis engine applies validated clinical thresholds from a YAML knowledge base. On top of that, we support advanced AI extensions: a RAG pipeline using ClinicalBERT embeddings, FAISS retrieval, and Gemini for context-aware reasoning, plus an optional fine-tuned classifier for diagnostic prediction. Finally, Gemini is also used for Arabic therapy recommendations and conversational chat. This design was chosen to balance reliability, explainability, and extensibility."

Then say:

> "The strongest production-ready part today is the OCR + rule-based diagnosis path. The advanced AI modules are integrated architecturally, but still need more validation, stronger safety controls, and unified evaluation before being considered fully mature."

## 8. Final Assessment

### Current maturity

The AI architecture is **good as a graduation-project architecture** because it is modular, hybrid, and technically ambitious.

### Current readiness

It is **good for presentation and discussion**, but **not yet fully production-ready**.

### Most honest overall judgment

The project is:
- architecturally strong
- technically rich
- discussion-ready after small cleanup
- still missing reliability and evaluation work for serious real-world medical deployment
