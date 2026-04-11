# Hybrid NLP Medical Assistant Architecture

## High-Level Diagram

```mermaid
flowchart TD
    U["User Input<br/>Image / Symptoms / Labs / Chat"] --> FE["React Frontend"]
    FE --> API["FastAPI API Layer"]

    API --> OCR["OCR Engine<br/>PaddleOCR + Report Parsing"]
    API --> NLP["Symptom Parsing + Validation + Normalization<br/>Arabic / English / Mixed Input"]

    OCR --> REP["Structured Report"]
    NLP --> REP

    REP --> DE["Diagnosis Engine"]

    DE --> RULES["Rule-Based Engine<br/>Clinical Rules + Safety Patterns"]
    DE --> CLS["Fine-Tuned ClinicalBERT Classifier<br/>Primary In-Scope Prediction"]
    DE --> RAG["RAG Layer<br/>BioClinicalBERT Embeddings + FAISS Retrieval"]

    RAG --> RET["Retrieved Similar Cases"]
    RET --> FUSION["Decision Fusion"]
    RULES --> FUSION
    CLS --> FUSION

    FUSION --> UNC["Uncertainty Check"]
    UNC -->|Low confidence / conflict / ambiguous input| CLAR["Clarification Mode"]
    CLAR --> Q["Targeted Follow-Up Questions<br/>Based on top candidate diseases"]
    Q --> ANS["User Answers"]
    ANS --> RERANK["Answer-Aware Re-Ranking"]
    RERANK --> FUSION

    UNC -->|Confident| OUT["Final Diagnosis + Confidence + Safety"]
    FUSION --> OUT

    OUT --> TH["Therapy / Recommendation Engine"]
    OUT --> CHAT["Chat + Explanation Layer"]

    TH --> GEM["Gemini Provider"]
    CHAT --> GEM
    RAG --> GEM
```

## Architecture Summary

- `Frontend`: React UI for image upload, text entry, results, and clarification flow.
- `API / Orchestration`: FastAPI plus `ChatManager` coordinates OCR, parsing, diagnosis, therapy, and chat.
- `Input Understanding`: The normalization layer converts noisy free text into a training-like clinical representation.
- `Diagnosis Core`: Hybrid combination of rules, fine-tuned classifier, and RAG retrieval.
- `Interactive Refinement`: If confidence is insufficient, the system asks follow-up questions tied to suspected diseases, then reranks.
- `Output Layer`: Returns diagnosis, safety metadata, therapy suggestions, and chat/explanations.

## Current Design Message For Defense

The most accurate way to present this architecture is:

> A hybrid, multilingual, interactive clinical decision-support prototype.

Not:

> A one-shot medical diagnosis model.
