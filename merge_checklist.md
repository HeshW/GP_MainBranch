# Merge Checklist: hesh_mego_final

## Goal
Integrate mego-final diagnosis intelligence into hesh-edits2 while preserving hesh-edits2 reliability, security, and UI/UX behavior.

## Branch + Safety
- [x] Create integration branch from hesh-edits2: hesh_mego_final
- [x] Confirm clean working tree before merge commit
- [x] Keep targeted artifacts local-only (no large files committed)

## Artifact Readiness
- [x] Verify targeted ClinicalBERT bundle exists
- [x] Verify targeted FAISS bundle exists
- [x] Create metadata_mapping.pkl.sha256 for targeted FAISS bundle
- [x] Wire local env to targeted artifact paths for validation run

## Merge Integration Steps
- [x] Keep hesh chat support intact (corruption-safe Arabic/English)
- [x] Hybrid-merge diagnosisengine.py (mEGO brain + hesh reliability knobs)
- [x] Hybrid-merge rag.py (mEGO rerank/pathology + hesh secure metadata loader)
- [x] Merge symptom_parser.py preserving richer mEGO negation/aliases
- [x] Validate chat_manager compatibility with merged diagnosis/rag contracts
- [x] Preserve hesh app/main, deps, pipeline security flow
- [x] Preserve hesh frontend dual-mode UI and streaming client behavior
- [x] Reconcile diagnosis-focused test suite for merged engine

## Verification Gates
- [x] Backend targeted unit tests pass (security/startup/rag/diagnosis)
- [x] Backend API integration tests pass
- [x] Frontend smoke tests pass
- [x] No startup crash when optional AI assets missing (degraded mode)
- [x] RAG secure metadata hash verification enforced by default

## Release Readiness
- [x] Summarize merged deltas
- [x] Record residual risks + mitigations
- [x] Final status: production-ready candidate

## Notes
- This checklist is updated after each major step.
- If any required final assets are missing, pause and ask user (do not guess).
