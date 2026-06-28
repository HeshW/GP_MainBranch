# Project Artifacts Structure

## Active Runtime Artifacts

### FAISS RAG Index

Active targeted FAISS bundle:

```text
backend/artifacts/artifacts/faiss_data_targeted/
```

Important files:

- `medical_cases.index`
- `metadata_mapping.pkl`
- `metadata_mapping.pkl.sha256`
- `metadata_mapping.json`
- `index_info.json`

Current validated counts:

- FAISS vectors: `12025`
- Metadata rows: `12025`
- Unique pathologies: `49`

Legacy/natural FAISS bundle retained for comparison/discussion:

```text
backend/artifacts/artifacts/faiss_data_natural/
```

## Active Classifier Artifacts

Targeted classifier bundle:

```text
backend/artifacts/artifacts/clinicalbert_classifier_targeted/
```

Natural classifier bundle retained for comparison/discussion:

```text
backend/artifacts/artifacts/clinicalbert_classifier_natural/
```

Important files commonly used by validation scripts:

- `label_map.json`
- `test_predictions.csv`
- model/tokenizer files used by Hugging Face loaders

## Active Evaluation Artifacts

Canonical RAG diagnostics:

```text
data/evaluation/rag_diagnostics/
```

Final files:

- `ddxplus_completeness_report.md`
- `ddxplus_completeness_report.json`
- `expanded_retrieval_eval_summary.json`
- `expanded_retrieval_eval_cases.csv`
- `expanded_retrieval_eval_report.md`
- `final_rag_investigation_report.md`

Cleanup audit:

- `data/evaluation/cleanup_audit_report.md`

## Archived Evaluation Artifacts

Archive root:

```text
data/evaluation/archive/
```

Archived groups:

- `archive/rag_diagnostics/`
  - Old smoke runs, baseline/improved runs, DDXPlus-scope intermediate runs, duplicate expanded labeled runs, and intermediate coverage reports.
- `archive/rag_natural/`
  - Historical natural-RAG metrics, predictions, classification report, and confusion matrix.
- `archive/discussion/`
  - Thesis/discussion summary and classifier discussion outputs.
- `archive/real_chat/`
  - Real-chat generated cases and classifier/rules evaluation outputs.

These files are not active final RAG diagnostics, but they are intentionally preserved for discussion and reproducibility context.

## Active RAG Scripts

```text
backend/scripts/evaluate_rag_retrieval_quality.py
backend/scripts/verify_ddxplus_completeness.py
backend/scripts/rag_health_check.py
```

Supporting retained scripts:

```text
backend/scripts/evaluate_rag_confusion.py
backend/scripts/build_discussion_evaluation.py
backend/scripts/rebuild_faiss_from_ddx.py
backend/scripts/train_clinicalbert_classifier.py
```

`rebuild_faiss_from_ddx.py` and `train_clinicalbert_classifier.py` are not part of normal cleanup or evaluation runs. Use them only when new data/labels are intentionally added.

## Documentation Files

RAG and cleanup documentation:

- `backend/docs/RAG_SCOPE_AND_LIMITATIONS.md`
- `backend/docs/CLEANUP_REPORT.md`
- `backend/docs/PROJECT_ARTIFACTS_STRUCTURE.md`
- `data/evaluation/cleanup_audit_report.md`

Existing general setup/architecture docs:

- `backend/docs/Architecture.md`
- `backend/docs/Fresh_Machine_Setup.md`

## Reproduction Commands

Completeness:

```powershell
python backend\scripts\verify_ddxplus_completeness.py
```

Expanded retrieval evaluation:

```powershell
python backend\scripts\evaluate_rag_retrieval_quality.py --case-set expanded --run-label expanded_final
```

Health check:

```powershell
python backend\scripts\rag_health_check.py --pretty
```

Selected tests:

```powershell
python -m pytest backend\tests\test_rag.py backend\tests\test_rag_health_check.py backend\tests\test_startup_guardrails.py backend\tests\test_diagnosis_engine.py -q
```
