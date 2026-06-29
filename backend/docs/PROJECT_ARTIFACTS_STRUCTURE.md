# Project Artifacts Structure

This document describes the current presentation-ready artifact layout after the final RAG, classifier, and pipeline evaluation cleanup.

## Active Model Artifacts

Active RAG / FAISS bundle:

```text
backend/artifacts/artifacts/faiss_data_targeted/
```

Required files:

- `medical_cases.index`
- `metadata_mapping.pkl`
- `metadata_mapping.pkl.sha256`
- `index_info.json`

Active fine-tuned classifier bundle:

```text
backend/artifacts/artifacts/clinicalbert_classifier_targeted/
```

Required files:

- `config.json`
- `label_map.json`
- `model.safetensors` or `pytorch_model.bin`
- tokenizer files
- `test_predictions.csv`

Historical/natural comparison bundles are retained under:

```text
backend/artifacts/artifacts/faiss_data_natural/
backend/artifacts/artifacts/clinicalbert_classifier_natural/
```

These are model artifacts and must not be deleted during cleanup.

Separate mental-health support LoRA adapter:

```text
backend/artifacts/artifacts/mental_health/complaint_model_final/
```

Earlier deployment notes referenced this requested default:

```text
backend/artifacts/mental-health/
```

Required files:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `README.md`

This adapter is for `/api/v1/mental-health/chat` only. It is not part of the RAG, classifier, rules, or diagnosis-fusion pipeline.
Artifact/config validation has passed. Full live generation is pending GPU validation.

## Active RAG Diagnostics

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

Primary docs:

- `backend/docs/RAG_SCOPE_AND_LIMITATIONS.md`

## Active Classifier Diagnostics

```text
data/evaluation/classifier_diagnostics/
```

Final files:

- `classifier_assets_report.md`
- `classifier_assets_report.json`
- `classifier_label_consistency_report.md`
- `classifier_label_consistency_report.json`
- `classifier_metrics_summary.json`
- `classifier_classification_report.csv`
- `classifier_confusion_matrix.csv`
- `classifier_confusion_pairs.csv`
- `classifier_eval_report.md`
- `classifier_smoke_eval_summary.json`
- `classifier_smoke_eval_cases.csv`
- `classifier_smoke_eval_report.md`
- `classifier_metrics_trustworthiness_report.md`
- `data_leakage_report.md`
- `data_leakage_report.json`
- `dataset_split_report.md`
- `dataset_split_report.json`
- `test_predictions_provenance_report.md`

Primary docs:

- `backend/docs/CLASSIFIER_EVALUATION.md`

## Active Pipeline Diagnostics

```text
data/evaluation/pipeline_diagnostics/
```

Final files:

- `pipeline_architecture_report.md`
- `pipeline_architecture_report.json`
- `pipeline_eval_summary.json`
- `pipeline_eval_cases.csv`
- `pipeline_eval_report.md`
- `pipeline_failure_analysis.md`
- `pipeline_safety_report.md`
- `pipeline_failure_fix_plan.md`
- `pipeline_failure_fix_report.md`

Case sets:

```text
data/evaluation/pipeline_diagnostics/cases/
```

Primary docs:

- `backend/docs/PIPELINE_EVALUATION.md`

## Active Mental-Health Diagnostics

```text
data/evaluation/mental_model_diagnostics/
```

Final files:

- `mental_model_assets_report.md`
- `mental_model_assets_report.json`
- `mental_eval_summary.json`
- `mental_eval_cases.csv`
- `mental_eval_report.md`

Primary docs:

- `backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md`

Endpoint:

```text
POST /api/v1/mental-health/chat
```

## Cleanup Documentation

- `backend/docs/CLEANUP_REPORT.md`
- `backend/docs/FINAL_CLEANUP_REPORT.md`
- `backend/docs/PROJECT_ARTIFACTS_STRUCTURE.md`
- `data/evaluation/cleanup_audit_report.md`

## Archive Layout

Historical artifacts are preserved under:

```text
data/evaluation/archive/
```

Current archive groups:

- `archive/rag_diagnostics/` - older baseline, improved, DDXPlus-scope, coverage, and labeled RAG retrieval runs.
- `archive/rag_natural/` - historical natural RAG predictions and metrics.
- `archive/discussion/` - thesis/discussion exports.
- `archive/real_chat/` - generated real-chat cases and historical rules/classifier outputs.
- `archive/final_cleanup/pipeline/` - duplicate run-labeled pipeline outputs from `pipeline_safety_parser_fix`; canonical copies remain active in `pipeline_diagnostics/`.

## Source Scripts

Active health and evaluation scripts:

```text
backend/scripts/rag_health_check.py
backend/scripts/classifier_health_check.py
backend/scripts/pipeline_health_check.py
backend/scripts/verify_ddxplus_completeness.py
backend/scripts/evaluate_rag_retrieval_quality.py
backend/scripts/evaluate_classifier_quality.py
backend/scripts/evaluate_pipeline_quality.py
backend/scripts/investigate_classifier_data_leakage.py
backend/scripts/mental_model_health_check.py
backend/scripts/evaluate_mental_health_model.py
```

Historical or supporting scripts are retained for reproducibility and thesis discussion. Legacy defaults that used to write loose outputs directly under `data/evaluation/` now point to archive locations.

## Reproduction Commands

RAG:

```powershell
python backend/scripts/verify_ddxplus_completeness.py
python backend/scripts/evaluate_rag_retrieval_quality.py --case-set expanded --run-label expanded_final
python backend/scripts/rag_health_check.py --pretty
```

Classifier:

```powershell
python backend/scripts/classifier_health_check.py --pretty
python backend/scripts/evaluate_classifier_quality.py --pretty
python backend/scripts/investigate_classifier_data_leakage.py --pretty
```

Pipeline:

```powershell
python backend/scripts/pipeline_health_check.py --pretty
python backend/scripts/evaluate_pipeline_quality.py --run-label pipeline_safety_parser_fix --pretty
```

Mental-health support:

```powershell
python backend/scripts/mental_model_health_check.py --pretty
python backend/scripts/evaluate_mental_health_model.py --pretty
```

Selected validation:

```powershell
python -m pytest backend/tests/test_diagnosis_engine.py backend/tests/test_pipeline_quality_evaluation.py backend/tests/test_evaluate_pipeline_metrics.py backend/tests/test_rag_health_check.py -q
```
