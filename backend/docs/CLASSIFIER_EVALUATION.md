# Classifier Evaluation

This document describes the reproducible ClinicalBERT classifier investigation.
It is intentionally evaluation-only: do not retrain, alter model weights, or
rebuild FAISS as part of this workflow.

## Active Runtime Path

The backend reads the fine-tuned classifier from `FINETUNED_MODEL_DIR`. In the
current local configuration this resolves to:

```text
backend/artifacts/artifacts/clinicalbert_classifier_targeted
```

The active RAG/FAISS path is:

```text
backend/artifacts/artifacts/faiss_data_targeted
```

## Reproduce Reports

Run from the repository root:

```bash
python backend/scripts/evaluate_classifier_quality.py --pretty
python backend/scripts/classifier_health_check.py --pretty
python -m py_compile backend/scripts/evaluate_classifier_quality.py backend/scripts/classifier_health_check.py
```

The evaluator writes reports to:

```text
data/evaluation/classifier_diagnostics/
```

Generated files include:

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

## Evaluation Scope

The rigorous metrics are recomputed from the saved `test_predictions.csv` file.
This gives full test-set top-1 metrics without rerunning training.

The independent smoke evaluation reloads the existing classifier weights and
runs a deterministic stratified sample from the saved test rows. By default it
uses 49 cases, enough to cover the label universe while keeping CPU runtime
practical. It measures top-1, top-3, top-5, MRR@5, and confidence calibration.
Use `--max-smoke-cases 0` only when you intentionally want to run inference over
all saved test rows.

## Current Findings

Current generated reports show:

- Active classifier path: `backend/artifacts/artifacts/clinicalbert_classifier_targeted`
- Label count: `49`
- Saved test rows: `2006`
- Saved-test accuracy: `0.991525`
- Saved-test macro F1: `0.988235`
- Smoke top-1/top-3/top-5: `0.979592` / `1.000000` / `1.000000`
- Smoke MRR@5: `0.989796`
- Smoke expected calibration error: `0.035768`

Top saved-test confusion pairs:

- `Acute rhinosinusitis` -> `Chronic rhinosinusitis`: `12`
- `Acute laryngitis` -> `Viral pharyngitis`: `2`
- `Unstable angina` -> `Stable angina`: `2`
- `Possible NSTEMI / STEMI` -> `Unstable angina`: `1`

Retraining is recommended later but is not currently necessary. The evidence
supports targeted data/calibration work for confusing labels before any new
training run. The current label universe is consistent across classifier, RAG,
FAISS metadata, and targeted/natural label maps.

## Retraining Policy

Retraining is not automatic. Treat retraining as justified only if the reports
show one or more of the following:

- Low overall accuracy or macro F1.
- Important classes with persistently weak recall/F1.
- Label inconsistency between classifier, RAG, and FAISS metadata.
- High-confidence wrong predictions or poor calibration.
- Real chat phrasing failures that cannot be fixed with preprocessing,
  confidence gating, or fusion logic.

If classifier labels change, retraining is required and FAISS may also need a
rebuild if the retrieval corpus label universe changes. If only calibration or
additional examples are added inside the same 49-label universe, FAISS does not
need to be rebuilt.
