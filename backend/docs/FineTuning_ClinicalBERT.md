# Fine-Tuning ClinicalBERT

## When fine-tuning is the better choice

Use fine-tuning when your main goal is:

- predicting a diagnosis label directly
- measuring real classification metrics on train/validation/test splits
- producing a proper confusion matrix on unseen test data

Keep `RAG + FAISS` when your main goal is:

- retrieving similar cases
- augmenting an LLM with supporting examples
- improving explainability and follow-up question generation

Best practice for this project:

- use a fine-tuned classifier for the primary diagnosis prediction
- keep RAG as a secondary support layer

## Required dataset format

The training script expects CSV files with at least:

- `combined_text`
- `pathology`

This matches the processed DDX-style structure already used in the notebook.

Example columns:

- `patient_id`
- `age`
- `sex`
- `symptoms_text`
- `pathology`
- `combined_text`

## Training command

Run from repo root:

```powershell
.\.venv_rag\Scripts\python.exe backend\scripts\train_clinicalbert_classifier.py `
  --train-csv path\to\train_processed.csv `
  --val-csv path\to\validate_processed.csv `
  --test-csv path\to\test_processed.csv
```

## Quick smoke test on a small subset

Before full training, you can verify the pipeline on a small subset:

```powershell
.\.venv_rag\Scripts\python.exe backend\scripts\train_clinicalbert_classifier.py `
  --train-csv path\to\train_processed.csv `
  --val-csv path\to\validate_processed.csv `
  --test-csv path\to\test_processed.csv `
  --max-train-samples 200 `
  --max-val-samples 80 `
  --max-test-samples 80 `
  --epochs 1
```

This is useful only to confirm:

- training runs successfully
- artifacts are saved
- inference can be integrated into the backend
- confusion matrix generation works

## Outputs

The script writes to:

- `backend/artifacts/clinicalbert_classifier/`

Generated files include:

- saved fine-tuned model
- tokenizer
- `label_map.json`
- `training_history.json`
- `training_summary.json`
- `test_confusion_matrix.csv`
- `test_classification_report.csv`
- `test_predictions.csv`

## Important note

The current project codebase was using ClinicalBERT as a pretrained encoder for
embeddings and retrieval. That is not the same as fine-tuning.

This script adds a real supervised fine-tuning path.
