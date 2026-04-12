# Accuracy Improvement Data Workflow

## Goal

Improve end-to-end robustness by creating more realistic text input for training
and evaluation, instead of relying only on the original DDX-style phrasing.

## New Script

- [generate_augmented_ddxplus_data.py](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/scripts/generate_augmented_ddxplus_data.py:1)

This script takes a natural DDX CSV and expands each case into multiple
patient-style text variants.

## Why This Helps

- Reduces train-inference mismatch.
- Exposes the classifier to less rigid phrasing.
- Adds English and mixed Arabic-style phrasing.
- Creates a reusable augmentation pipeline instead of hand-editing data.

## Example Command

```powershell
.\.venv_rag\Scripts\python.exe backend\scripts\generate_augmented_ddxplus_data.py `
  --input-csv data\processed_ddxplus\train_natural.csv `
  --output-csv data\processed_ddxplus\train_augmented.csv `
  --include-original `
  --styles clinical_en patient_en mixed_ar
```

## Suggested Usage

1. Build the base natural CSVs from DDXPlus.
2. Generate augmented training rows.
3. Fine-tune the classifier on the augmented CSV.
4. Re-run end-to-end evaluation on the same benchmark.
5. Compare:
   - one-shot Top-1
   - one-shot Top-3
   - post-clarification Top-1
   - post-clarification Top-3

## Recommended Next Step

After this augmentation pipeline, the next improvement should be either:

- adding clarification-aware training examples

or

- improving answer-to-candidate reranking with explicit symptom scoring
