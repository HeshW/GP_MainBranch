# Retraining Workflow For Major Accuracy Improvement

## Goal

Use high-value targeted hard cases to improve the classifier itself, not only the
clarification reranking layer.

## New Scripts

- [build_targeted_training_csv.py](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/scripts/build_targeted_training_csv.py:1)
- [merge_training_csvs.py](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/scripts/merge_training_csvs.py:1)

## Important Training Upgrade

- [train_clinicalbert_classifier.py](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/scripts/train_clinicalbert_classifier.py:1)

It now supports:

- `--model-name-or-dir`

This means we can continue fine-tuning from the current trained classifier
instead of restarting from plain Bio_ClinicalBERT every time.

## Suggested Drive Folder Layout

Put these items together in one folder on Drive, for example:

- `processed_ddxplus/`
- `clinicalbert_classifier_natural/`
- `targeted_cases_v1.json`
- `build_targeted_training_csv.py`
- `merge_training_csvs.py`
- `train_clinicalbert_classifier.py`

Then run the commands from inside that same folder in Colab.

## Step 1: Build Targeted CSVs

```bash
python build_targeted_training_csv.py \
  --include-follow-up
```

## Step 2: Merge With Base Natural CSVs

```bash
python merge_training_csvs.py \
  --base-csv processed_ddxplus/train_natural.csv \
  --targeted-csv targeted_training/train_targeted.csv \
  --output-csv targeted_training/train_merged.csv
```

Repeat the same for validation and test if needed.

## Step 3: Continue Fine-Tuning From The Current Model

```bash
python train_clinicalbert_classifier.py \
  --train-csv targeted_training/train_merged.csv \
  --val-csv targeted_training/validate_merged.csv \
  --test-csv targeted_training/test_merged.csv \
  --model-name-or-dir clinicalbert_classifier_natural \
  --output-dir clinicalbert_classifier_targeted \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 1e-5
```

## Step 4: Point The Pipeline To The New Model

- `FINETUNED_MODEL_DIR=backend/artifacts/clinicalbert_classifier_targeted`

## Why This Is The Major Enhancement Path

- It improves the classifier on the exact hard cases that currently fail.
- It can move one-shot accuracy, not only post-clarification behavior.
- It preserves the current model knowledge while adapting it to high-confusion cases.
