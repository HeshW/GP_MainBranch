# Colab Runbook

Use [Colab_Natural_AI_Pipeline.ipynb](C:\Users\10\Downloads\New folder (5)\GP_MainBranch-master\notebooks\Colab_Natural_AI_Pipeline.ipynb) in Colab.

## Goal

Produce two clean artifacts:

- `backend/artifacts/clinicalbert_classifier_natural`
- `backend/faiss_data_natural`

These replace the older artifacts that were built on noisy / leaky text representations.

## Colab Inputs

You need:

- the repo uploaded or cloned into Colab
- the DDXPlus HuggingFace dataset directory

Expected DDX directory structure:

- `train/`
- `validate/` or `validation/`
- `test/`

## Colab Outputs

The notebook exports to Drive:

- `clinicalbert_classifier_natural/`
- `faiss_data_natural/`
- `processed_ddxplus/`

## After Colab

Copy the exported folders into the local project and then update the backend config to point at:

- `backend/artifacts/clinicalbert_classifier_natural`
- `backend/faiss_data_natural`

Then rerun end-to-end evaluation.
