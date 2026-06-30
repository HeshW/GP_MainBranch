# Repository Cleanup Report

Date: 2026-06-30

## Classification

Required for runtime:
- `backend/app/`, `backend/manager/`, `backend/models/`, `backend/.env.example`
- `backend/requirements-runtime.txt`, `backend/requirements.txt`
- `frontend_next/src/`, `frontend_next/public/`, `frontend_next/doctors/`, `frontend_next/package.json`, `frontend_next/package-lock.json`

Required for development:
- `requirements.txt`, `pytest.ini`
- `backend/tests/`, `backend/requirements-test.txt`
- `frontend_next/tsconfig.json`, `frontend_next/eslint.config.mjs`, `frontend_next/next.config.ts`
- `backend/models/ocr/scripts/` smoke/debug utilities

Required for evaluation and reproducibility:
- `backend/scripts/`
- `backend/docs/`
- `backend/requirements-ai.txt`
- `notebooks/ClinicalBERT_finetuning_ddx.ipynb`
- `notebooks/Colab_All_In_One_Natural_AI.ipynb`
- `notebooks/Colab_Natural_AI_Pipeline.ipynb`
- `notebooks/COLAB_RUNBOOK.md`

Legacy, duplicate, or unused:
- `frontend/` Vite application, archived to `archive/frontend-vite/`
- `notebooks/notebook_code.py`, archived to `archive/notebooks/notebook_code.py`
- `docs/state.md`, archived to `archive/project-state/state.md`
- `.github/agents/cleaner.agent.md`, archived to `archive/agents/cleaner.agent.md`
- Duplicate public assets removed from `frontend_next/public/`

Temporary, cache, and generated:
- Python `__pycache__/` directories
- `.pytest_cache/`
- `.ipynb_checkpoints/`
- `frontend_next/.next/`
- `frontend/node_modules/`, `frontend/dist/`
- Local generated evaluation outputs under `data/evaluation/` and `backend/data/`

Generated files intentionally kept out of Git:
- `data/raw/**`
- `data/processed/**`
- `data/evaluation/**`
- `backend/artifacts/**`
- local virtual environments, databases, logs, Node installs, and build outputs

## Files Deleted

Deleted generated/cache directories:
- `frontend/node_modules/`
- `frontend/dist/`
- `frontend_next/.next/`
- backend Python `__pycache__/` directories

Deleted verified duplicate unused assets:
- `frontend_next/public/nabda-logo (1).svg`
- `frontend_next/public/nabda-icon (1).svg`
- `frontend_next/public/hero-medical-bots.png`

These public assets were byte-for-byte duplicates of kept files and were not referenced by `frontend_next/src`, `frontend_next/public/site.webmanifest`, or frontend docs.

## Files Moved Or Archived

Tracked archive moves:
- `frontend/` -> `archive/frontend-vite/`
- `notebooks/notebook_code.py` -> `archive/notebooks/notebook_code.py`

Local ignored archive moves:
- `docs/state.md` -> `archive/project-state/state.md`
- `.github/agents/cleaner.agent.md` -> `archive/agents/cleaner.agent.md`
- `backend/data/` -> `data/evaluation/archive/repository_cleanup_20260630/backend_data_legacy/`
- loose `data/evaluation/pipeline_end_to_end_*` outputs -> `data/evaluation/archive/repository_cleanup_20260630/loose_pipeline_end_to_end/`

## Files Intentionally Kept

- Diagnosis pipeline code, rules, label mappings, and model-provider code.
- All final notebooks referenced by docs or evaluation scripts.
- All backend evaluation scripts, including train/rebuild scripts needed for reproducibility. They were not executed.
- All model artifacts and FAISS/index files. No artifacts were changed, moved, rebuilt, or deleted.
- `frontend_next/doctors/drs/**` generated doctor images because they are imported by `frontend_next/src/lib/mock-doctors.ts`.
- Backend docs describing classifier, RAG, pipeline, and mental-health evaluation outputs.

## Estimated Size Reduction

Approximate local workspace reduction:
- `frontend_next/.next/`: 376 MB
- `frontend/node_modules/` and `frontend/dist/`: about 114 MB
- duplicate public assets: about 1.5 MB
- Python cache directories: small, under 1 MB

Tracked repository size reduction is mainly from moving the legacy Vite app to `archive/` rather than deleting it. The active top-level tree is cleaner, while historical files remain reviewable.

## Manual Review Items

- `archive/frontend-vite/` can be deleted later if no one needs the old Vite UI.
- `archive/notebooks/notebook_code.py` can be deleted if the retained notebooks fully cover reproducibility.
- `backend/artifacts/` contains large ignored local model artifacts and possible duplicate natural/targeted variants. They were intentionally not touched.
- `data/evaluation/archive/repository_cleanup_20260630/` contains ignored generated evaluation outputs. Keep only if needed for local comparison.

## Validation

Validation performed after the cleanup pass:
- Backend health/startup check: passed with `200 {'status': 'ok', 'service': 'gp-medical-api'}`.
- Backend API integration subset: passed, `5 passed`.
- Frontend production build: passed with `npm run build` in `frontend_next/`.

Notes:
- The first frontend build attempt compiled successfully but failed at TypeScript with `spawn EPERM` under the restricted sandbox. The same command passed when rerun with subprocess execution allowed.
- `frontend_next/.next/` was deleted again after the successful build because it is generated output.
