# Pipeline Evaluation

This evaluation covers the full diagnosis path the user experiences:

`user text -> parser -> validator -> normalizer -> rules -> classifier -> RAG -> fusion -> final diagnosis -> safety metadata`

It differs from standalone classifier or RAG evaluation because it measures the combined behavior of parsing, scope handling, rule safety, model agreement, fusion confidence, clarification, and final response metadata.

## Active Artifacts

- RAG / FAISS: `backend/artifacts/artifacts/faiss_data_targeted`
- Classifier: `backend/artifacts/artifacts/clinicalbert_classifier_targeted`
- Current label universe: 49 DDXPlus-derived pathologies

Diabetes/hyperglycemia and UTI/cystitis are outside this AI label universe and appear only as safety probes.

## Case Sets

Cases live under `data/evaluation/pipeline_diagnostics/cases/`:

- `pipeline_in_scope_cases.json`
- `pipeline_natural_text_cases.json`
- `pipeline_arabic_cases.json`
- `pipeline_noisy_typo_cases.json`
- `pipeline_ambiguous_cases.json`
- `pipeline_out_of_scope_safety_cases.json`

Each case includes `case_id`, `input_text`, expected label or family, scope, language, required safety behavior, and notes.

## Commands

Run the full offline evaluation:

```powershell
python backend/scripts/evaluate_pipeline_quality.py --pretty
```

Run without failing CI thresholds:

```powershell
python backend/scripts/evaluate_pipeline_quality.py
```

Run with threshold enforcement:

```powershell
python backend/scripts/evaluate_pipeline_quality.py --fail-on-threshold
```

Run the health check:

```powershell
python backend/scripts/pipeline_health_check.py --pretty
```

Quick artifact-only health check:

```powershell
python backend/scripts/pipeline_health_check.py --skip-model-load --pretty
```

## Outputs

The evaluator writes:

- `data/evaluation/pipeline_diagnostics/pipeline_architecture_report.md`
- `data/evaluation/pipeline_diagnostics/pipeline_architecture_report.json`
- `data/evaluation/pipeline_diagnostics/pipeline_eval_summary.json`
- `data/evaluation/pipeline_diagnostics/pipeline_eval_cases.csv`
- `data/evaluation/pipeline_diagnostics/pipeline_eval_report.md`
- `data/evaluation/pipeline_diagnostics/pipeline_failure_analysis.md`
- `data/evaluation/pipeline_diagnostics/pipeline_safety_report.md`

## Metrics

In-scope metrics:

- final diagnosis Top-1 accuracy
- Top-3 accuracy
- expected label/family hit rate
- classifier, RAG, and rules agreement rates
- fusion source contribution
- clarification-needed rate
- low-confidence rate
- abstention/safe fallback rate

Out-of-scope metrics:

- correct low-confidence or safe handling rate
- unsafe confident in-scope diagnosis rate
- RAG/classifier domination count
- professional-care or safe-fallback rate

Arabic, noisy, and natural text metrics:

- parser success rate
- normalization success rate
- final diagnosis hit rate
- safety fallback rate

## Thresholds

Thresholds are optional and only enforced with `--fail-on-threshold`.

- In-scope Top-1 >= 0.70
- In-scope Top-3 >= 0.85
- Out-of-scope safe handling >= 0.90
- Parser success >= 0.85
- Unsafe confident diagnosis rate <= 0.05

## Safety Limitations

This is an evaluation harness for AI-assisted decision support, not clinical validation. It cannot prove medical safety. Out-of-scope cases should avoid confident in-scope diagnoses, surface low confidence or safety fallback behavior, and advise professional care when appropriate.

## Known Failure Modes

- Parser misses rare labels that have no direct symptom alias.
- Arabic input can be limited when offline translation is disabled.
- Typo/noisy text may reach classifier/RAG through raw text while parser metrics still fail.
- Generic symptom rules can be too broad and require fusion calibration.
- Out-of-scope cases can look similar to supported respiratory, cardiac, or GI labels.

## Latest Safety/Parser Fix Run

Run command:

```powershell
python backend/scripts/evaluate_pipeline_quality.py --run-label pipeline_safety_parser_fix --pretty
```

Latest metrics:

- In-scope Top-1: 0.8125
- In-scope Top-3: 1.000
- Expected label/family hit: 1.000
- Parser success overall: 0.8974
- Natural/Arabic/noisy parser success: 0.875
- Out-of-scope safe handling: 1.000
- Unsafe confident diagnosis rate: 0.000
- Failed cases: 0

The fix run added deterministic typo normalization, deterministic Arabic cue expansion, unsupported emergency scope gating, confidence calibration metadata, and feature-based promotion for PE, pneumonia-family, stable angina, and acute dystonic reaction contexts.

## Model Update Assessment

### A. No retraining/rebuild needed.

Retraining is not currently necessary by default. FAISS rebuild is not currently necessary by default. The first fixes should target parser aliases, normalization, safety gating, fusion thresholds, and clarification logic.

### B. Retraining/rebuild recommended later.

Consider retraining or FAISS rebuild later if corrected parser/fusion/scope behavior still leaves systematic in-scope misses, or if additional validated DDXPlus-style cases become available.

### C. Retraining/rebuild strongly recommended.

Retraining and FAISS rebuild become strongly recommended if the supported label universe changes, new diseases such as diabetes or UTI are moved in-scope, or the indexed corpus no longer matches the classifier label universe.
