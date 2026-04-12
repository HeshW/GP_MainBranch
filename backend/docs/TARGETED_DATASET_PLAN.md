# Targeted Dataset Plan

## What We Need

To improve diagnosis accuracy in a meaningful way, we need a curated dataset that
matches the actual failure mode of the project:

- natural free-text symptom descriptions
- in-scope diseases only
- ambiguity between clinically similar conditions
- gold follow-up answers
- final expected diagnosis

## Why This Is Better Than Generic External Data

- It matches our supported 49-disease scope.
- It matches our interactive clarification workflow.
- It avoids label-mapping chaos from unrelated datasets.
- It directly targets the cases where end-to-end accuracy currently fails.

## Recommended Fields Per Case

- `id`
- `language`
- `difficulty`
- `ambiguity_group`
- `raw_text`
- `expected_conditions`
- `follow_up_answers`
- `discriminative_symptoms`
- `negated_symptoms`
- `notes`

## Starter Template

- [targeted_cases_template.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/data/evaluation/targeted_cases_template.json:1)

## Validation Tool

- [validate_targeted_cases.py](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/scripts/validate_targeted_cases.py:1)

Example:

```powershell
.\.venv_rag\Scripts\python.exe backend\scripts\validate_targeted_cases.py `
  --cases data\evaluation\targeted_cases_template.json
```

## Best Case Types To Add First

1. Atrial fibrillation vs PSVT
2. Pneumonia vs bronchospasm
3. Stable angina vs unstable angina
4. Pulmonary embolism vs panic / nonspecific cardiopulmonary rule pattern
5. Myasthenia gravis vs Guillain-Barre syndrome
6. GERD vs cardiopulmonary chest pain rule patterns

## Recommended Build Order

1. Add 2-3 cases for each high-confusion pair.
2. Mix English, Arabic, and mixed phrasing.
3. Add explicit negation and discriminative follow-up details.
4. Validate the file with the script.
5. Run end-to-end evaluation on the curated set.
6. Use results to guide reranking or retraining.
