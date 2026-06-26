No files were modified. I inspected the tracked source, generated/runtime files, docs, and the diagnosis path. I also attempted to run the example symptom flow, but local execution failed before diagnosis because the environment cannot import `google.genai`, even though `google-genai` is listed in requirements.

**1. Codebase Map**

Backend flow:
`FastAPI routers` -> `ChatManager` -> preprocessing/OCR -> `DiagnosisEngine` -> therapy/chat response.

Key connections:
- API entrypoint: [main.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/app/main.py:58) creates one `ChatManager` at startup.
- Pipeline routes: [pipeline.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/app/routers/pipeline.py:72) exposes labs, image, symptoms, diagnosis, clarification.
- Orchestration: [chat_manager.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/chat_manager.py:27) owns diagnosis, therapy, chat sessions, and lazy OCR.
- Symptom text path:
  [symptom_parser.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/symptom_parser.py:394) -> [symptom_validator.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/symptom_validator.py:69) -> [symptom_normalizer.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/symptom_normalizer.py:91) -> [pipeline_support.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/pipeline_support.py:70).
- Diagnosis core:
  [rules.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/rules.py:112) provides lab/symptom rules.
  [diagnosisengine.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/diagnosisengine.py:27) fuses rules, RAG, classifier, candidate expansion, clarification, safety, and final summary.
  [rag.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/rag.py:78) handles ClinicalBERT/FAISS search and classifier loading.
- Frontend:
  [client.ts](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/frontend/src/shared/api/client.ts:103) calls symptom API.
  [useAnalysis.ts](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/frontend/src/features/analysis/hooks/useAnalysis.ts:60) stores pipeline result.
  [ResultView.tsx](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/frontend/src/features/results/components/ResultView.tsx:49) leads with “Final Diagnosis”.
  [UserInterfaceView.tsx](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/frontend/src/features/user/components/UserInterfaceView.tsx:52) says “Likely condition”.

**2. Redundancy Candidates**

- [diagnosisengine.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/diagnosisengine.py:27): not removable, but structurally overloaded. It contains constants, scoring, candidate expansion, clarification, final selection, safety, canonicalization, and synthesis orchestration. Split, do not delete.
- [symptom_parser.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/symptom_parser.py:11), [symptom_normalizer.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/symptom_normalizer.py:34), and [rag.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/rag.py:86) duplicate symptom vocabulary. Merge into a single symptom taxonomy/config.
- [diagnosis_adapter.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/diagnosis_adapter.py:1) and [manager_tester.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/manager_tester.py:1) overlap as CLI/adaptation helpers. Keep one CLI entrypoint; keep adapter only if external imports rely on it.
- `GP_(1) (1).ipynb`, `backend/models/diagnosis/diagnosisproto.ipynb`, `implementation_plan.md`, `merge_checklist.md` are tracked but currently deleted in the worktree. They appear to be legacy/prototype artifacts; do not resurrect unless needed.
- Docs are partly stale: [AI_Architecture_Evaluation.md](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/docs/AI_Architecture_Evaluation.md:122) says there is no fusion layer, but fusion now exists. Merge/update architecture docs.
- `backend/faiss_data/medical_cases.index` is tracked while metadata/artifacts are ignored. This is an incomplete artifact pattern. Move all model/index assets under `backend/artifacts/` or external artifact storage.

**3. Architecture Weaknesses**

- Loose dict contracts between parser, manager, diagnosis, and frontend make clinical state ambiguous. Use internal Pydantic/domain models for `SymptomEvidence`, `LabEvidence`, `CandidateDiagnosis`, `DifferentialResult`.
- `DiagnosisEngine` is a monolith. Extract `CandidateGenerator`, `EvidenceScorer`, `FusionRanker`, `ClarificationPlanner`, `SafetyTriage`.
- Confidence values are mixed scales: retrieval similarity, classifier probability, rule confidence strings, and handcrafted boosts are treated as comparable.
- Provider import is too eager: [ai_provider.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/common/ai_provider.py:13) imports `google.genai` at module import time, breaking even no-key fallback if the package is missing.
- UI overstates certainty. Both [ResultView.tsx](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/frontend/src/features/results/components/ResultView.tsx:49) and [UserInterfaceView.tsx](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/frontend/src/features/user/components/UserInterfaceView.tsx:52) foreground a single likely/final diagnosis before clarification is complete.

**4. Diagnosis Deep Dive**

Current flow:

```text
User text
 -> parse exact/fuzzy symptoms + labs
 -> validate symptoms/labs
 -> normalized text with DDX-style cue questions
 -> report {raw_text, symptoms, labs}
 -> lab rules + symptom rules
 -> optional RAG retrieval + optional classifier
 -> collect candidates
 -> expand handcrafted candidate families
 -> rerank by confidence + signal matches + boosts
 -> choose final diagnosis
 -> build clarification after final diagnosis
 -> frontend displays final/likely diagnosis
```

Root causes of over-diagnosis:
- The parser recognizes `"thirst"` but not `"thirsty"` in [symptom_parser.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/manager/symptom_parser.py:58). Input like `tired/thirsty` can become only `fatigue`, losing the hyperglycemia/dehydration path.
- Dehydration is not represented as a symptom-rule candidate at all. Hyperglycemia requires the right symptoms; cold/flu requires fever/cough/headache/sore throat clusters.
- `myocarditis` has `"fatigue"` as a positive signal in [diagnosisengine.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/diagnosisengine.py:288), without requiring chest pain, dyspnea, palpitations, troponin, or clear viral cardiac context.
- RAG reranking has a myocarditis penalty only for some mismatched sudden/unilateral contexts, not for absence of core myocarditis features: [rag.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/rag.py:347).
- Candidate scoring is additive, not Bayesian: [diagnosisengine.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/diagnosisengine.py:1159). Rare/high-risk labels are not prior-penalized.
- Clarification is built after `final_diagnosis`: [diagnosisengine.py](/C:/Users/Hesh/Desktop/GP/GPProject/GP_MainBranch/backend/models/diagnosis/diagnosisengine.py:2972). It asks follow-up questions, but the UI has already shown the alarming diagnosis.

**Proposed Flow**

```text
User text
 -> evidence extraction with synonyms, negations, duration, severity
 -> EvidenceFrame
 -> candidate generation:
      common baseline candidates + rule candidates + RAG/classifier candidates
 -> calibrated ranking:
      prior/prevalence + symptom likelihood + absence penalties + lab evidence
 -> safety triage:
      urgent red flags are shown, but do not inflate likelihood
 -> if uncertain/high-risk-under-supported:
      show differential + ask clarification before single final label
 -> after answers:
      rerank full differential, including common candidates
 -> output:
      likely/common possibilities, serious-not-to-miss, confidence, next steps, disclaimers
```

**Code-Level Changes Needed**

Introduce local clinical knowledge, no external KB required:

```python
@dataclass
class ConditionProfile:
    label: str
    prior: float              # local prevalence tier converted to log prior
    commonness: str           # common / uncommon / rare
    required_features: set[str]
    supportive_features: set[str]
    negative_if_absent: set[str]
    red_flags: set[str]
    safety_level: str
```

Add `backend/models/diagnosis/condition_profiles.yaml` with profiles for dehydration, viral illness/URTI, influenza-like illness, diabetes/hyperglycemia, anemia, myocarditis, PE, pneumonia, etc.

Replace final ranking logic with calibrated scoring:

```python
score = log(profile.prior)
score += sum(weight(symptom, condition) for present symptoms)
score += sum(lab_likelihood(lab, condition) for labs)
score -= missing_required_feature_penalty(condition, evidence)
score -= rare_serious_under_support_penalty(condition, evidence)
safety_alert = has_red_flags(condition, evidence)
```

Key edits:
- Expand symptom aliases: `thirsty`, `dry mouth`, `reduced intake`, `dark urine`, `body aches`, `runny nose`.
- Add common candidate generator before AI candidates.
- Move serious-condition gating before final label selection.
- Make `clarification.needed=True` suppress “final diagnosis” wording and return `assessment_state="needs_clarification"`.
- Frontend: rename “Final Diagnosis” to “Assessment” unless `assessment_state == "final"`, and show differential first.

**Testing Strategy**

- Parser tests: `tired/thirsty for a week` must extract `fatigue` and `thirst`.
- Golden diagnosis tests:
  - fatigue + thirsty + one week -> dehydration/hyperglycemia/viral illness differential, not myocarditis as final.
  - fatigue + chest pain + shortness of breath + palpitations/recent viral illness -> myocarditis can appear as serious differential.
  - vague single symptom -> no final diagnosis; ask clarification.
- Calibration tests: rare serious diagnoses require required-feature support or remain “not-to-miss”.
- UI tests: when clarification is needed, no “Likely condition: myocarditis” text appears.
- End-to-end evaluation: measure top-3 differential accuracy, dangerous false-positive rate, dangerous false-negative rate, and clarification usefulness.

**Roadmap**

1. Fix symptom vocabulary and add dehydration/common illness candidates.
2. Add `assessment_state` and stop showing one final label when clarification is required.
3. Extract condition profiles and calibrated ranker from `diagnosisengine.py`.
4. Add priors, required-feature penalties, and serious-condition support gates.
5. Refactor UI to display differential diagnosis with confidence bands.
6. Consolidate docs and CLI helpers.
7. Add regression/evaluation suite for common-symptom over-diagnosis.