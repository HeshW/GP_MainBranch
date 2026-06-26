# Gemini Plan: Mitigating Over-Diagnosis in the Medical AI System

## 1. Executive Summary

The core problem is that the system is architecturally and algorithmically biased towards finding a "best match" for a diagnosis, even from weak or ambiguous evidence. This leads to dangerous over-diagnosis of rare conditions (e.g., myocarditis) for common, vague symptoms (e.g., "fatigue and thirst").

This plan outlines a fundamental shift in the system's philosophy: from a **"forced-choice predictor"** to a **"conservative differential diagnosis assistant."**

## 2. The Strategic Shift: From Additive to Subtractive Logic

*   **Current Flawed Model (Additive):** The system collects weak positive signals (like "fatigue" for myocarditis) and adds them up. A rare disease can "win" if it accumulates enough minor points, even if its key red-flag symptoms are absent.
*   **Proposed New Model (Prevalence-Weighted & Subtractive):** The new system will start by assuming common conditions are more likely (a prevalence "prior"). A rare disease must then overcome this initial handicap by providing strong, specific evidence. Crucially, the system will heavily penalize or disqualify a rare diagnosis if its *required* symptoms are absent.

## 3. Phased Implementation Plan

This plan is broken into three phases to deliver the most critical safety improvements first.

### Phase 1: Foundational Rearchitecture & Logic Overhaul

**Goal:** Immediately stop the most dangerous over-diagnoses by fundamentally changing the ranking algorithm and the API contract.

#### Step 1.1: Centralize Clinical Knowledge

Separate clinical knowledge (prevalence, key features) from the engine's orchestration logic. This makes the system easier to audit, update, and reason about.

**Action:** Create a new file `backend/models/diagnosis/condition_profiles.yaml`.

```yaml
# Centralized clinical knowledge profiles for the diagnosis engine.
# This file separates clinical data from Python logic.

# commonness: A categorical representation of prevalence.
#   - common: >5%
#   - uncommon: 1-5%
#   - rare: 0.1-1%
#   - very_rare: <0.1%
# required_features: A set of symptoms, one of which MUST be present for the
#   condition to be seriously considered. Absence leads to a heavy penalty.
# specific_signals: Symptoms that are highly suggestive of this condition.

myocarditis:
  commonness: very_rare
  required_features:
    - chest pain
    - shortness of breath
    - dyspnea
    - palpitations
  specific_signals:
    - viral prodrome
    - worse when lying down
    - myalgia

dehydration:
  commonness: common
  required_features: [] # No single required feature
  specific_signals:
    - thirst
    - thirsty
    - dry mouth
    - reduced intake
    - dark urine

urti: # Upper Respiratory Tract Infection
  commonness: common
  required_features: []
  specific_signals:
    - sore throat
    - nasal congestion
    - runny nose
    - cough
```

#### Step 1.2: Refactor the API Schema for Differential Diagnosis

The current API forces a single "final" diagnosis. We must change this contract to support a ranked list of possibilities and communicate uncertainty to the frontend.

**Action:** Modify `backend/app/schemas/ai.py` to match the already implemented changes. The current schema with `AssessmentState` and `DifferentialDiagnosis` is correct and serves as the foundation for this plan. No further changes are needed if it matches the version from the last analysis.

#### Step 1.3: Overhaul the Diagnosis Engine Ranking Logic

This is the most critical step. We will replace the additive scoring with the new prevalence-weighted, subtractive model.

**Action:** Modify `backend/models/diagnosis/diagnosisengine.py`.

1.  **Load the new profiles:** The existing `load_condition_profiles` function is correct.
2.  **Replace ranking logic:** The core change is in `_rerank_base_candidates`. The old version uses a complex series of additive boosts. The new version must be replaced with a function that performs these steps in order:
    *   Start with the base confidence from the AI models (classifier, RAG).
    *   Apply a **prevalence penalty** based on the `commonness` field from `condition_profiles.yaml`. Rare diseases get a significant negative adjustment.
    *   Apply a **required feature gate**. If a condition in the YAML file has `required_features` and none are present in the patient's symptoms, apply a heavy penalty or cap its confidence at a very low value (e.g., 0.10).
    *   Apply a smaller bonus for `specific_signals` if they are present.

**Proposed Implementation for `_rerank_base_candidates` (to replace the existing one):**

```python
    # In DiagnosisEngine class

    @classmethod
    def _get_condition_profile(cls, label: str) -> dict:
        """Safely get a condition profile by normalized label."""
        normalized_label = cls._normalize_label(label)
        return cls.CONDITION_PROFILES.get(normalized_label, {})

    @classmethod
    def _apply_prevalence_penalty(cls, label: str, confidence: float) -> float:
        """Applies a penalty based on the 'commonness' category from the profile."""
        profile = cls._get_condition_profile(label)
        commonness = profile.get("commonness", "uncommon")

        prevalence_weights = {
            "common": 0.0,        # No penalty
            "uncommon": -0.15,    # Moderate penalty
            "rare": -0.25,        # Strong penalty
            "very_rare": -0.40,   # Very strong penalty
        }
        penalty = prevalence_weights.get(commonness, -0.20)
        return max(0.01, confidence + penalty)

    @classmethod
    def _has_required_features(cls, label: str, patient_symptoms: list[str]) -> bool:
        """Checks if at least one of the required features for a condition is present."""
        profile = cls._get_condition_profile(label)
        required = profile.get("required_features")
        if not required:
            return True  # No required features defined, so it passes.

        normalized_symptoms = {cls._normalize_label(s) for s in patient_symptoms if str(s).strip()}
        return any(feature in normalized_symptoms for feature in required)

    @classmethod
    def _rerank_base_candidates(
        cls,
        *,
        candidates: list[Dict[str, Any]],
        patient_symptoms: list[str],
        # other params...
    ) -> list[Dict[str, Any]]:
        reranked: list[Dict[str, Any]] = []
        for item in candidates:
            candidate = dict(item)
            label = str(candidate.get("label", "")).strip()
            base_confidence = cls._normalize_confidence(candidate.get("confidence"))

            # 1. Start with base confidence
            adjusted_confidence = base_confidence

            # 2. Apply strong penalty for low prevalence.
            adjusted_confidence = cls._apply_prevalence_penalty(label, adjusted_confidence)

            # 3. Apply a heavy penalty if required features are missing.
            if not cls._has_required_features(label, patient_symptoms):
                adjusted_confidence -= 0.50 # Heavy subtractive penalty

            # (Optional) Add smaller bonus for specific signals
            # ...

            candidate["confidence"] = max(0.01, min(round(adjusted_confidence, 4), 0.98))
            reranked.append(candidate)

        return sorted(reranked, key=lambda item: item.get("confidence", 0.0), reverse=True)
```

### Phase 2: Evidence & Clarification Refinement

**Goal:** Improve the quality of the initial evidence and make the clarification loop smarter.

#### Step 2.1: Add Baseline Symptom Rules

The `temproadmap.md` correctly notes that `dehydration` is missing as a rule. We will add it and expand its trigger keywords.

**Action:** Modify `backend/models/diagnosis/rules.py` to ensure a rule for dehydration exists and is not duplicated. The current file has two identical rules for dehydration; one should be removed.

#### Step 2.2: Refine Clarification Logic

The `_build_clarification` function will now operate on a much more reliable list of candidates from the new reranking logic. This means it will naturally ask better questions (e.g., differentiating a cold from dehydration) instead of getting stuck on two rare diseases. No immediate code change is required here, as the improved input is the primary fix.

### Phase 3: Frontend Integration

**Goal:** Ensure the user interface safely and clearly presents the new differential diagnosis.

#### Step 3.1: Update API Client and State Management

The frontend needs to be aware of the new `AIDiagnosisResponse` schema.

**Action:** The `postSymptoms` and `postClarification` functions in `frontend/src/shared/api/client.ts` will now receive the new `AIDiagnosisResponse` object. The state management hook (`useAnalysis.ts`) must be updated to store this new structure, including the `assessment_state` and the `differential_diagnosis` array.

#### Step 3.2: Redesign the Results UI

The UI must stop overstating certainty.

**Action:** Modify `ResultView.tsx` and `UserInterfaceView.tsx`.
*   **Check `assessment_state`:**
    *   If `assessment_state` is `"needs_clarification"`, the main heading should be **"Preliminary Assessment"** or **"Possible Conditions"**, not "Final Diagnosis".
    *   If `assessment_state` is `"final"`, it can display **"Most Likely Condition"**.
*   **Display the Differential Diagnosis:**
    *   Render the `differential_diagnosis` array as a ranked list.
    *   For each item, show the `condition` and a visual representation of its `confidence` (e.g., a confidence bar).
    *   The `assessment_summary` from the API should be displayed prominently to explain the overall picture.

## 4. Testing Strategy

1.  **Vague Symptom Benchmark:** Create a new test file `backend/tests/test_overdiagnosis_regression.py` with inputs like `"I feel tired and thirsty"`.
    *   **Metric 1 (Alarmism Rate):** Assert that for this input, `myocarditis` is **NOT** the top-ranked diagnosis in the returned `differential_diagnosis`.
    *   **Metric 2 (Top-3 DDx Accuracy):** Assert that `Dehydration` or `Acute viral illness` **IS** present in the top 3 of the `differential_diagnosis`.
2.  **Required Feature Gating Test:** Create a test case for `"chest pain and fatigue after a viral illness"`.
    *   Assert that `myocarditis` **CAN** appear high in the DDx in this case, confirming we haven't lost sensitivity for legitimate presentations.
3.  **UI State Test:** In the frontend tests, simulate an API response where `assessment_state` is `"needs_clarification"`.
    *   Assert that the text "Final Diagnosis" is **NOT** rendered in the DOM.