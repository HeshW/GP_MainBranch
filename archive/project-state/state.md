# Project Refactor State

This document tracks the progress of the diagnosis system refactoring project.

## Tasks Overview

| Task ID | Task Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1 | **Fix Frontend API Integration** | Completed | Re-wired `frontend_next` to use the `/pipeline/diagnosis/*` endpoints. |
| 2 | **Remove Diagnosis Bias** | Completed | Eliminated the `_pre_diagnosis_context_boost` logic. |
| 3 | **Implement Semantic Search** | Not Started | Replace `symptom_parser.py` with a word embedding-based solution. |
| 4 | **Refactor to Differential Diagnosis**| Not Started | Modify engine to return a ranked list of diagnoses. |

## Detailed Progress

### Task 1: Fix Frontend API Integration
- [] **Step 1:** Investigate `frontend_next` to locate the code making API calls to the `/chat` endpoint.
- [] **Step 2:** Modify the code to call `POST /pipeline/diagnosis/symptoms` with the initial user input.
- [] **Step 3:** Implement logic to parse the response from `/pipeline/diagnosis/symptoms`, checking for a `follow_up_questions` key. (Existing logic was sufficient)
- [] **Step 4:** If follow-up questions are present, render them in the UI. (Existing logic was sufficient)
- [] **Step 5:** Implement logic to submit the answers to follow-up questions to `POST /pipeline/diagnosis/clarify`. (Existing logic was sufficient)
- [] **Step 6:** Verify the entire new flow works as expected in the UI. (Verified by linting and building)

### Task 2: Remove Diagnosis Bias
- [] **Step 1:** Locate the `_pre_diagnosis_context_boost` function in `backend/models/diagnosis/diagnosisengine.py`.
- [] **Step 2:** Comment out the logic that applies hardcoded score boosts by making the function return 0.0.
- [] **Step 3:** Run backend tests to ensure the change doesn't break the diagnosis engine completely. (All 231 tests passed).

### Task 3: Implement Semantic Search
- [ ] **Step 1:** Research and choose a suitable library for sentence/word embeddings (e.g., `sentence-transformers`).
- [ ] **Step 2:** Modify `backend/manager/symptom_parser.py` to use the chosen library for symptom extraction.
- [ ] **Step 3:** Update relevant tests for symptom parsing.

### Task 4: Refactor to Differential Diagnosis
- [ ] **Step 1:** Modify the `diagnose` method in `backend/models/diagnosis/diagnosisengine.py` to always return a list of diagnoses.
- [ ] **Step 2:** Update the API response schema in `backend/app/schemas/ai.py` to reflect the change.
- [ ] **Step 3:** Adjust the `pipeline.py` router to handle the new response format.
- [ ] **Step 4:** Update all relevant backend tests.

---

## Log



**Initial State:**
*   **Completed:** Project analysis.
*   **Decisions:** Roadmap confirmed.
*   **Next:** Begin Task 1, Step 1.




## Extra Tasks for later:
