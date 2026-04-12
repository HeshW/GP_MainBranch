# Branch Merge Analysis: `hesh-edits2` vs `mego-final`

> **Generated:** 2026-04-12  
> **Repository:** `HeshW/GP_MainBranch`  
> **Base of comparison:** `origin/mego-final` ← compared against → `origin/hesh-edits2`

---

## Table of Contents

1. [Executive Summary](#executive-summary)  
2. [File-Level Differences](#file-level-differences)  
3. [Diagnosis Model & Pipeline Differences](#diagnosis-model--pipeline-differences)  
4. [Fixes, UI/UX Changes, and Improvements in `hesh-edits2`](#fixes-uiux-changes-and-improvements-in-hesh-edits2)  
5. [Potential Merge Conflicts](#potential-merge-conflicts)  
6. [Branch Strengths Summary](#branch-strengths-summary)  
7. [Actionable Merge Recommendations](#actionable-merge-recommendations)

---

## Executive Summary

| Dimension | `mego-final` | `hesh-edits2` |
|---|---|---|
| Diagnosis engine size | ~2,715 lines — richly annotated, high-complexity | ~1,154 lines — cleaned, focused |
| Diagnosis accuracy emphasis | More aggressive multi-signal, multi-threshold tuning | Simplified thresholds, focuses on reliability |
| UI/UX | Single-mode chat-style interface | Dual-mode: patient "User Interface" + "Dev Workbench" |
| Security hardening | Minimal | API-key auth, upload-size limits, hash-verified pickle |
| Session management | Basic | TTL expiry, LRU eviction, max-session cap |
| Chat support language | Garbled/corrupted Arabic strings | Fixed: auto-detects Arabic/English, correct strings |
| API dependencies | Flexible ranges (`>=`) | Pinned versions for reproducibility |
| Targeted training data | Large CSV training sets committed to repo | Removed (data files excluded from VCS) |
| Test coverage | Legacy tests for removed scripts | New comprehensive suite (security, integration, etc.) |
| Gemini model default | `gemini-2.5-flash` | `gemini-2.5-flash-lite` (lower cost) |

**Bottom line:**  
`mego-final` contains the more powerful diagnostic brain (richer clarification bank, pathology penalties, pre-diagnosis signal weighting). `hesh-edits2` is the more production-ready branch with security, UX, stability fixes, and a clean dependency story. The ideal final state is **merging `hesh-edits2` into `mego-final`** (or the other way around) so the refined diagnosis logic of `mego-final` is preserved while all reliability/security/UI improvements of `hesh-edits2` are applied on top.

---

## File-Level Differences

### Files Added in `hesh-edits2` (not in `mego-final`)

| File | Purpose |
|---|---|
| `backend/requirements-ai.txt` | Pinned AI/RAG/classifier deps (faiss-cpu 1.7.4, torch, transformers, datasets) with explicit NumPy ABI compatibility note |
| `backend/docs/Fresh_Machine_Setup.md` | Step-by-step new-machine setup guide |
| `backend/tests/test_ai_provider.py` | Unit tests for GeminiProvider |
| `backend/tests/test_api_integration.py` | End-to-end API integration tests |
| `backend/tests/test_chat_support.py` | Tests for language detection and message building |
| `backend/tests/test_check_env_mismatch.py` | Tests for env-check script |
| `backend/tests/test_diagnosis_adapter.py` | Tests for `adapt_and_diagnose` wrapper |
| `backend/tests/test_diagnosis_synthesis.py` | Tests for synthesis layer |
| `backend/tests/test_manager.py` | Tests for ChatManager |
| `backend/tests/test_pipeline_upload_limits.py` | Tests enforcing 10 MB upload cap |
| `backend/tests/test_rag.py` | Tests for RAG/FAISS metadata loading including pickle hash-verification |
| `backend/tests/test_security_baseline.py` | Tests for API-key enforcement and injection-path safety |
| `backend/tests/test_session_store.py` | Tests for TTL expiry and LRU eviction |
| `backend/tests/test_startup_guardrails.py` | Tests for graceful degraded-mode startup |
| `backend/tests/test_therapy_engine.py` | Tests for TherapyEngine fallback paths |
| `frontend/src/App.smoke.test.tsx` | Smoke test for App render |
| `frontend/src/features/chat/components/ChatInterface.test.tsx` | Tests for standalone ChatInterface |
| `frontend/src/features/user/components/UserInterfaceView.tsx` | New patient-facing unified UI view |
| `frontend/src/features/user/components/index.ts` | Re-export barrel |
| `frontend/src/features/user/index.ts` | Feature-level re-export |
| `frontend/vitest.config.ts` | Vitest configuration |

### Files Deleted in `hesh-edits2` (exist only in `mego-final`)

| File | Why removed |
|---|---|
| `backend/docs/ACCURACY_DATA_WORKFLOW.md` | Superseded / research artefact |
| `backend/docs/RETRAINING_WORKFLOW.md` | Superseded |
| `backend/docs/TARGETED_DATASET_PLAN.md` | Superseded |
| `backend/scripts/ablate_rag_frontier.py` | Research script, not production |
| `backend/scripts/build_targeted_training_csv.py` | Targeted training removed |
| `backend/scripts/check_release_gates.py` | Superseded by `check_unified_release_gates.py` → also removed |
| `backend/scripts/check_unified_release_gates.py` | Superseded |
| `backend/scripts/expand_targeted_cases.py` | Targeted training removed |
| `backend/scripts/generate_augmented_ddxplus_data.py` | Targeted training removed |
| `backend/scripts/grid_search_diagnosis_thresholds.py` | Research only |
| `backend/scripts/merge_training_csvs.py` | Targeted training removed |
| `backend/scripts/report_ambiguity_group_deltas.py` | Research only |
| `backend/scripts/validate_targeted_cases.py` | Targeted training removed |
| `backend/tests/test_check_release_gates.py` | Gates script removed |
| `backend/tests/test_check_unified_release_gates.py` | Gates script removed |
| `backend/tests/test_generate_augmented_ddxplus_data.py` | Script removed |
| `backend/tests/test_report_ambiguity_group_deltas.py` | Script removed |
| `backend/tests/test_symptom_normalizer.py` | Normalizer simplified/removed |
| `backend/tests/test_symptom_parser.py` | Parser simplified |
| `backend/tests/test_targeted_training_pipeline.py` | Targeted training removed |
| `backend/tests/test_validate_targeted_cases.py` | Script removed |
| `frontend/src/features/analysis/hooks/useDiagnosticChat.ts` | Replaced by `useAnalysis` hook |
| `frontend/src/features/results/components/ReviewerPanel.tsx` | Replaced by `ResultView` improvements |
| `targeted_training/test_merged.csv` | Data files removed from VCS |
| `targeted_training/test_targeted.csv` | Data files removed from VCS |
| `targeted_training/train_merged.csv` | Data files removed from VCS |
| `targeted_training/train_targeted.csv` | Data files removed from VCS |
| `targeted_training/validate_merged.csv` | Data files removed from VCS |
| `targeted_training/validate_targeted.csv` | Data files removed from VCS |

### Files Modified in Both Branches (conflict candidates)

All 55+ modified files below have diverged and **all** are potential merge conflict sites. The high-risk subset is identified in [§ Potential Merge Conflicts](#potential-merge-conflicts).

`backend/app/main.py`, `backend/app/config.py`, `backend/app/deps.py`, `backend/app/routers/pipeline.py`, `backend/app/routers/chat.py`, `backend/app/routers/health.py`, `backend/manager/chat_manager.py`, `backend/manager/chat_support.py`, `backend/manager/diagnosis_adapter.py`, `backend/manager/session_store.py`, `backend/manager/symptom_parser.py`, `backend/models/common/ai_provider.py`, `backend/models/diagnosis/diagnosisengine.py`, `backend/models/diagnosis/rag.py`, `backend/models/diagnosis/synthesis.py`, `backend/models/diagnosis/text.py`, `backend/models/ocr/engine.py`, `backend/models/ocr/parsing.py`, `backend/models/therapy/engine.py`, `backend/requirements.txt`, `backend/requirements-runtime.txt`, `backend/requirements-test.txt`, `backend/scripts/check_env_mismatch.py`, `backend/scripts/evaluate_pipeline_end_to_end.py`, `backend/scripts/train_clinicalbert_classifier.py`, `backend/tests/test_diagnosis_engine.py`, `backend/tests/test_evaluate_pipeline_metrics.py`, `backend/tests/test_manager.py`, `backend/tests/test_rag.py`, `backend/tests/test_therapy_engine.py`, `frontend/src/App.tsx`, `frontend/src/features/chat/components/ChatInterface.tsx`, `frontend/src/features/results/components/ResultView.tsx`, `frontend/src/index.css`, `frontend/src/shared/api/client.ts`, `frontend/src/shared/layout/AppHeader.tsx`, `frontend/src/shared/types/index.ts`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/tsconfig.json`, `frontend/vite.config.ts`, `notebooks/Colab_All_In_One_Natural_AI.ipynb`, `requirements.txt`, `.gitignore`, `README.md`, `backend/.env.example`

---

## Diagnosis Model & Pipeline Differences

### `mego-final` — Richer, More Accurate Diagnosis Brain

`mego-final`'s `diagnosisengine.py` is **2,715 lines** vs `hesh-edits2`'s **1,154 lines**. The extra ~1,560 lines in `mego-final` implement several layers of disambiguation and accuracy that are **absent** in `hesh-edits2`:

#### 1. Advanced Clarification Banks (removed in hesh-edits2)

```
CLARIFICATION_PAIR_BANK  — 6 condition-pair clarification entries, e.g.:
  • "AF vs PSVT" — irregular vs sudden-onset palpitations
  • "Pneumonia vs Asthma/Bronchospasm" — infection signs vs wheeze
  • "Pneumonia vs Bronchitis" — pleuritic pain vs post-cold cough
  • "Unstable vs Stable Angina" — rest pain vs exertional
  • "PE vs Bronchitis" — sudden pleuritic vs gradual URI
  • "Myasthenia Gravis vs GBS" — fluctuating bulbar vs ascending

DIAGNOSTIC_SIGNAL_BANK — per-condition positive/negative signal lists for:
  GERD, Laryngospasm, Viral Pharyngitis, Acute Laryngitis,
  PE, Spontaneous Pneumothorax, Myocarditis, Pericarditis,
  Bronchitis, Pulmonary Neoplasm, Pancreatic Neoplasm,
  Sarcoidosis, Pneumonia, Bronchospasm/Asthma
```

These structures allow `mego-final` to **ask patients targeted yes/no follow-up questions** for ambiguous close-call diagnoses and to apply positive/negative signal scoring to narrow the differential — this is the main driver of diagnostic accuracy.

#### 2. Generic Rule Pattern Resolution (removed in hesh-edits2)

```python
GENERIC_RULE_PATTERN_DIRECT_MAP = {
    "possible gastroesophageal reflux pattern": "GERD",
    "possible anemia related symptom pattern": "Anemia",
}
GENERIC_RULE_PATTERN_FAMILY_KEYWORDS = {
    "possible lower respiratory infection pattern": ("pneumonia", "bronchitis", ...),
    "possible acute viral illness pattern": ("influenza", "viral pharyngitis", ...),
    "possible cardiopulmonary red flag symptom pattern": ("pulmonary embolism", ...),
}
```

These mappings convert broad rule-based pattern labels into specific diagnoses or family candidates, enabling the AI to go further from a generic rule hit.

#### 3. Pre-Diagnosis Signal Weighting (removed in hesh-edits2)

```python
PRE_DIAGNOSIS_SIGNAL_WEIGHT = 0.35
PRE_DIAGNOSIS_RULE_ALIGNMENT_BONUS = 0.06
PRE_DIAGNOSIS_GENERIC_PATTERN_PENALTY = 0.04
```

A pre-pass that weights early diagnostic signals before the main AI call to prime confidence scoring.

#### 4. Clarification Override / Gain Thresholds (removed in hesh-edits2)

```python
CLARIFICATION_OVERRIDE_MARGIN = 0.05
CLARIFICATION_OVERRIDE_GAIN_THRESHOLD = 0.12
CLARIFICATION_LEADER_MARGIN = 0.05
```

These control when a clarification answer is strong enough to override the initial diagnosis, creating a more responsive post-clarification re-ranking system.

#### 5. RAG Re-rank Weight Tuning (removed in hesh-edits2)

`mego-final`'s `rag.py` exposes `configure_rerank_weights()` and `configure_search_expansion()` class methods allowing runtime adjustment of:
- `RERANK_WEIGHT_EMBEDDING` (0.50)
- `RERANK_WEIGHT_SYMPTOM_OVERLAP` (0.28)
- `RERANK_WEIGHT_LEXICAL` (0.18)
- `RERANK_WEIGHT_FEATURE_ALIGNMENT` (0.24)
- `RERANK_PENALTY_MISMATCH` (0.23)
- `RERANK_PENALTY_PATHOLOGY` (0.30)

And a per-pathology `_pathology_mismatch_penalty()` function that adds targeted penalties for unlikely diagnoses (Ebola without hemorrhagic context, Guillain-Barré without ascending weakness, etc.).

#### 6. Bronchospasm/Asthma Clarification Question (removed in hesh-edits2)

`mego-final` has an additional clarification question specifically for asthma/bronchospasm disambiguation:
> "Are symptoms mainly wheeze/chest tightness without fever or productive sputum, and do bronchodilators help?"

`hesh-edits2` removes this question, which could lead to false pneumonia/bronchitis calls when the actual diagnosis is asthma.

#### 7. Symptom Parser: More Negation Patterns (removed in hesh-edits2)

`mego-final`'s `symptom_parser.py` includes:
- Extended Arabic negation cues: `"ما في"`, `"مافي"`, `"ما عندي"`, `"ليس"`
- Sophisticated "productive cough" negation disambiguation (only negates plain cough, not productive cough, when negation pattern matches only productive form)
- Regex patterns for detecting encoded DDXPlus symptom sequences to skip non-natural-language inputs
- "ptosis" with plural form aliases (`"drooping eyelids"`, `"droopy eyelids"`) 
- "shortness of breath" with `"short of breath"` alias  

#### Threshold Differences

| Constant | `mego-final` | `hesh-edits2` |
|---|---|---|
| `CLASSIFIER_PRIMARY_THRESHOLD` | `0.52` | `0.55` |
| `CLARIFICATION_MARGIN_THRESHOLD` | `0.10` | `0.12` |
| `CLARIFICATION_OVERRIDE_MARGIN` | `0.05` | *(removed)* |
| `CLARIFICATION_OVERRIDE_GAIN_THRESHOLD` | `0.12` | *(removed)* |
| `CLARIFICATION_LEADER_MARGIN` | `0.05` | *(removed)* |
| `rag_top_k` default | `7` | `5` |

`mego-final`'s lower primary threshold (0.52) accepts classifier predictions at lower confidence, which should improve recall. The higher margin (0.12 in hesh-edits2) means fewer clarification questions are asked — this may reduce false trigger of clarification flow but could miss borderline ambiguities.

---

## Fixes, UI/UX Changes, and Improvements in `hesh-edits2`

### Backend Fixes

#### 1. Chat Support Language Corruption Fix (Critical)
`mego-final`'s `chat_support.py` contains **garbled/corrupted Arabic text** for all user-facing messages (e.g., `"??? ??????? ??? ??? ?????"`, `"?????? ???? ???????? ??? ???? ??????"`). This appears to be a character encoding corruption that occurred during a commit.

`hesh-edits2` replaces all corrupted strings with proper bilingual messages and adds automatic language detection:
```python
def detect_response_language(message: str) -> str:
    return "ar" if _contains_arabic(message) else "en"
```
Error messages, unavailable messages, and role labels now respond in the same language as the user's query.

#### 2. Startup Graceful Degradation
`hesh-edits2` wraps `ChatManager` initialization in a try/except:
- If RAG index directory doesn't exist → disables RAG silently, logs warning
- If fine-tuned classifier directory doesn't exist → disables classifier silently
- If full initialization fails → falls back to degraded mode (Gemini only) instead of crashing

#### 3. Request Audit Middleware
Every HTTP request is now logged with method, path, status code, duration, and a `X-Request-ID` header for tracing:
```python
@application.middleware("http")
async def request_audit_middleware(request, call_next): ...
```

#### 4. API Key Authentication
`hesh-edits2` adds optional service-level API key enforcement:
- `REQUIRE_SERVICE_API_KEY` environment flag
- Accepts key via `X-API-Key` header or `Authorization: Bearer <key>`
- Uses `hmac.compare_digest` to prevent timing attacks
- Returns `401 Unauthorized` when invalid
- Frontend `client.ts` reads `VITE_API_KEY` and attaches it to all requests

#### 5. Upload Size Enforcement (CVE prevention)
Pipeline image and OCR endpoints now stream uploads in 1 MB chunks with a configurable `MAX_UPLOAD_BYTES` cap (default 10 MB):
```python
async def _save_upload_to_temp(file, *, suffix, max_upload_bytes) -> str:
    # streams in chunks, raises HTTP 413 if total > max_upload_bytes
```

#### 6. ChatManager State Check
`get_chat_manager()` in `deps.py` now returns HTTP 503 if `chat_manager` was not successfully initialized, instead of raising an `AttributeError`.

#### 7. RAG Pickle Security (supply-chain attack mitigation)
`mego-final` loads `metadata_mapping.pkl` directly with `pickle.load()` — a security risk if the file is ever replaced.

`hesh-edits2` adds:
- SHA-256 hash verification of the pickle before loading (file `metadata_mapping.pkl.sha256` must exist)
- Preferential loading from `metadata_mapping.json` (safe) if available
- `ALLOW_UNSAFE_PICKLE_METADATA=true` override for trusted local dev environments

#### 8. Session Store: Memory Safety
`hesh-edits2` adds TTL expiry (default 1 hour) and max-session LRU eviction (default 500 sessions) to `ChatSessionStore`. Without this, `mego-final`'s session store grows indefinitely and leaks memory under sustained usage.

#### 9. Gemini SDK Streaming Compatibility Fix
`mego-final`'s `generate_content_stream` call may fail on some google.genai SDK versions where it returns a coroutine rather than an async iterator.

`hesh-edits2` fixes this:
```python
if inspect.isawaitable(stream_handle):
    stream_handle = await stream_handle
```

#### 10. OCR Engine Singleton
`hesh-edits2` changes OCR engine from per-call instantiation to a lazy singleton in `ChatManager._get_ocr_engine()`, preventing repeated model loading overhead on repeated calls.

#### 11. TherapyEngine: Better API Key Validation & Error Classification
`mego-final` checks `"AIza" in gemini_api_key` — fragile for keys with different prefixes.  
`hesh-edits2` uses `bool(str(gemini_api_key or "").strip())`.

Also adds `_provider_status_from_exception()` to categorize errors as `provider_unauthorized`, `provider_rate_limited`, `provider_timeout`, or `provider_unavailable` in fallback payloads.

#### 12. Diagnosis Adapter Async Fix
`mego-final` calls `diagnose(...)` directly from `adapt_and_diagnose()` (a sync function). If `diagnose` returns a coroutine, this silently discards the result.

`hesh-edits2` wraps it: `return run_async(diagnose(...))`.

#### 13. Pinned Dependencies
`mego-final` uses floating ranges (`fastapi>=0.115.0`). `hesh-edits2` pins everything to known-good versions, preventing CI breakage from upstream releases.

#### 14. Gemini Model Default Change
Default changed from `gemini-2.5-flash` → `gemini-2.5-flash-lite` across all components. This is a cost-saving change. The model can be overridden via `GEMINI_MODEL_NAME` env var.

### Frontend UI/UX Changes

#### 1. Dual-Mode Application
`hesh-edits2` introduces an `AppMode` toggle in the header:
- **User Interface** mode — simplified patient-facing chat experience (`UserInterfaceView.tsx`, 530 lines)
- **Dev Workbench** mode — technical tab-based panel (Labs / Image / Symptoms with full `ResultView`)

The default is "User Interface", which is the intended patient-facing flow.

#### 2. New `UserInterfaceView` Component
A full timeline-based chat interface with:
- Multi-modal composer (text symptoms, labs JSON attachment, image upload, free-form chat)
- Auto-routing: detects symptom-like text vs free chat using a regex heuristic
- Streaming AI responses with fallback to non-streaming
- Clarification question flow integrated into the timeline
- Session-level state (session ID, conversation history)
- Bilingual UI (Arabic RTL labels for Arabic users)
- Analysis result cards embedded directly in the chat timeline

#### 3. `ChatInterface.tsx` Simplified
The original 236-line component (multi-mode form with clarification state) is replaced with a focused 130-line Arabic-first standalone chat widget (`dir="rtl"`). This widget handles only the therapy/follow-up chat, while diagnosis input moves to `UserInterfaceView`.

#### 4. New Design System
`index.css` overhauled:
- New color palette: teal accent (`#0d9488`), dark slate text (`#0f172a`), light grey background
- New font: `DM Sans` replacing `Space Grotesk`
- Mode-aware header: workbench → teal gradient, user mode → solid blue
- `mode-toggle` pill component in header
- Responsive, readable layout tokens

#### 5. Streaming Chat Frontend Support
`frontend/src/shared/api/client.ts` adds `postChatStream()` with proper SSE parsing (chunk buffering, `\n\n` event-block splitting, trailing-buffer flush). Falls back to `postChat()` if streaming fails.

#### 6. Frontend API Key Forwarding
All API calls now attach `X-API-Key` from `VITE_API_KEY` if set.

#### 7. ResultView Improvements
`ResultView.tsx` expanded with better structured display of diagnosis results and clarification prompts.

#### 8. `ReviewerPanel` Removed
The old side-panel reviewer is removed. Its functionality is absorbed into `ResultView` and `UserInterfaceView`.

---

## Potential Merge Conflicts

### 🔴 High-Risk Conflicts (require manual resolution)

| File | Conflict Nature |
|---|---|
| `backend/models/diagnosis/diagnosisengine.py` | **Largest single conflict.** `mego-final` has 2,715 lines with CLARIFICATION_PAIR_BANK, DIAGNOSTIC_SIGNAL_BANK, generic pattern maps, and pre-diagnosis signal weighting. `hesh-edits2` has a 1,154-line simplified version with different thresholds. These files diverged significantly on both structure and logic. |
| `backend/models/diagnosis/rag.py` | `mego-final` has configurable rerank weights, `_pathology_mismatch_penalty`, and environment-variable-controlled parameters. `hesh-edits2` replaces unsafe pickle loading with hash-verified loading. Both changes are valuable and must be combined carefully. |
| `backend/manager/chat_support.py` | `mego-final` has corrupted Arabic strings; `hesh-edits2` replaces them entirely. Automatic — take `hesh-edits2`'s version. |
| `backend/app/main.py` | `mego-final` has simple lifespan; `hesh-edits2` adds degraded startup, request audit middleware, and `runtime_features`. Both extend the same lifespan function. |
| `backend/app/routers/pipeline.py` | `mego-final` has basic file handling; `hesh-edits2` adds streaming chunk upload, 413 enforcement, and API-key router dependency. |
| `backend/manager/symptom_parser.py` | `mego-final` has richer negation logic and extra symptom aliases; `hesh-edits2` simplifies. The merged version should keep `mego-final`'s broader patterns with `hesh-edits2`'s formatting. |
| `backend/tests/test_diagnosis_engine.py` | `mego-final` has 1,636-line test suite for old engine; `hesh-edits2` has 376-line simplified tests. Must reconcile against whichever diagnosis engine version is kept. |

### 🟡 Medium-Risk Conflicts

| File | Conflict Nature |
|---|---|
| `backend/manager/chat_manager.py` | `mego-final` passes `normalized_follow_up_text` to diagnosis; `hesh-edits2` removes that param. Also streaming message history issue fixed in `hesh-edits2`. |
| `backend/manager/session_store.py` | `mego-final` has basic store; `hesh-edits2` adds TTL/eviction. Non-overlapping additions — mergeable. |
| `frontend/src/App.tsx` | Completely restructured. `mego-final` has single-mode; `hesh-edits2` has dual-mode. |
| `frontend/src/features/chat/components/ChatInterface.tsx` | Both modified the same component with different intent. |
| `frontend/src/features/results/components/ResultView.tsx` | Both modified; `mego-final` has reviewer panel; `hesh-edits2` expands inline result display. |
| `frontend/src/index.css` | Different design systems — take `hesh-edits2`'s version for the new design system. |
| `backend/requirements.txt` | Floating ranges vs pinned versions — take `hesh-edits2` pinned. |

### 🟢 Low-Risk / Auto-resolvable

| File | Notes |
|---|---|
| `backend/app/config.py` | `hesh-edits2` adds `max_upload_bytes`, `require_service_api_key`, `service_api_key`, `allow_unsafe_pickle_metadata`, `gemini_model_name`. `mego-final` lacks these. Easy: add `hesh-edits2` additions to `mego-final`. |
| `backend/app/deps.py` | `hesh-edits2` adds `require_service_api_key` function and null-check. Easy: add to `mego-final`. |
| `backend/models/common/ai_provider.py` | `hesh-edits2` adds SDK-version-compatibility fix for async streaming. Easy: apply fix to `mego-final`. |
| `backend/models/therapy/engine.py` | `hesh-edits2` adds `provider_status` and better key validation. Easy: apply to `mego-final`. |
| `backend/manager/diagnosis_adapter.py` | `hesh-edits2` wraps async call with `run_async`. Easy: apply to `mego-final`. |
| `frontend/src/shared/api/client.ts` | `hesh-edits2` adds `postChatStream`, `withServiceApiKey`. Apply to `mego-final`. |
| `frontend/src/shared/layout/AppHeader.tsx` | `hesh-edits2` adds mode toggle. Easy: apply to `mego-final`. |
| `.gitignore` | Minor additions. |
| `README.md` | New content added in `hesh-edits2`. |

---

## Branch Strengths Summary

### `mego-final` Unique Strengths

1. **Richer differential diagnosis** — `CLARIFICATION_PAIR_BANK` with 6 condition-pair clarifying questions forces the AI to disambiguate before committing to a diagnosis, directly improving accuracy on borderline cases (AF/PSVT, PE/Bronchitis, Unstable/Stable Angina, MG/GBS, etc.).

2. **Per-condition signal scoring** — `DIAGNOSTIC_SIGNAL_BANK` with 14 conditions enables positive/negative evidence weighting after the AI response, reducing false positives.

3. **Pathology-specific RAG penalty** — `_pathology_mismatch_penalty()` suppresses unlikely RAG hits (e.g., Ebola without hemorrhagic context, Chagas without travel history), making retrieval more clinically coherent.

4. **Fine-tuned ClinicalBERT integration** — A properly trained fine-tuned classifier (`train_clinicalbert_classifier.py`, targeted training datasets) was part of the development workflow, though the data files are not in VCS on `hesh-edits2`.

5. **Pre-diagnosis signal weighting** — Early pattern alignment bonuses/penalties before the main AI call provide a better starting prior.

6. **Richer negation handling** — `symptom_parser.py` has more Arabic negation cues and sophisticated productive-cough negation logic, correctly handling "I don't have a cough" vs "I don't have productive cough".

7. **Broader clarification question set** — Includes the bronchospasm/asthma clarification question for wheeze-vs-infection disambiguation.

8. **Lower `CLASSIFIER_PRIMARY_THRESHOLD` (0.52)** — Better recall for the fine-tuned classifier.

### `hesh-edits2` Unique Strengths

1. **Critical bug fix: corrupted chat messages** — All user-facing messages in `mego-final` are garbled/corrupted strings. `hesh-edits2` fixes this.

2. **Production security hardening** — API key auth, upload size limits, hash-verified pickle loading. None of these exist in `mego-final`.

3. **Memory-safe session store** — TTL expiry + LRU eviction prevents unbounded memory growth under production load.

4. **Startup graceful degradation** — Application continues to function without RAG/classifier if model directories are absent. `mego-final` crashes on startup if configured paths are missing.

5. **Gemini SDK streaming fix** — `mego-final` may fail silently on some SDK versions due to awaitable vs async-iterator mismatch.

6. **OCR engine singleton** — Avoids repeated model loading on every OCR request.

7. **New patient-facing UI** — `UserInterfaceView.tsx` provides a full chat-timeline experience suitable for a graduation demo with real users. `mego-final` has only the technical workbench view.

8. **Dual-mode UX** — Medical staff can use the Dev Workbench; patients use User Interface.

9. **Bilingual error messages** — Auto-detected Arabic/English for all system messages.

10. **SSE streaming frontend** — Proper chunked streaming with fallback, better perceived responsiveness.

11. **Comprehensive new test suite** — 15 new test files covering security, integration, startup, session memory, upload limits, RAG safety.

12. **Pinned dependencies** — Reproducible builds across team members and CI.

13. **TherapyEngine error classification** — Distinguishes auth errors, rate limits, timeouts, and availability failures in fallback payloads.

---

## Actionable Merge Recommendations

### Strategy: Use `mego-final` as the target, apply `hesh-edits2` on top

`mego-final` has the better diagnosis brain; `hesh-edits2` has the better everything else. The recommended approach is:

```
git checkout mego-final
git checkout -b merge/hesh-into-mego
git merge hesh-edits2
# resolve conflicts per the guidance below
```

### Step-by-Step Conflict Resolution Guide

---

#### Step 1 — `backend/manager/chat_support.py` (TAKE `hesh-edits2` ENTIRELY)

**Reason:** `mego-final` has corrupted Arabic strings. `hesh-edits2`'s version is correct and an improvement.

```bash
git checkout hesh-edits2 -- backend/manager/chat_support.py
```

---

#### Step 2 — `backend/models/diagnosis/diagnosisengine.py` (MANUAL MERGE — highest priority)

Start from `mego-final`'s version (richer logic), then apply these `hesh-edits2` changes:

1. **Keep `mego-final`'s thresholds except:** set `CLASSIFIER_PRIMARY_THRESHOLD = 0.55` (from hesh-edits2 — reduces false classifier accepts).
2. **Keep ALL of `mego-final`'s** `CLARIFICATION_PAIR_BANK`, `DIAGNOSTIC_SIGNAL_BANK`, `GENERIC_RULE_PATTERN_*`, `PRE_DIAGNOSIS_*`, `CLARIFICATION_OVERRIDE_*` constants and logic.
3. **Keep `mego-final`'s** bronchospasm clarification question.
4. **Apply from `hesh-edits2`:** `CLARIFICATION_MARGIN_THRESHOLD = 0.12` (fewer false clarification triggers).
5. **Remove `import re`** if it is only used by removed logic (check usage in `mego-final` first).

---

#### Step 3 — `backend/models/diagnosis/rag.py` (MANUAL MERGE)

Start from `mego-final`'s version, apply `hesh-edits2` security improvements:

1. **Add** `hashlib` and `pickle` imports from `hesh-edits2`.
2. **Replace** the simple `pickle.load(handle)` with `hesh-edits2`'s `_load_metadata()` method (hash-verified loading with JSON fallback).
3. **Keep** `mego-final`'s `configure_rerank_weights()`, `configure_search_expansion()`, `_pathology_mismatch_penalty()`, and re-rank weight constants.
4. **Keep** `mego-final`'s `hoarseness` and `weight_loss` in `_SYMPTOM_FEATURE_MAP` and `_DISCRIMINATIVE_FEATURES` (removed in hesh-edits2 — these are medically relevant).
5. **Add** `allow_unsafe_pickle: bool = False` parameter to `__init__` and pass it through from `ChatManager`.

---

#### Step 4 — `backend/app/main.py` (MANUAL MERGE)

Start from `hesh-edits2` (it has all the important fixes), verify `mego-final` didn't add any startup logic that needs to be preserved. Key things to keep from `hesh-edits2`:
- `_resolve_optional_ai_flags()` for graceful degradation
- Request audit middleware
- `runtime_features` dict on app state

---

#### Step 5 — `backend/app/config.py` (ADD to `mego-final`)

Add these fields from `hesh-edits2` to `mego-final`'s `Settings`:
```python
max_upload_bytes: int = 10 * 1024 * 1024
require_service_api_key: bool = False
service_api_key: Optional[str] = None
allow_unsafe_pickle_metadata: bool = False
gemini_model_name: str = "gemini-2.5-flash-lite"
```
Also change `rag_top_k` default from `7` → `5`.

---

#### Step 6 — `backend/app/deps.py` (ADD to `mego-final`)

Add `hesh-edits2`'s `require_service_api_key()` dependency function and the null-check in `get_chat_manager()`.

---

#### Step 7 — `backend/app/routers/pipeline.py` (TAKE `hesh-edits2`, verify)

`hesh-edits2` has all the important improvements (upload limits, chunk streaming, API-key dependency). Verify that any new routes added in `mego-final` (if any) are also present.

---

#### Step 8 — `backend/manager/chat_manager.py` (MANUAL MERGE)

Keep `mego-final`'s full constructor signature, but apply:
1. `allow_unsafe_pickle_metadata` and `gemini_model_name` params from `hesh-edits2`
2. OCR engine singleton pattern (`_get_ocr_engine()`)
3. `model_name` pass-through to `TherapyEngine`
4. Streaming history fix (user message appended before availability check)
5. `get_chat_error_message(message)` / `get_stream_error_message(message)` dynamic messages

Check whether `normalized_follow_up_text` in the `mego-final` diagnosis call is intentional or vestigial — remove only if the diagnosis engine doesn't use it.

---

#### Step 9 — `backend/manager/session_store.py` (TAKE `hesh-edits2`, test)

`hesh-edits2`'s version is strictly a superset (TTL + LRU). No regression risk. Take it.

---

#### Step 10 — `backend/manager/symptom_parser.py` (BASE on `mego-final`)

Keep `mego-final`'s richer version, apply only `hesh-edits2`'s code formatting cleanups (line wrapping). In particular:
- Keep extended Arabic negation cues (`"ما في"`, `"مافي"`, `"ما عندي"`, `"ليس"`)
- Keep productive-cough negation disambiguation
- Keep `"short of breath"` alias
- Keep plural ptosis aliases
- Keep encoded-sequence pattern detection

---

#### Step 11 — `backend/models/common/ai_provider.py` (APPLY fix from `hesh-edits2`)

Apply the `inspect.isawaitable(stream_handle)` fix to `mego-final`'s version. Also update default model to `gemini-2.5-flash-lite`.

---

#### Step 12 — `backend/models/therapy/engine.py` (APPLY improvements from `hesh-edits2`)

Apply:
- `_provider_status_from_exception()` method
- `provider_status` in `_fallback_payload()`
- Better API key check (remove `"AIza" in` prefix check)
- `model_name` parameter pass-through

---

#### Step 13 — `backend/manager/diagnosis_adapter.py` (TAKE `hesh-edits2`)

The `run_async` wrap is a bug fix. Take it.

---

#### Step 14 — Frontend (`frontend/src/`)

1. **Take `hesh-edits2`'s entire `frontend/src/`** — it is a complete redesign with new components, new CSS, new structure.
2. `mego-final` frontend changes are largely superseded by `hesh-edits2`.
3. Verify `ResultView.tsx` shows all the diagnosis fields that `mego-final`'s result structure exposes (clarification questions, signal bank matches, etc.).

---

#### Step 15 — Requirements & Dependencies

Use `hesh-edits2`'s pinned versions as the baseline. If `mego-final` added any new libraries for the advanced diagnosis features, add those in pinned form to `requirements-ai.txt`.

---

#### Step 16 — Tests

1. **Keep all new tests from `hesh-edits2`** (security, integration, session, upload limits, RAG, startup, etc.)
2. **Rebuild `test_diagnosis_engine.py`** based on whichever diagnosis engine version is chosen — the `mego-final` 1,636-line test suite is more complete but tests the more complex logic; adapt it to the merged engine.
3. **Delete** tests for scripts that no longer exist.

---

#### Step 17 — Data Files

Do NOT re-add the `targeted_training/` CSV files to VCS. Add them to `.gitignore` (already done in `hesh-edits2`). Store them in a separate data repository or cloud storage if needed for re-training.

---

### Risk of Regressions to Watch

| Risk | Mitigation |
|---|---|
| `mego-final`'s diagnosis accuracy regression if CLARIFICATION_PAIR_BANK is omitted | Follow Step 2 carefully — keep the bank |
| `hesh-edits2` misses `mego-final`'s RAG re-rank weights | Follow Step 3 — keep `configure_rerank_weights()` |
| `hesh-edits2`'s higher `CLASSIFIER_PRIMARY_THRESHOLD` (0.55) reduces recall | Monitor classification test cases; revert to 0.52 if recall drops |
| Removing `normalized_follow_up_text` breaks clarification re-rank | Investigate in chat_manager/diagnosis flow before removing |
| SSE streaming errors if Gemini SDK version changes | Keep `inspect.isawaitable` guard in `ai_provider.py` |
| Session store TTL too short under long diagnosis sessions | Consider making `session_ttl_seconds` configurable via env var |

---

*End of analysis. Document produced from `git diff origin/mego-final origin/hesh-edits2` — 99 files changed, 6,180 insertions, 15,968 deletions.*
