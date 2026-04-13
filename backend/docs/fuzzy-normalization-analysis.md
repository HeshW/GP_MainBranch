# Fuzzy-Matching and Auto-Normalization Analysis — `mego-final` Branch

## 1. Executive Summary

The `mego-final` branch ships a **comprehensive auto-normalization layer** but
**no fuzzy-matching** prior to this PR.  All text-to-canonical resolution was
performed through exact string lookups and compiled regex patterns.  Because
`rapidfuzz` was already declared as a runtime dependency
(`requirements-runtime.txt`) but never imported anywhere in the source tree,
this PR activates fuzzy-matching as a transparent fallback layer on top of the
existing exact-match pipeline.

---

## 2. Pre-existing Auto-Normalization Logic

### 2.1 Whitespace / OCR text normalization

| Location | Function | What it does |
|---|---|---|
| `backend/models/ocr/parsing.py` | `normalise_text(raw)` | Collapses all whitespace (tabs, newlines, multiple spaces) to a single space and strips leading/trailing whitespace. Used on every OCR text block before lab-pattern matching. |

### 2.2 Lab-name synonym mapping

| Location | Mechanism | What it does |
|---|---|---|
| `backend/models/ocr/patterns.py` | `SYNONYM_MAP` dict | Maps ~40 hardcoded abbreviations and alternate spellings (e.g. `"hgb"`, `"haematocrit"`, `"blood urea nitrogen"`) to canonical keys like `"hemoglobin"`, `"hematocrit"`, `"urea"`. |
| `backend/models/ocr/patterns.py` | `synonyms_v15.json` merge | Optionally extends `SYNONYM_MAP` with aliases ingested from the GPProject OC Version 15 document (if the JSON file is present). |
| `backend/manager/symptom_parser.py` | `_load_aliases()` / `_get_lab_aliases()` | Loads the same `synonyms_v15.json` into a cached dict used for alias → canonical resolution during free-text lab extraction. |
| `backend/manager/symptom_validator.py` | `_canonical_lab_name(name)` | Looks up a raw lab name in the alias dict; returns the canonical key or the original (lowercased) string. |

**Gap before this PR:** all lookups were *exact* — a one-character OCR error
like `"haemoglobn"` (missing `i`) returned itself rather than `"hemoglobin"`.

### 2.3 Unit normalization

| Location | Function | What it does |
|---|---|---|
| `backend/manager/symptom_validator.py` | `_normalize_unit(lab_key, unit)` | Normalizes common lab units to a canonical casing (`"mg/dl"` → `"mg/dL"`, `"mmol/l"` → `"mmol/L"`, etc.) and replaces Unicode micro-sign variants (`µ`, `μ`) with `u`. |

### 2.4 Symptom keyword extraction

| Location | Variable / Function | What it does |
|---|---|---|
| `backend/manager/symptom_parser.py` | `SYMPTOM_PATTERNS` list | 32 canonical symptom entries, each with 1–8 English and Arabic aliases. Exact word-boundary (`\b`) regex matching against lowercased input. |
| `backend/manager/symptom_parser.py` | `_extract_symptoms(raw_text)` | Iterates all canonical/alias pairs; on a hit, appends `{symptom, source, confidence: 0.85}`. Handles Arabic substring matching and negation detection (English + Arabic cues). |

**Gap before this PR:** tokens like `"fatiqued"` (misspelling of `"fatigue"`)
or `"headeache"` failed to match any pattern.

### 2.5 Symptom canonicalization map

| Location | Variable | What it does |
|---|---|---|
| `backend/manager/symptom_validator.py` | `SYMPTOM_CANONICAL_MAP` dict | Maps a small set of known synonyms that bypass `SYMPTOM_PATTERNS` (e.g. `"dyspnea"` → `"shortness of breath"`, `"heartburn"` → `"reflux"`) to their canonical form. |

### 2.6 Structured symptom-text normalization

| Location | Function | What it does |
|---|---|---|
| `backend/manager/symptom_normalizer.py` | `build_normalized_symptom_text(parsed, validated)` | Assembles a structured, DDX-style text block from validated symptoms, raw complaint text, duration mentions, context cues, and Arabic context patterns. Used to enrich the prompt sent to the RAG / Gemini layer. |

---

## 3. Fuzzy-Matching — State Before This PR

**Result: absent.**

A search across all `*.py` files under `backend/` found **zero** usages of:
- `rapidfuzz` (even though it appears in `requirements-runtime.txt`)
- `difflib`
- `Levenshtein`
- Any other edit-distance / phonetic matching library

All symptom and lab-name resolution was purely exact-string or regex-based.

---

## 4. Fuzzy-Matching Implemented in This PR

### 4.1 New module: `backend/manager/fuzzy_utils.py`

| Symbol | Purpose |
|---|---|
| `_build_alias_index(symptom_patterns)` | Builds a flat `{alias_lower: canonical}` dict from `SYMPTOM_PATTERNS`, excluding Arabic aliases (edit-distance on Arabic is unreliable). |
| `extract_candidate_tokens(text)` | Returns de-duplicated single words and adjacent bigrams from an English text string; short tokens (< 4 chars), pure digits, and Arabic tokens are excluded. |
| `fuzzy_match_symptom(token, alias_index, threshold=75)` | Calls `rapidfuzz.process.extractOne` with the WRatio scorer against the alias index. Returns `(canonical, matched_alias, score)` or `None`. |
| `fuzzy_match_lab_name(name, aliases, threshold=80)` | Same mechanism for lab-name canonicalization; higher default threshold (80) because lab names require greater precision. |
| `FUZZY_SYMPTOM_CONFIDENCE` | Confidence value (0.65) assigned to fuzzy-matched symptoms, lower than the 0.85 assigned to exact matches. |

### 4.2 Symptom parser integration (`_extract_symptoms`)

After the exact-match loop, the fuzzy fallback:
1. Builds the alias index once per call.
2. Extracts candidate tokens from the lowercased text.
3. For each candidate not yet matched to a canonical, calls `fuzzy_match_symptom`.
4. Appends `{symptom, source, confidence: 0.65, fuzzy: True}` for accepted matches.

The `fuzzy=True` flag allows downstream consumers to distinguish fuzzy hits
from exact ones (e.g., for audit trails or review-required logic).

### 4.3 Lab-name validator integration (`_canonical_lab_name`)

When an exact alias lookup returns no match, `_canonical_lab_name` now calls
`fuzzy_match_lab_name`.  On success it returns the fuzzy-resolved canonical;
on failure it passes through the original lowercased string unchanged (same
behaviour as before this PR).

---

## 5. Design Notes

| Concern | Decision |
|---|---|
| **False-positive rate** | Exact matching always runs first; fuzzy kicks in only for unmatched tokens, minimising spurious hits. |
| **Threshold tuning** | Symptoms: 75 (WRatio); labs: 80. These were validated empirically against the test suite. |
| **Graceful degradation** | Both integration points wrap the fuzzy import in `try/except ImportError`, so the pipeline continues to function even in environments where `rapidfuzz` is not installed. |
| **Arabic text** | Arabic aliases are excluded from the fuzzy index — edit-distance on short Arabic words is unreliable and phonetic matching is not implemented. Arabic exact matching is preserved as-is. |
| **Confidence transparency** | Fuzzy-matched symptom entries carry `confidence: 0.65` and `fuzzy: True`, making them distinguishable from exact matches for audit purposes. |

---

## 6. Suggested Future Improvements

1. **Extend fuzzy lab matching to the OCR layer** — `models/ocr/patterns.py`
   builds compile-time regex patterns; a post-regex fuzzy rescue pass on the
   raw OCR line could catch label errors before the cross-line fallback.

2. **Phonetic matching for Arabic symptoms** — Libraries such as `jaro-winkler`
   adapted for Arabic, or a dedicated Arabic NLP pre-processing step, could
   improve recall for Arabic misspellings.

3. **Threshold configuration via environment variables** — expose
   `SYMPTOM_FUZZY_THRESHOLD` and `LAB_FUZZY_THRESHOLD` as configurable
   settings (e.g. via `pydantic-settings`) so they can be adjusted without code
   changes.

4. **Fuzzy negation detection** — the negation cue list is currently exact;
   fuzzy cue matching (e.g. `"denys"` → `"denies"`) could improve recall.

5. **Offline evaluation dataset** — build a small test corpus of deliberately
   misspelled inputs and run a precision/recall evaluation to empirically tune
   thresholds.
