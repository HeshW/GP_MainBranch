# RAG Scope And Limitations

## Current Scope

The current medical RAG bundle is built from local DDXPlus-derived artifacts with a fixed 49-pathology label universe. It is not a general medical diagnosis knowledge base.

The active runtime artifact path is configured through `FAISS_INDEX_DIR` and currently points to:

```text
backend/artifacts/artifacts/faiss_data_targeted
```

Current artifact counts:

- FAISS vectors: `12025`
- Metadata rows: `12025`
- Unique pathologies: `49`
- Active FAISS index completeness: complete relative to local DDXPlus-derived artifacts

Diabetes/hyperglycemia and UTI/cystitis are not present in the active indexed metadata or classifier label maps. They are treated as out-of-scope safety cases, not in-scope retrieval failures.

## What The RAG Is For

The RAG layer is useful for retrieval over the current DDXPlus-style label space, especially respiratory, ENT, cardiopulmonary, infectious, neurological, GI, allergic/toxic, and related acute-care patterns that exist in the 49-label metadata.

It should not be presented as a broad medical diagnosis system. Conditions outside the indexed label space may still retrieve superficially similar cases, so confidence gating is required before RAG evidence is allowed into final diagnosis fusion.

## Runtime Improvements

No model retraining or FAISS rebuild was used for the current improvements.

Implemented runtime changes:

- FAISS IVF `nprobe` is configured on load (`RAG_FAISS_NPROBE`, default `16`) to improve recall.
- Query construction includes structured clinical fields while preserving natural text compatibility.
- Reranking combines:
  - embedding similarity
  - symptom overlap
  - lexical overlap
  - feature alignment
  - lab match
  - age/sex alignment
  - disease-family hints
  - mismatch and context penalties
- Confidence gating prevents low-confidence or out-of-scope RAG from overpowering rules/classifier.
- Debug/admin metadata exposes `rag_confidence`, `usable_for_fusion`, `rag_scope_status`, score components, and detected out-of-scope signals when RAG is enabled.

## Current Metrics

DDXPlus-scoped smoke evaluation:

- Top-1: `0.8`
- Top-3: `1.0`
- Top-5: `1.0`
- MRR: `0.8667`
- Out-of-scope low-confidence rate: `1.0`

Expanded DDXPlus-scoped evaluation:

- In-scope cases: `44`
- Out-of-scope safety cases: `2`
- Top-1: `0.8636`
- Top-3: `0.8636`
- Top-5: `0.8864`
- MRR: `0.8682`
- Out-of-scope low-confidence rate: `1.0`

Threshold defaults:

- Top-5 >= `0.85`
- MRR >= `0.70`
- Out-of-scope low-confidence rate >= `0.90`

## Reproduction Commands

Run DDXPlus completeness verification:

```powershell
python backend\scripts\verify_ddxplus_completeness.py
```

Run smoke retrieval evaluation:

```powershell
python backend\scripts\evaluate_rag_retrieval_quality.py --case-set smoke --run-label smoke_current
```

Run expanded retrieval evaluation:

```powershell
python backend\scripts\evaluate_rag_retrieval_quality.py --case-set expanded --run-label expanded_current
```

Run expanded evaluation with CI-style threshold failure:

```powershell
python backend\scripts\evaluate_rag_retrieval_quality.py --case-set expanded --fail-on-threshold
```

Run RAG health check:

```powershell
python backend\scripts\rag_health_check.py --pretty
```

Run selected tests:

```powershell
python -m pytest backend\tests\test_rag.py backend\tests\test_rag_health_check.py backend\tests\test_startup_guardrails.py backend\tests\test_diagnosis_engine.py -q
```

## Output Artifacts

Completeness reports:

- `data/evaluation/rag_diagnostics/ddxplus_completeness_report.json`
- `data/evaluation/rag_diagnostics/ddxplus_completeness_report.md`

Expanded retrieval reports:

- `data/evaluation/rag_diagnostics/expanded_retrieval_eval_summary.json`
- `data/evaluation/rag_diagnostics/expanded_retrieval_eval_cases.csv`
- `data/evaluation/rag_diagnostics/expanded_retrieval_eval_report.md`

## Known Limitations

- The knowledge base is limited to the current 49 DDXPlus-derived pathologies.
- Out-of-scope conditions can retrieve similar but incorrect in-scope neighbors.
- Some closely related DDXPlus labels remain hard to separate with embeddings alone, for example cough/ENT syndromes and pulmonary neoplasm vs tuberculosis-like presentations.
- The expanded evaluation is still a curated smoke set, not a clinical benchmark.
- External LLM narrative synthesis is separate from retrieval and may fail due to provider quota or availability.

## Future Work

If broader medical coverage is required:

1. Add new curated indexed data for the new disease families.
2. Rebuild the FAISS index after adding data or changing embedding strategy.
3. Retrain or fine-tune the classifier if the label space changes.
4. Expand evaluation to include more cases per pathology and more out-of-scope safety cases.
5. Add CI thresholds once the evaluation set is stable enough for release gating.
