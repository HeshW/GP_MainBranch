DIAGNOSIS ENGINE
================

Overview
--------
`models/diagnosis/diagnosisengine.py` implements a lightweight rule-based
diagnosis path (default) and an optional RAG path (ClinicalBERT + FAISS +
Gemini) that is lazy-loaded and only used when explicitly enabled.

Quick Usage
-----------
- One-shot convenience:  `from models.diagnosis.diagnosisengine import diagnose`
- Stateful (recommended for repeated RAG queries):
  `engine = DiagnosisEngine(use_rag=False)` then `engine.diagnose(report)`

Expected Input
--------------
The engine accepts the dict returned by `OCREngine.extract()`.
Only `report["labs"]` is required for the lightweight path. `labs` should be
either a mapping of lab keys to numeric values or to dicts like:

```
{
  "glucose": {"value": 130, "unit": "mg/dL"},
  "hemoglobin": 11.2
}
```

Validation
----------
The engine now validates that `report` is a dict and that `report['labs']`
is a dict. Malformed inputs will raise `TypeError` with a clear message.

Notes
-----
- RAG features require heavy optional dependencies and are disabled by
  default. See the top-level module docstring for details.
