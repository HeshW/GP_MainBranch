"""models/diagnosis/diagnosisengine.py

Production diagnosis engine derived from models/diagnosis/diagnosisproto.ipynb.

Two diagnosis paths
-------------------
Lightweight (default)
    Rule-based lab-value threshold analysis — no heavy dependencies.
    Works directly on any ``OCREngine.extract()`` output dict.

RAG (opt-in, ``use_rag=True``)
    ClinicalBERT + FAISS vector search + Gemini LLM.
    Requires a pre-built FAISS index directory and a Gemini API key.
    Heavy dependencies (torch, transformers, faiss-cpu, google-generativeai)
    are imported lazily; the lightweight path is fully usable without them.

Public API
----------
``diagnose(report, ...)``         module-level convenience function
``DiagnosisEngine``               stateful engine (useful for repeated RAG queries)
``build_combined_text(report)``   OCR dict → free-text patient summary for embedding
``EvidenceMapper``                evidence code → human-readable text (from prototype)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Finding:
    condition: str
    confidence: str   # "high" | "moderate" | "low"
    evidence: str
    severity: str     # "critical" | "high" | "moderate" | "low" | "info"


@dataclass
class _Rule:
    """A single threshold-based clinical rule."""

    lab: str          # canonical lab key produced by OCREngine
    condition: str    # human-readable condition name
    check: Any        # callable(float) -> bool
    evidence_fmt: str # format string accepting {val} and {unit}
    confidence: str = "moderate"
    severity: str = "moderate"


# ---------------------------------------------------------------------------
# Clinical threshold rules
# Units are those typically produced by the OCR engine (mg/dL, g/dL,
# ×10³/µL, mEq/L, %, µg/dL).  Rules are evaluated top-to-bottom; the first
# matching rule for a given lab key wins unless multiple rules cover
# non-overlapping ranges.
# ---------------------------------------------------------------------------

_RULES: List[_Rule] = [
    # ── Glucose (mg/dL) ────────────────────────────────────────────────────
    _Rule("glucose", "Hypoglycemia",
          lambda v: v < 70,
          "glucose={val} {unit} (reference: 70–100 mg/dL fasting)",
          confidence="high", severity="critical"),
    _Rule("glucose", "Impaired Fasting Glucose / Prediabetes",
          lambda v: 100 <= v < 126,
          "glucose={val} {unit} (reference: <100 mg/dL fasting)",
          confidence="moderate", severity="moderate"),
    _Rule("glucose", "Diabetes Mellitus (suspected)",
          lambda v: v >= 126,
          "glucose={val} {unit} (diagnostic threshold: ≥126 mg/dL fasting)",
          confidence="high", severity="high"),

    # ── Hemoglobin (g/dL) ──────────────────────────────────────────────────
    _Rule("hemoglobin", "Severe Anemia",
          lambda v: v < 8.0,
          "hemoglobin={val} {unit} (reference: 12–17 g/dL)",
          confidence="high", severity="critical"),
    _Rule("hemoglobin", "Moderate Anemia",
          lambda v: 8.0 <= v < 10.0,
          "hemoglobin={val} {unit} (reference: 12–17 g/dL)",
          confidence="high", severity="high"),
    _Rule("hemoglobin", "Mild Anemia",
          lambda v: 10.0 <= v < 12.0,
          "hemoglobin={val} {unit} (reference: 12–17 g/dL)",
          confidence="high", severity="moderate"),
    _Rule("hemoglobin", "Polycythemia (suspected)",
          lambda v: v > 17.5,
          "hemoglobin={val} {unit} (reference: 12–17 g/dL; >17.5 elevated)",
          confidence="moderate", severity="moderate"),

    # ── WBC (×10³/µL) ──────────────────────────────────────────────────────
    _Rule("wbc", "Leukopenia",
          lambda v: v < 4.0,
          "wbc={val} {unit} (reference: 4.5–11.0 ×10³/µL)",
          confidence="high", severity="moderate"),
    _Rule("wbc", "Leukocytosis / Possible Infection or Inflammation",
          lambda v: 11.0 < v <= 30.0,
          "wbc={val} {unit} (reference: 4.5–11.0 ×10³/µL)",
          confidence="high", severity="moderate"),
    _Rule("wbc", "Severe Leukocytosis (leukemia screening warranted)",
          lambda v: v > 30.0,
          "wbc={val} {unit} (critically elevated; >30.0 ×10³/µL)",
          confidence="moderate", severity="critical"),

    # ── RBC (×10⁶/µL) ──────────────────────────────────────────────────────
    _Rule("rbc", "Erythropenia (low red cell count)",
          lambda v: v < 4.0,
          "rbc={val} {unit} (reference: 4.2–6.1 ×10⁶/µL)",
          confidence="moderate", severity="moderate"),
    _Rule("rbc", "Erythrocytosis (elevated red cell count)",
          lambda v: v > 6.1,
          "rbc={val} {unit} (reference: 4.2–6.1 ×10⁶/µL)",
          confidence="moderate", severity="moderate"),

    # ── Platelets (×10³/µL) ────────────────────────────────────────────────
    _Rule("platelets", "Thrombocytopenia",
          lambda v: v < 150,
          "platelets={val} {unit} (reference: 150–400 ×10³/µL)",
          confidence="high", severity="moderate"),
    _Rule("platelets", "Thrombocytosis",
          lambda v: v > 450,
          "platelets={val} {unit} (reference: 150–400 ×10³/µL)",
          confidence="moderate", severity="moderate"),

    # ── Hematocrit (%) ─────────────────────────────────────────────────────
    _Rule("hematocrit", "Low Hematocrit (anemia indicator)",
          lambda v: v < 36.0,
          "hematocrit={val} {unit} (reference: 36–53%)",
          confidence="moderate", severity="moderate"),
    _Rule("hematocrit", "High Hematocrit (polycythemia indicator)",
          lambda v: v > 54.0,
          "hematocrit={val} {unit} (reference: 36–53%)",
          confidence="moderate", severity="moderate"),

    # ── Cholesterol (mg/dL) ────────────────────────────────────────────────
    _Rule("cholesterol", "Borderline High Cholesterol",
          lambda v: 200 <= v < 240,
          "cholesterol={val} {unit} (reference: <200 mg/dL)",
          confidence="moderate", severity="low"),
    _Rule("cholesterol", "Hypercholesterolemia",
          lambda v: v >= 240,
          "cholesterol={val} {unit} (reference: <200 mg/dL)",
          confidence="high", severity="moderate"),

    # ── Creatinine (mg/dL) ─────────────────────────────────────────────────
    _Rule("creatinine", "Elevated Creatinine (renal dysfunction suspected)",
          lambda v: 1.3 < v <= 2.0,
          "creatinine={val} {unit} (reference: 0.6–1.3 mg/dL)",
          confidence="moderate", severity="moderate"),
    _Rule("creatinine", "Severely Elevated Creatinine",
          lambda v: v > 2.0,
          "creatinine={val} {unit} (reference: 0.6–1.3 mg/dL; >2.0 critical)",
          confidence="high", severity="high"),

    # ── Urea / BUN (mg/dL) ────────────────────────────────────────────────
    _Rule("urea", "Elevated BUN / Azotemia",
          lambda v: 20 < v <= 50,
          "urea/BUN={val} {unit} (reference: 7–20 mg/dL)",
          confidence="moderate", severity="moderate"),
    _Rule("urea", "Severely Elevated BUN",
          lambda v: v > 50,
          "urea/BUN={val} {unit} (reference: 7–20 mg/dL; >50 critical)",
          confidence="high", severity="high"),

    # ── Sodium (mEq/L) ────────────────────────────────────────────────────
    _Rule("sodium", "Hyponatremia",
          lambda v: v < 135,
          "sodium={val} {unit} (reference: 135–145 mEq/L)",
          confidence="high", severity="moderate"),
    _Rule("sodium", "Hypernatremia",
          lambda v: v > 145,
          "sodium={val} {unit} (reference: 135–145 mEq/L)",
          confidence="high", severity="moderate"),

    # ── Potassium (mEq/L) ────────────────────────────────────────────────
    _Rule("potassium", "Hypokalemia",
          lambda v: v < 3.5,
          "potassium={val} {unit} (reference: 3.5–5.0 mEq/L)",
          confidence="high", severity="moderate"),
    _Rule("potassium", "Hyperkalemia",
          lambda v: 5.0 < v <= 6.0,
          "potassium={val} {unit} (reference: 3.5–5.0 mEq/L)",
          confidence="high", severity="moderate"),
    _Rule("potassium", "Severe Hyperkalemia",
          lambda v: v > 6.0,
          "potassium={val} {unit} (reference: 3.5–5.0 mEq/L; >6.0 critical)",
          confidence="high", severity="critical"),

    # ── Calcium (mg/dL) ──────────────────────────────────────────────────
    _Rule("calcium", "Hypocalcemia",
          lambda v: v < 8.5,
          "calcium={val} {unit} (reference: 8.5–10.5 mg/dL)",
          confidence="moderate", severity="moderate"),
    _Rule("calcium", "Hypercalcemia",
          lambda v: v > 10.5,
          "calcium={val} {unit} (reference: 8.5–10.5 mg/dL)",
          confidence="moderate", severity="moderate"),

    # ── Iron (µg/dL) ─────────────────────────────────────────────────────
    _Rule("iron", "Iron Deficiency",
          lambda v: v < 50,
          "iron={val} {unit} (reference: 50–175 µg/dL)",
          confidence="moderate", severity="moderate"),
    _Rule("iron", "Iron Overload (suspected)",
          lambda v: v > 175,
          "iron={val} {unit} (reference: 50–175 µg/dL)",
          confidence="low", severity="low"),
]


# ---------------------------------------------------------------------------
# Lightweight rule-based analysis
# ---------------------------------------------------------------------------

def _diagnose_from_labs(labs: Dict[str, Any]) -> List[_Finding]:
    """Apply threshold rules to OCR-extracted lab values.

    Parameters
    ----------
    labs:
        ``report["labs"]`` dict from ``OCREngine.extract()``.
        Each entry is expected to be a dict ``{"value": float, "unit": str|None, …}``
        but plain floats are also accepted for easier testing.

    Returns
    -------
    list of :class:`_Finding`
        One item per triggered rule; empty if all values are within range.
    """
    findings: List[_Finding] = []
    seen: set = set()

    for rule in _RULES:
        entry = labs.get(rule.lab)
        if entry is None:
            continue
        try:
            value = float(
                entry.get("value", entry) if isinstance(entry, dict) else entry
            )
        except (TypeError, ValueError):
            logger.debug("Cannot parse lab value for %r: %r", rule.lab, entry)
            continue

        if not rule.check(value):
            continue

        key = (rule.lab, rule.condition)
        if key in seen:
            continue
        seen.add(key)

        unit = entry.get("unit", "") if isinstance(entry, dict) else ""
        evidence = rule.evidence_fmt.format(val=value, unit=unit or "?")
        findings.append(
            _Finding(
                condition=rule.condition,
                confidence=rule.confidence,
                evidence=evidence,
                severity=rule.severity,
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Text-building utility (for the RAG path)
# ---------------------------------------------------------------------------

def build_combined_text(report: Dict[str, Any]) -> str:
    """Convert an OCREngine result dict to a free-text patient summary.

    Produces the format expected by the ClinicalBERT encoder (analogous to
    ``_create_combined_text`` in the prototype notebook).  Returns at most
    ~512 characters to stay within the tokeniser's window.

    Parameters
    ----------
    report:
        The full dict returned by ``OCREngine.extract()``.
    """
    parts: List[str] = []

    # Demographic hint
    sex_age = (report.get("fields") or {}).get("sex_age", "")
    if sex_age:
        parts.append(f"Patient: {sex_age}.")

    # Lab values
    labs = report.get("labs") or {}
    if labs:
        lab_strs = []
        for key, entry in labs.items():
            if isinstance(entry, dict):
                val = entry.get("value", "?")
                unit = (entry.get("unit") or "").strip()
                lab_strs.append(f"{key}={val} {unit}".rstrip())
            else:
                lab_strs.append(f"{key}={entry}")
        parts.append("Labs: " + ", ".join(lab_strs) + ".")

    # Clinical narrative sections
    sections = report.get("sections") or {}
    for sec_name in ("Clinical", "Diagnosis", "Microscopic"):
        text = (sections.get(sec_name) or "").strip()
        if text:
            parts.append(f"{sec_name}: {text[:300]}")

    combined = " ".join(parts)
    return combined[:512] if combined else (report.get("raw_text") or "")[:512]


# ---------------------------------------------------------------------------
# EvidenceMapper  (lightweight utility from the prototype)
# ---------------------------------------------------------------------------

class EvidenceMapper:
    """Maps evidence/symptom codes to human-readable text.

    Ported from the prototype notebook.  Works entirely in-memory with a
    simple string-cleaning heuristic; no external resources required.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def get_text(self, code: str) -> str:
        """Return a capitalised, space-separated rendering of *code*."""
        if code in self._cache:
            return self._cache[code]
        cleaned = str(code).replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"^[Ee]\s+", "", cleaned)
        readable = " ".join(w.capitalize() for w in cleaned.split())
        result = readable if readable else str(code)
        self._cache[code] = result
        return result


# ---------------------------------------------------------------------------
# Optional heavy components  (ClinicalBERT + FAISS + Gemini)
# All imports are guarded so the lightweight path has zero heavy deps.
# ---------------------------------------------------------------------------

class ClinicalBERTEmbedder:
    """Encodes text to 768-dim vectors using ``emilyalsentzer/Bio_ClinicalBERT``.

    Requires: ``torch``, ``transformers``
    """

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, device: Optional[str] = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ClinicalBERTEmbedder requires 'torch' and 'transformers'. "
                "Install them with:  pip install torch transformers"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("ClinicalBERT loaded on %s", self.device)

    def encode_text(self, text: str) -> "np.ndarray":
        """Return a L2-normalised 768-dim float32 vector for *text*."""
        import numpy as np

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype("float32")

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        """Encode a list of texts and return an (N, 768) float32 array."""
        import numpy as np

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            all_embeddings.append(
                outputs.last_hidden_state[:, 0, :].cpu().numpy()
            )
        result = np.vstack(all_embeddings).astype("float32")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return result / norms


class MedicalCaseSearcher:
    """FAISS-backed cosine-similarity search over medical case embeddings.

    Expects a directory containing:

    * ``medical_cases.index``     – FAISS IndexFlatIP built from L2-normalised vectors
    * ``metadata_mapping.pkl``    – pickle with keys ``pathologies``, ``symptoms``,
                                    ``patient_ids``, ``num_vectors``

    Requires: ``faiss-cpu`` (or ``faiss-gpu``), ``numpy``
    """

    def __init__(self, index_dir: Path) -> None:
        try:
            import faiss
            import pickle
        except ImportError as exc:
            raise ImportError(
                "MedicalCaseSearcher requires 'faiss-cpu'. "
                "Install with:  pip install faiss-cpu"
            ) from exc

        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "medical_cases.index"))
        with open(index_dir / "metadata_mapping.pkl", "rb") as fh:
            self.metadata = pickle.load(fh)
        logger.info("FAISS index loaded: %d cases", self.index.ntotal)

    def search(self, query_embedding: "np.ndarray", k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-*k* most similar cases for *query_embedding*.

        Each result is a dict with keys ``similarity``, ``pathology``,
        ``symptoms``, and ``patient_id``.
        """
        import faiss
        import numpy as np

        q = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            symptoms = self.metadata["symptoms"][idx]
            if not symptoms or str(symptoms).strip() in ("", "None reported"):
                symptoms = f"Patient with {self.metadata['pathologies'][idx]}"
            results.append(
                {
                    "similarity": float(score),
                    "pathology": self.metadata["pathologies"][idx],
                    "symptoms": symptoms,
                    "patient_id": self.metadata["patient_ids"][idx],
                }
            )
        return results


class MedicalRAGAssistant:
    """Full RAG pipeline: ClinicalBERT encoder → FAISS retrieval → Gemini LLM.

    Requires: ``torch``, ``transformers``, ``faiss-cpu``, ``google-generativeai``
    """

    MEDICAL_DISCLAIMER = (
        "\n\n⚠️  MEDICAL DISCLAIMER: This output is AI-generated for "
        "informational purposes only and is NOT a substitute for professional "
        "medical advice, diagnosis, or treatment. Always consult a qualified "
        "healthcare provider. In an emergency call emergency services immediately."
    )

    def __init__(
        self,
        embedder: ClinicalBERTEmbedder,
        searcher: MedicalCaseSearcher,
        gemini_api_key: str,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "MedicalRAGAssistant requires 'google-generativeai'. "
                "Install with:  pip install google-generativeai"
            ) from exc

        genai.configure(api_key=gemini_api_key)
        self._genai_model = genai.GenerativeModel(model_name)
        self.embedder = embedder
        self.searcher = searcher

    # ------------------------------------------------------------------

    def _build_prompt(
        self, patient_text: str, cases: List[Dict[str, Any]]
    ) -> str:
        context = "SIMILAR MEDICAL CASES FROM DATABASE:\n\n"
        for i, case in enumerate(cases, 1):
            context += (
                f"Case {i} (Similarity: {case['similarity'] * 100:.1f}%):\n"
                f"- Diagnosis: {case['pathology']}\n"
                f"- Symptoms: {case['symptoms']}\n\n"
            )
        return (
            "You are a medical AI assistant analysing patient data.\n\n"
            f"PATIENT INFORMATION:\n{patient_text}\n\n"
            f"{context}\n"
            "Based on these similar cases provide:\n"
            "1. Most Likely Diagnosis\n"
            "2. Reasoning (reference the similar cases above)\n"
            "3. Key Lab / Symptom Matches\n"
            "4. Recommended Actions\n"
            "5. Warning Signs requiring immediate attention\n\n"
            "Keep the response clear, professional, and evidence-based."
        )

    def query(
        self,
        patient_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run the full RAG pipeline for *patient_text*.

        Returns a dict with ``retrieved_cases``, ``raw_response``,
        and ``response`` (raw + disclaimer appended).
        """
        embedding = self.embedder.encode_text(patient_text)
        cases = self.searcher.search(embedding, k=top_k)
        prompt = self._build_prompt(patient_text, cases)
        try:
            response = self._genai_model.generate_content(
                prompt.encode("utf-8", "ignore").decode("utf-8"),
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                },
            )
            raw_response = response.text
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            raw_response = f"[LLM error: {exc}]"

        return {
            "retrieved_cases": cases,
            "raw_response": raw_response,
            "response": raw_response + self.MEDICAL_DISCLAIMER,
        }


# ---------------------------------------------------------------------------
# DiagnosisEngine — unified entry point
# ---------------------------------------------------------------------------

class DiagnosisEngine:
    """Orchestrates lightweight and/or RAG-based diagnosis.

    Parameters
    ----------
    use_rag:
        Enable the ClinicalBERT + FAISS + LLM path (requires heavy deps,
        a valid ``faiss_index_dir``, and a ``gemini_api_key``).
    faiss_index_dir:
        Directory containing ``medical_cases.index`` and
        ``metadata_mapping.pkl``.  Required when *use_rag* is ``True``.
    gemini_api_key:
        Google Gemini API key.  Required when *use_rag* is ``True``.
    rag_top_k:
        Number of similar cases to retrieve in the RAG path.
    """

    DISCLAIMER = (
        "This analysis is produced by a rule-based AI system for "
        "informational purposes only.  It does NOT constitute medical "
        "advice.  Always consult a licensed healthcare professional."
    )

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        gemini_api_key: Optional[str] = None,
        rag_top_k: int = 5,
    ) -> None:
        self._rag_assistant: Optional[MedicalRAGAssistant] = None
        self._rag_top_k = rag_top_k

        if use_rag:
            if faiss_index_dir is None:
                raise ValueError(
                    "faiss_index_dir is required when use_rag=True"
                )
            if not gemini_api_key:
                raise ValueError(
                    "gemini_api_key is required when use_rag=True"
                )
            embedder = ClinicalBERTEmbedder()
            searcher = MedicalCaseSearcher(Path(faiss_index_dir))
            self._rag_assistant = MedicalRAGAssistant(
                embedder=embedder,
                searcher=searcher,
                gemini_api_key=gemini_api_key,
            )
            logger.info(
                "RAG path initialised (%d cases)", searcher.index.ntotal
            )

    # ------------------------------------------------------------------

    def diagnose(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse *report* (output of ``OCREngine.extract()``) and return findings.

        Parameters
        ----------
        report:
            The dict returned by ``OCREngine.extract()``.  Only
            ``report["labs"]`` is required for the lightweight path;
            ``report["fields"]`` and ``report["sections"]`` are used by the
            RAG path to build the patient-summary query.

        Returns
        -------
        dict
            ``findings``
                List of dicts, each with keys ``condition``, ``confidence``,
                ``evidence``, and ``severity``.
            ``summary``
                Short human-readable summary of the findings.
            ``disclaimer``
                Mandatory medical disclaimer string.
            ``rag_response``  *(RAG only)*
                LLM-generated narrative.
            ``retrieved_cases``  *(RAG only)*
                Top-k similar cases from the FAISS index.
        """
        labs = report.get("labs") or {}
        findings = _diagnose_from_labs(labs)

        # Build summary
        n = len(findings)
        if n == 0:
            summary = "No clinically significant lab abnormalities detected."
        elif n == 1:
            summary = f"1 potential finding: {findings[0].condition}."
        else:
            names = ", ".join(f.condition for f in findings[:3])
            more = f" (+{n - 3} more)" if n > 3 else ""
            summary = f"{n} potential findings including: {names}{more}."

        result: Dict[str, Any] = {
            "findings": [
                {
                    "condition": f.condition,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                    "severity": f.severity,
                }
                for f in findings
            ],
            "summary": summary,
            "disclaimer": self.DISCLAIMER,
        }

        if self._rag_assistant is not None:
            combined = build_combined_text(report)
            rag_out = self._rag_assistant.query(combined, top_k=self._rag_top_k)
            result["rag_response"] = rag_out["response"]
            result["retrieved_cases"] = rag_out["retrieved_cases"]

        return result


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def diagnose(
    report: Dict[str, Any],
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    gemini_api_key: Optional[str] = None,
    rag_top_k: int = 5,
) -> Dict[str, Any]:
    """Diagnose lab abnormalities from an ``OCREngine.extract()`` result dict.

    This is a convenience wrapper around :class:`DiagnosisEngine` for
    one-shot use.  For repeated queries (especially with RAG) prefer
    instantiating ``DiagnosisEngine`` once and calling ``diagnose()`` on it.

    Parameters
    ----------
    report:
        The dict returned by ``OCREngine.extract()``.
    use_rag:
        Enable the ClinicalBERT + FAISS + Gemini path (heavy deps required).
    faiss_index_dir:
        Path to a directory with a pre-built FAISS index.
    gemini_api_key:
        Google Generative AI API key (required when *use_rag* is ``True``).
    rag_top_k:
        Number of similar cases to retrieve when RAG is enabled.

    Returns
    -------
    dict
        Same structure as :meth:`DiagnosisEngine.diagnose`.
    """
    engine = DiagnosisEngine(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        gemini_api_key=gemini_api_key,
        rag_top_k=rag_top_k,
    )
    return engine.diagnose(report)
