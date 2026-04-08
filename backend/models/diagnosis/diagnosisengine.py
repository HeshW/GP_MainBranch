"""models/diagnosis/diagnosisengine.py

Production diagnosis engine derived from models/diagnosis/diagnosisproto.ipynb.

Two diagnosis paths
-------------------
Lightweight (default)
    Rule-based lab-value threshold analysis — no heavy dependencies.
    Works directly on any ``OCREngine.extract()`` output dict.

RAG (opt-in, ``use_rag=True``)
    ClinicalBERT (mean pooling, matching the prototype notebook) + FAISS
    vector search + Gemini LLM, with optional Arabic→English medical
    translation before encoding.  Requires a pre-built FAISS index directory
    and a Gemini API key.  Heavy dependencies (torch, transformers,
    faiss-cpu, google-generativeai) are imported lazily; the lightweight path
    is fully usable without them.

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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

from models.common.ai_provider import GeminiProvider

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


import yaml

@dataclass
class _Rule:
    """A single threshold-based clinical rule loaded from YAML."""

    lab: str          # canonical lab key
    condition: str    # human-readable name
    operator: str     # 'lt', 'gt', 'le', 'ge', 'range', etc.
    evidence_fmt: str # format string
    confidence: str = "moderate"
    severity: str = "moderate"
    limit: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def matches(self, val: float) -> bool:
        """Evaluate rule criteria against a value."""
        if self.operator == "lt": return val < self.limit
        if self.operator == "le": return val <= self.limit
        if self.operator == "gt": return val > self.limit
        if self.operator == "ge": return val >= self.limit
        if self.operator == "eq": return val == self.limit
        if self.operator == "range":
            return (self.min_val <= val < self.max_val) if self.min_val is not None and self.max_val is not None else False
        return False


def _load_clinical_rules() -> List[_Rule]:
    """Load and parse rules from clinical_rules.yaml."""
    rules_path = Path(__file__).parent / "clinical_rules.yaml"
    if not rules_path.exists():
        logger.warning("clinical_rules.yaml not found at %s. No threshold rules loaded.", rules_path)
        return []

    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            raw_rules = data.get("rules", [])
            
        parsed = []
        for r in raw_rules:
            parsed.append(_Rule(
                lab=r["lab"],
                condition=r["condition"],
                operator=r["operator"],
                evidence_fmt=r["evidence_fmt"],
                confidence=r.get("confidence", "moderate"),
                severity=r.get("severity", "moderate"),
                limit=r.get("limit"),
                min_val=r.get("min"),
                max_val=r.get("max")
            ))
        logger.info("Loaded %d clinical rules from YAML", len(parsed))
        return parsed
    except Exception as e:
        logger.error("Failed to load clinical rules: %s", e)
        return []

# Dynamic rules list
_RULES: List[_Rule] = _load_clinical_rules()


# ---------------------------------------------------------------------------
# Lightweight rule-based analysis
# ---------------------------------------------------------------------------

def _diagnose_from_labs(labs: Dict[str, Any]) -> List[_Finding]:
    """Apply threshold rules to OCR-extracted lab values."""
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

        if not rule.matches(value):
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
    """Convert an OCREngine result dict to a free-text patient summary."""
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
    """Maps evidence/symptom codes to human-readable text."""

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
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    import numpy as np


def _mean_pooling(model_output: Any, attention_mask: Any, torch: Any) -> Any:
    """Token mean pooling (prototype notebook)."""
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, 1)
    denom = torch.clamp(mask.sum(1), min=1e-9)
    return summed / denom


class ClinicalBERTEmbedder:
    """Encodes text to 768-dim vectors using ``emilyalsentzer/Bio_ClinicalBERT``."""

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, device: Optional[str] = None) -> None:
        try:
            import importlib
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModel = getattr(transformers, "AutoModel")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
        except Exception as exc:
            raise ImportError("ClinicalBERTEmbedder requires 'torch' and 'transformers'.") from exc

        self._torch = torch
        self._F = importlib.import_module("torch.nn.functional")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device).eval()
        logger.info("ClinicalBERT loaded on %s", self.device)

    def encode_text(self, text: str) -> "np.ndarray":
        import numpy as np
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        embedding = _mean_pooling(outputs, inputs["attention_mask"], self._torch)
        embedding = self._F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].astype("float32")


class MedicalCaseSearcher:
    """FAISS-backed cosine-similarity search over medical case embeddings."""

    def __init__(self, index_dir: Path) -> None:
        try:
            import importlib
            faiss = importlib.import_module("faiss")
            import pickle
        except Exception as exc:
            raise ImportError("MedicalCaseSearcher requires 'faiss-cpu'.") from exc

        index_dir = Path(index_dir)
        self._faiss = faiss
        self.index = self._faiss.read_index(str(index_dir / "medical_cases.index"))
        with open(index_dir / "metadata_mapping.pkl", "rb") as fh:
            self.metadata = pickle.load(fh)
        logger.info("FAISS index loaded: %d cases", self.index.ntotal)

    def search(self, query_embedding: "np.ndarray", k: int = 5) -> List[Dict[str, Any]]:
        import numpy as np
        q = query_embedding.reshape(1, -1).astype("float32")
        self._faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "similarity": float(score),
                "pathology": self.metadata["pathologies"][idx],
                "symptoms": self.metadata["symptoms"][idx],
                "patient_id": self.metadata["patient_ids"][idx],
            })
        return results


class ArabicToEnglishTranslator:
    """Medical Arabic→English via Gemini."""

    def __init__(self, provider: GeminiProvider) -> None:
        self._provider = provider

    @staticmethod
    def is_arabic(text: str) -> bool:
        if not text or not text.strip(): return False
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return arabic_chars / max(len(text), 1) > 0.3

    async def translate(self, arabic_text: str) -> str:
        prompt = (
            "Translate the following Arabic medical symptoms to English.\n"
            "Use proper medical terminology, not literal translation.\n"
            "Return ONLY the English translation, nothing else.\n\n"
            f"Arabic symptoms: {arabic_text}\n\nEnglish translation:"
        )
        try:
            return await self._provider.generate_content(prompt)
        except Exception as exc:
            logger.warning("Arabic translation failed: %s", exc)
            return arabic_text


from app.schemas.ai import AIDiagnosisResponse


class FineTunedDiagnosisClassifier:
    """Load a fine-tuned ClinicalBERT classifier and run local inference."""

    def __init__(self, model_dir: Path | str, max_length: int = 256, device: Optional[str] = None) -> None:
        try:
            import importlib

            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
            AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")
        except Exception as exc:
            raise ImportError(
                "FineTunedDiagnosisClassifier requires 'torch' and 'transformers'."
            ) from exc

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Fine-tuned model directory not found: {self.model_dir}")

        label_map_path = self.model_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"Missing label_map.json in model directory: {self.model_dir}")

        with label_map_path.open("r", encoding="utf-8") as fh:
            label_map = json.load(fh)

        raw_id_to_label = label_map.get("id_to_label", {})
        self.id_to_label = {int(k): v for k, v in raw_id_to_label.items()}
        self.label_to_id = label_map.get("label_to_id", {})

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device).eval()
        logger.info("Fine-tuned ClinicalBERT classifier loaded from %s", self.model_dir)

    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = self._torch.softmax(logits, dim=0)
        pred_idx = int(self._torch.argmax(probs).item())

        top_probs, top_indices = self._torch.topk(probs, k=min(3, probs.shape[0]))
        top_predictions = [
            {
                "label": self.id_to_label.get(int(idx.item()), str(int(idx.item()))),
                "confidence": float(score.item()),
            }
            for score, idx in zip(top_probs, top_indices)
        ]

        return {
            "predicted_label": self.id_to_label.get(pred_idx, str(pred_idx)),
            "confidence": float(probs[pred_idx].item()),
            "top_predictions": top_predictions,
        }

class MedicalRAGAssistant:
    """Full RAG pipeline: ClinicalBERT encoder → FAISS retrieval → Gemini LLM."""

    MEDICAL_DISCLAIMER = (
        "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ IMPORTANT MEDICAL DISCLAIMER\n\n"
        "This response is generated by AI based on pattern matching with medical "
        "cases. It is NOT a substitute for professional medical advice.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    def __init__(
        self,
        embedder: ClinicalBERTEmbedder,
        searcher: MedicalCaseSearcher,
        gemini_api_key: str,
        model_name: str = "gemini-2.5-flash",
        *,
        translate_arabic: bool = True,
    ) -> None:
        self.embedder = embedder
        self.searcher = searcher
        self._translate_arabic = translate_arabic
        self._provider = GeminiProvider(api_key=gemini_api_key, model_name=model_name)
        self._translator = ArabicToEnglishTranslator(self._provider)

    def _build_prompt(self, user_symptoms: str, retrieved_cases: List[Dict[str, Any]]) -> str:
        context = "SIMILAR MEDICAL CASES FROM DATABASE:\n\n"
        for i, case in enumerate(retrieved_cases, 1):
            context += f"Case {i}: Diagnosis: {case['pathology']}, Symptoms: {case['symptoms']}\n"
        return (
            "You are a professional medical diagnostic assistant. Your goal is to provide a structured "
            "preliminary assessment based on patient symptoms and similar clinical cases.\n\n"
            f"PATIENT'S SYMPTOMS:\n{user_symptoms}\n\n"
            f"{context}\n"
            "Analyze the correlation between symptoms and database cases. provide follow-up questions."
        )

    async def query(self, patient_text: str, top_k: int = 5) -> Dict[str, Any]:
        query_text = patient_text
        if self._translate_arabic and self._translator.is_arabic(patient_text):
            query_text = await self._translator.translate(patient_text)
            
        embedding = self.embedder.encode_text(query_text)
        cases = self.searcher.search(embedding, k=top_k)
        prompt = self._build_prompt(query_text, cases)
        
        system_instruction = (
            "You are a professional medical AI. You must return a valid JSON object matching the requested schema. "
            "Be precise, clinical, and conservative in your assessments."
        )
        
        response_json = await self._provider.generate_content(
            prompt, 
            system_instruction=system_instruction,
            response_model=AIDiagnosisResponse
        )
        
        try:
            structured_data = json.loads(response_json)
            # Add disclaimer to narrative
            assessment = structured_data.get("assessment_summary", "")
            return {
                "retrieved_cases": cases,
                "response": assessment + self.MEDICAL_DISCLAIMER,
                "structured_diagnosis": structured_data,
                "rag_query_text": query_text,
            }
        except Exception as e:
            logger.error(f"Failed to parse structured RAG response: {e}")
            return {
                "retrieved_cases": cases,
                "response": response_json + self.MEDICAL_DISCLAIMER,
                "rag_query_text": query_text,
            }


class DiagnosisEngine:
    """Orchestrates lightweight and/or RAG-based diagnosis."""

    DISCLAIMER = "This is a rule-based AI system. Consult a professional."

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        gemini_api_key: Optional[str] = None,
        rag_top_k: int = 5,
        rag_translate_arabic: bool = True,
        use_finetuned_classifier: bool = False,
        finetuned_model_dir: Optional[Path | str] = None,
        classifier_max_length: int = 256,
        classifier_translate_arabic: bool = True,
    ) -> None:
        self._rag_assistant: Optional[MedicalRAGAssistant] = None
        self._classifier: Optional[FineTunedDiagnosisClassifier] = None
        self._classifier_translator: Optional[ArabicToEnglishTranslator] = None
        self._classifier_translate_arabic = classifier_translate_arabic
        self._rag_top_k = rag_top_k
        if use_rag:
            if not faiss_index_dir or not gemini_api_key:
                raise ValueError("RAG requires faiss_index_dir and gemini_api_key")
            self._rag_assistant = MedicalRAGAssistant(
                embedder=ClinicalBERTEmbedder(),
                searcher=MedicalCaseSearcher(Path(faiss_index_dir)),
                gemini_api_key=gemini_api_key,
                translate_arabic=rag_translate_arabic,
            )
        if use_finetuned_classifier:
            if not finetuned_model_dir:
                raise ValueError("Fine-tuned classifier requires finetuned_model_dir")
            self._classifier = FineTunedDiagnosisClassifier(
                model_dir=finetuned_model_dir,
                max_length=classifier_max_length,
            )
            if classifier_translate_arabic and gemini_api_key:
                self._classifier_translator = ArabicToEnglishTranslator(
                    GeminiProvider(api_key=gemini_api_key, model_name="gemini-2.5-flash")
                )
            elif classifier_translate_arabic and not gemini_api_key:
                logger.warning(
                    "classifier_translate_arabic=True but GEMINI_API_KEY is missing; classifier will use raw input text."
                )

    async def diagnose(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse *report* and return findings."""
        labs = report.get("labs", {}) or {}
        findings = _diagnose_from_labs(labs)
        
        result = {
            "findings": [
                {"condition": f.condition, "confidence": f.confidence, "evidence": f.evidence, "severity": f.severity}
                for f in findings
            ],
            "summary": f"Detected {len(findings)} potential findings." if findings else "No findings.",
            "disclaimer": self.DISCLAIMER,
        }

        if self._rag_assistant:
            combined = build_combined_text(report)
            rag_out = await self._rag_assistant.query(combined, top_k=self._rag_top_k)
            result["rag_response"] = rag_out["response"]
            result["retrieved_cases"] = rag_out["retrieved_cases"]

        if self._classifier:
            combined = build_combined_text(report)
            classifier_query_text = combined
            translated_from_arabic = False

            if (
                self._classifier_translate_arabic
                and self._classifier_translator
                and self._classifier_translator.is_arabic(combined)
            ):
                classifier_query_text = await self._classifier_translator.translate(combined)
                translated_from_arabic = classifier_query_text != combined

            classifier_prediction = self._classifier.predict(classifier_query_text)
            classifier_prediction["input_text"] = combined
            classifier_prediction["query_text"] = classifier_query_text
            classifier_prediction["translated_from_arabic"] = translated_from_arabic
            result["classifier_prediction"] = classifier_prediction

        return result


async def diagnose(report: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Diagnose lab abnormalities (async wrapper)."""
    return await DiagnosisEngine(**kwargs).diagnose(report)
