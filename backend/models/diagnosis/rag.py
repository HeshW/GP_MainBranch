from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models.common.ai_provider import GeminiProvider

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


def mean_pooling(model_output: Any, attention_mask: Any, torch: Any) -> Any:
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, 1)
    denom = torch.clamp(mask.sum(1), min=1e-9)
    return summed / denom


class ClinicalBERTEmbedder:
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(
        self,
        device: Optional[str] = None,
        model_dir: Optional[Path | str] = None,
    ) -> None:
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
        resolved_model_source = self._resolve_model_source(model_dir)
        self.model_source = str(resolved_model_source)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_source)
        self.model = AutoModel.from_pretrained(resolved_model_source)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device).eval()
        logger.info("ClinicalBERT loaded from %s on %s", self.model_source, self.device)

    @classmethod
    def _resolve_model_source(cls, model_dir: Optional[Path | str]) -> str | Path:
        if model_dir:
            resolved_path = Path(model_dir)
            if not resolved_path.exists():
                raise FileNotFoundError(
                    "Local ClinicalBERT directory was not found: "
                    f"{resolved_path}. Download the model files and place them there."
                )
            return resolved_path
        return cls.MODEL_NAME

    def encode_text(self, text: str) -> "np.ndarray":
        import numpy as np
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        embedding = mean_pooling(outputs, inputs["attention_mask"], self._torch)
        embedding = self._F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].astype("float32")


class MedicalCaseSearcher:
    _TEXT_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z()/-]+")
    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
        "have", "how", "in", "is", "it", "of", "on", "or", "related", "somewhere",
        "that", "the", "to", "today", "usual", "with", "you", "your",
    }
    _FEATURE_PATTERNS = {
        "fatigue": ("fatigue", "tired", "weak", "malaise"),
        "thirst": ("thirst", "thirsty"),
        "polyuria": ("polyuria", "urinating more often", "frequent urination"),
        "chest_pain": ("chest pain", "pain somewhere in your chest", "lower chest", "upper chest", "breast("),
        "shortness_breath": ("shortness of breath", "out of breath", "difficulty breathing", "dyspnea"),
        "palpitations": ("palpitations", "heart is beating fast", "racing"),
        "viral_infection": ("viral infection",),
        "cough": ("cough",),
        "fever": ("fever",),
        "sore_throat": ("sore throat", "throat pain", "pharynx"),
        "nasal_congestion": ("nasal congestion", "runny nose"),
        "wheezing": ("wheezing",),
        "abdominal_pain": ("abdominal pain", "epigastric pain"),
        "vomiting": ("vomiting", "vomited"),
        "diarrhea": ("diarrhea",),
    }
    _DISCRIMINATIVE_FEATURES = {
        "thirst",
        "polyuria",
        "chest_pain",
        "shortness_breath",
        "palpitations",
        "viral_infection",
        "sore_throat",
        "nasal_congestion",
        "wheezing",
        "abdominal_pain",
        "vomiting",
        "diarrhea",
    }

    def __init__(self, index_dir: Path, *, allow_unsafe_pickle: bool = False) -> None:
        try:
            import importlib
            faiss = importlib.import_module("faiss")
        except Exception as exc:
            raise ImportError("MedicalCaseSearcher requires 'faiss-cpu'.") from exc

        index_dir = Path(index_dir)
        self._faiss = faiss
        self.index = self._faiss.read_index(str(index_dir / "medical_cases.index"))
        self.metadata = self._load_metadata(index_dir, allow_unsafe_pickle=allow_unsafe_pickle)
        logger.info("FAISS index loaded: %d cases", self.index.ntotal)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def _load_metadata(cls, index_dir: Path, *, allow_unsafe_pickle: bool) -> Dict[str, Any]:
        json_path = index_dir / "metadata_mapping.json"
        if json_path.is_file():
            with json_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not isinstance(metadata, dict):
                raise ValueError("metadata_mapping.json must contain a JSON object.")
            return metadata

        pickle_path = index_dir / "metadata_mapping.pkl"
        if not pickle_path.is_file():
            raise FileNotFoundError(
                "Missing RAG metadata mapping file. Expected metadata_mapping.json or metadata_mapping.pkl."
            )

        hash_path = index_dir / "metadata_mapping.pkl.sha256"
        if not allow_unsafe_pickle:
            if not hash_path.is_file():
                raise ValueError(
                    "Refusing to load metadata_mapping.pkl without hash verification. "
                    "Provide metadata_mapping.pkl.sha256 or set ALLOW_UNSAFE_PICKLE_METADATA=true "
                    "for trusted local artifacts."
                )
            expected_hash = hash_path.read_text(encoding="utf-8").strip().split()[0].lower()
            actual_hash = cls._sha256_file(pickle_path)
            if expected_hash != actual_hash:
                raise ValueError(
                    "metadata_mapping.pkl hash verification failed. "
                    f"expected={expected_hash} actual={actual_hash}"
                )

        with pickle_path.open("rb") as handle:
            metadata = pickle.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("metadata_mapping.pkl must deserialize to a dictionary.")
        return metadata

    @staticmethod
    def _looks_like_encoded_symptoms(text: str) -> bool:
        compact = text.strip()
        if not compact:
            return False
        return bool(re.fullmatch(r"[\d,\s@Vv._-]+", compact))

    def _get_metadata_text(self, idx: int) -> str:
        for key in ("combined_text", "case_text", "texts", "symptoms"):
            values = self.metadata.get(key)
            if isinstance(values, list) and idx < len(values):
                value = str(values[idx]).strip()
                if value:
                    return value
        return ""

    def _format_case_text(self, idx: int) -> Dict[str, Any]:
        raw_text = self._get_metadata_text(idx)
        if not raw_text:
            return {
                "case_text": "No case text available in current FAISS metadata.",
                "case_text_raw": "",
                "case_text_is_natural": False,
            }
        if self._looks_like_encoded_symptoms(raw_text):
            return {
                "case_text": "Encoded DDX evidence values in current FAISS metadata.",
                "case_text_raw": raw_text,
                "case_text_is_natural": False,
            }
        return {
            "case_text": raw_text,
            "case_text_raw": raw_text,
            "case_text_is_natural": True,
        }

    def metadata_has_natural_text(self) -> bool:
        sample_size = min(25, len(self.metadata.get("patient_ids", [])))
        if sample_size == 0:
            return False
        natural_count = 0
        for idx in range(sample_size):
            payload = self._format_case_text(idx)
            if payload["case_text_is_natural"]:
                natural_count += 1
        return natural_count > 0

    @classmethod
    def _tokenize_text(cls, text: str) -> set[str]:
        tokens = {
            match.group(0).lower()
            for match in cls._TEXT_TOKEN_PATTERN.finditer(text or "")
        }
        return {token for token in tokens if token not in cls._STOPWORDS and len(token) > 2}

    @classmethod
    def _symptom_overlap_score(cls, query_symptoms: List[str], case_text: str) -> float:
        normalized_query = [str(item).strip().lower() for item in query_symptoms if str(item).strip()]
        if not normalized_query:
            return 0.0
        case_text_lower = case_text.lower()
        matched = 0
        for symptom in normalized_query:
            if symptom in case_text_lower:
                matched += 1
        return matched / max(len(normalized_query), 1)

    @classmethod
    def _lexical_overlap_score(cls, query_text: str, case_text: str) -> float:
        query_tokens = cls._tokenize_text(query_text)
        case_tokens = cls._tokenize_text(case_text)
        if not query_tokens or not case_tokens:
            return 0.0
        overlap = query_tokens.intersection(case_tokens)
        return len(overlap) / max(len(query_tokens), 1)

    @classmethod
    def _extract_feature_flags(
        cls,
        query_text: str,
        query_symptoms: Optional[List[str]],
    ) -> set[str]:
        combined_text = " ".join(
            part for part in [query_text or "", " ".join(query_symptoms or [])] if part
        ).lower()
        features: set[str] = set()
        for feature_name, patterns in cls._FEATURE_PATTERNS.items():
            if any(pattern in combined_text for pattern in patterns):
                features.add(feature_name)
        return features

    @classmethod
    def _feature_alignment_score(
        cls,
        query_features: set[str],
        case_text: str,
    ) -> float:
        if not query_features:
            return 0.0
        case_features = cls._extract_feature_flags(case_text, None)
        if not case_features:
            return 0.0
        aligned = query_features.intersection(case_features)
        return len(aligned) / max(len(query_features), 1)

    @classmethod
    def _feature_mismatch_penalty(
        cls,
        query_features: set[str],
        case_text: str,
    ) -> float:
        case_features = cls._extract_feature_flags(case_text, None)
        discriminative_case_features = case_features.intersection(cls._DISCRIMINATIVE_FEATURES)
        if not discriminative_case_features:
            return 0.0
        mismatched = discriminative_case_features.difference(query_features)
        return len(mismatched) / max(len(discriminative_case_features), 1)

    @classmethod
    def _rerank_results(
        cls,
        results: List[Dict[str, Any]],
        *,
        query_text: str,
        query_symptoms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        symptom_terms = query_symptoms or []
        query_features = cls._extract_feature_flags(query_text, query_symptoms)

        def score(item: Dict[str, Any]) -> tuple[float, float, float]:
            case_text = str(item.get("case_text", "") or item.get("symptoms", ""))
            embedding_score = float(item.get("similarity", 0.0))
            symptom_overlap = cls._symptom_overlap_score(symptom_terms, case_text)
            lexical_overlap = cls._lexical_overlap_score(query_text, case_text)
            feature_alignment = cls._feature_alignment_score(query_features, case_text)
            mismatch_penalty = cls._feature_mismatch_penalty(query_features, case_text)
            blended = (
                (0.55 * embedding_score)
                + (0.28 * symptom_overlap)
                + (0.15 * lexical_overlap)
                + (0.22 * feature_alignment)
                - (0.24 * mismatch_penalty)
            )
            return (blended, feature_alignment, symptom_overlap)

        reranked = sorted(results, key=score, reverse=True)
        return reranked

    def search(
        self,
        query_embedding: "np.ndarray",
        k: int = 5,
        *,
        query_text: str = "",
        query_symptoms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        q = query_embedding.reshape(1, -1).astype("float32")
        self._faiss.normalize_L2(q)
        search_k = min(max(k * 80, 400), self.index.ntotal)
        scores, indices = self.index.search(q, search_k)
        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            patient_id = str(self.metadata["patient_ids"][idx])
            if patient_id.lower().startswith("test_"):
                continue

            case_payload = self._format_case_text(idx)
            results.append(
                {
                    "similarity": float(score),
                    "pathology": self.metadata["pathologies"][idx],
                    "patient_id": patient_id,
                    "symptoms": case_payload["case_text"],
                    "symptoms_raw": case_payload["case_text_raw"],
                    **case_payload,
                }
            )
        if query_text or query_symptoms:
            results = self._rerank_results(
                results,
                query_text=query_text,
                query_symptoms=query_symptoms,
            )

        return results[:k]


class ArabicToEnglishTranslator:
    def __init__(self, provider: GeminiProvider) -> None:
        self._provider = provider

    @staticmethod
    def is_arabic(text: str) -> bool:
        if not text or not text.strip():
            return False
        arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06ff")
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


class FineTunedDiagnosisClassifier:
    def __init__(self, model_dir: Path | str, max_length: int = 256, device: Optional[str] = None) -> None:
        try:
            import importlib
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
            AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")
        except Exception as exc:
            raise ImportError("FineTunedDiagnosisClassifier requires 'torch' and 'transformers'.") from exc

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Fine-tuned model directory not found: {self.model_dir}")
        label_map_path = self.model_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"Missing label_map.json in model directory: {self.model_dir}")
        with label_map_path.open("r", encoding="utf-8") as handle:
            label_map = json.load(handle)
        raw_id_to_label = (
            label_map.get("id_to_label")
            or label_map.get("id2label")
            or {}
        )
        raw_label_to_id = (
            label_map.get("label_to_id")
            or label_map.get("label2id")
            or {}
        )
        self.id_to_label = {int(key): value for key, value in raw_id_to_label.items()}
        self.label_to_id = raw_label_to_id
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        model_load_attempts: List[Dict[str, Any]] = [
            {"local_files_only": True, "low_cpu_mem_usage": True},
            {"local_files_only": True},
            {},
        ]
        model = None
        for load_kwargs in model_load_attempts:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    **load_kwargs,
                )
                break
            except TypeError:
                continue
        if model is None:
            raise TypeError(
                "Unable to load fine-tuned classifier with the available from_pretrained signatures."
            )
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device).eval()
        logger.info("Fine-tuned ClinicalBERT classifier loaded from %s", self.model_dir)

    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0]
        probs = self._torch.softmax(logits, dim=0)
        pred_idx = int(self._torch.argmax(probs).item())
        top_probs, top_indices = self._torch.topk(probs, k=min(3, probs.shape[0]))
        return {
            "predicted_label": self.id_to_label.get(pred_idx, str(pred_idx)),
            "confidence": float(probs[pred_idx].item()),
            "top_predictions": [
                {
                    "label": self.id_to_label.get(int(idx.item()), str(int(idx.item()))),
                    "confidence": float(score.item()),
                }
                for score, idx in zip(top_probs, top_indices)
            ],
        }


class MedicalRAGAssistant:
    MEDICAL_DISCLAIMER = (
        "\n\nIMPORTANT MEDICAL DISCLAIMER\n\n"
        "This response is generated by AI based on pattern matching with medical cases. "
        "It is NOT a substitute for professional medical advice."
    )

    def __init__(
        self,
        embedder: ClinicalBERTEmbedder,
        searcher: MedicalCaseSearcher,
        *,
        translate_arabic: bool = True,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.embedder = embedder
        self.searcher = searcher
        self._translate_arabic = translate_arabic
        self._provider = GeminiProvider(api_key=gemini_api_key, model_name=model_name) if gemini_api_key else None
        self._translator = ArabicToEnglishTranslator(self._provider) if self._provider else None

    @staticmethod
    def _build_retrieval_only_response(retrieved_cases: List[Dict[str, Any]]) -> str:
        if not retrieved_cases:
            return (
                "RAG retrieval completed, but no similar cases were found. "
                "An AI-generated summary is currently unavailable."
            )

        top_case = retrieved_cases[0]
        related_conditions = ", ".join(
            sorted({str(case.get("pathology", "unknown")) for case in retrieved_cases[:3]})
        )
        return (
            "RAG retrieval completed, but the Gemini summary is currently unavailable. "
            f"The most similar case suggests {top_case.get('pathology', 'an unknown condition')}. "
            f"Top retrieved conditions: {related_conditions}."
        )

    async def query(self, patient_text: str, top_k: int = 5, query_symptoms: Optional[List[str]] = None) -> Dict[str, Any]:
        query_text = patient_text
        if self._translate_arabic and self._translator and self._translator.is_arabic(patient_text):
            query_text = await self._translator.translate(patient_text)
        embedding = self.embedder.encode_text(query_text)
        cases = self.searcher.search(
            embedding,
            k=top_k,
            query_text=query_text,
            query_symptoms=query_symptoms,
        )
        fallback_response = self._build_retrieval_only_response(cases)
        return {
            "retrieved_cases": cases,
            "response": fallback_response + self.MEDICAL_DISCLAIMER,
            "rag_query_text": query_text,
            "rag_mode": "retrieval_only",
        }
