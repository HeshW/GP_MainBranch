from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models.common.ai_provider import BaseModelProvider
from models.common.provider_factory import create_model_provider

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
        "age", "sex", "clinical", "context", "positive", "negative", "symptoms",
        "normalized", "labs", "patient", "presenting",
    }
    _FEATURE_PATTERNS = {
        "fatigue": ("fatigue", "tired", "weak", "malaise"),
        "thirst": ("thirst", "thirsty"),
        "polyuria": ("polyuria", "urinating more often", "frequent urination", "urination"),
        "hyperglycemia": ("hyperglycemia", "elevated glucose", "marked_hyperglycemia", "glucose"),
        "chest_pain": ("chest pain", "chest pressure", "pain somewhere in your chest", "lower chest", "upper chest", "breast("),
        "exertional": ("exertion", "exertional", "exercise", "effort"),
        "rest_relief": ("improves with rest", "better with rest", "relief with rest"),
        "shortness_breath": ("shortness of breath", "out of breath", "difficulty breathing", "dyspnea"),
        "chest_tightness": ("chest tightness", "tightness"),
        "palpitations": ("palpitations", "heart is beating fast", "racing"),
        "viral_infection": ("viral infection",),
        "cough": ("cough",),
        "productive_cough": ("productive cough", "sputum", "phlegm"),
        "fever": ("fever",),
        "no_fever": ("negative_symptoms: fever", "without fever", "no fever", "denies fever"),
        "no_productive_cough": ("negative_symptoms: productive cough", "without productive cough", "no productive cough"),
        "sore_throat": ("sore throat", "throat pain", "pharynx"),
        "nasal_congestion": ("nasal congestion", "runny nose"),
        "wheezing": ("wheezing",),
        "abdominal_pain": ("abdominal pain", "epigastric pain"),
        "vomiting": ("vomiting", "vomited"),
        "diarrhea": ("diarrhea",),
        "hoarseness": ("hoarseness", "hoarse voice"),
        "weight_loss": ("weight loss", "losing weight", "unexplained weight loss"),
        "headache": ("headache", "head pain"),
        "photophobia": ("photophobia", "sensitivity to light"),
        "dysuria": ("dysuria", "burning urination", "burning when urinating"),
        "urinary_frequency": ("urinary frequency", "frequent urination"),
        "urgency": ("urgency", "urgent urination"),
        "suprapubic_pain": ("suprapubic", "suprapubic pain"),
        "sudden": ("sudden", "abrupt", "suddenly"),
        "hemoptysis": ("hemoptysis", "coughing blood", "coughing up blood"),
        "night_sweats": ("night sweats", "increased sweating", "sweating"),
        "leg_swelling": ("leg swelling", "swollen leg"),
        "immobility": ("immobility", "prolonged sitting"),
        "pleuritic": ("pleuritic", "worse when breathing", "worsens with breathing"),
        "one_sided": ("one sided", "one-sided", "unilateral"),
        "orthopnea": ("lying flat", "worse lying flat", "orthopnea"),
        "ear_pain": ("ear pain", "pulling at the ear", "ear infection"),
        "facial_pain": ("facial pain", "forehead pain", "cheek pain", "sinus pressure"),
        "itchy_eyes": ("itchy eyes", "itching in one or both eyes"),
        "hay_fever": ("hay fever", "pollen", "allergies"),
        "stridor": ("stridor", "high-pitched", "high pitched", "breathing in"),
        "drooling": ("drooling", "more saliva"),
        "barking_cough": ("barking", "croup", "whooping cough"),
        "post_tussive_vomiting": ("vomiting after coughing", "vomit after coughing", "post-tussive"),
        "fish_exposure": ("fish", "eating fish"),
        "flushing": ("flushing", "cheeks turned red", "turned red"),
        "food_allergy": ("food allergy", "allergen", "allergy to"),
        "irregular_heartbeat": ("irregular heartbeat", "irregular rhythm"),
        "sudden_palpitations": ("sudden rapid heartbeat", "start and stop abruptly", "energy drinks"),
        "positional_chest_pain": ("worse lying down", "better sitting", "sitting forward"),
        "forceful_vomiting": ("forceful vomiting", "violent vomiting", "repeated vomiting"),
        "groin_pain": ("groin", "testicular", "testicle"),
        "ascending_weakness": ("ascending weakness", "both legs and arms", "tingling", "numbness"),
        "bulbar_weakness": ("drooping eyelids", "double vision", "difficulty speaking", "difficulty swallowing", "fatigable"),
        "dystonia_medication": ("antipsychotic", "neck spasm", "jaw stiffness", "dystonic"),
        "joint_pain": ("joint pain", "wrist", "shoulder", "photosensitive"),
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
        "hoarseness",
        "weight_loss",
        "hyperglycemia",
        "headache",
        "photophobia",
        "dysuria",
        "urinary_frequency",
        "urgency",
        "hemoptysis",
        "night_sweats",
        "leg_swelling",
        "immobility",
        "pleuritic",
        "one_sided",
        "ear_pain",
        "facial_pain",
        "itchy_eyes",
        "stridor",
        "drooling",
        "post_tussive_vomiting",
        "fish_exposure",
        "flushing",
        "food_allergy",
        "irregular_heartbeat",
        "sudden_palpitations",
        "forceful_vomiting",
        "groin_pain",
        "ascending_weakness",
        "bulbar_weakness",
        "dystonia_medication",
    }
    _OUT_OF_SCOPE_SIGNAL_FEATURES = {
        "diabetes_hyperglycemia": ("thirst", "polyuria", "hyperglycemia"),
        "uti_cystitis": ("dysuria", "urinary_frequency", "urgency", "suprapubic_pain"),
    }
    # Rerank scores are normalized to 0..1 where possible. Embedding similarity
    # remains the anchor, while sparse clinical signals can rescue cases that
    # share explicit symptoms/labs/demographics with the query.
    RERANK_WEIGHT_EMBEDDING = float(os.getenv("RAG_RERANK_WEIGHT_EMBEDDING", "0.34"))
    RERANK_WEIGHT_SYMPTOM_OVERLAP = float(os.getenv("RAG_RERANK_WEIGHT_SYMPTOM_OVERLAP", "0.22"))
    RERANK_WEIGHT_LEXICAL = float(os.getenv("RAG_RERANK_WEIGHT_LEXICAL", "0.10"))
    RERANK_WEIGHT_FEATURE_ALIGNMENT = float(os.getenv("RAG_RERANK_WEIGHT_FEATURE_ALIGNMENT", "0.12"))
    RERANK_WEIGHT_LAB_MATCH = float(os.getenv("RAG_RERANK_WEIGHT_LAB_MATCH", "0.08"))
    RERANK_WEIGHT_DEMOGRAPHIC = float(os.getenv("RAG_RERANK_WEIGHT_DEMOGRAPHIC", "0.04"))
    RERANK_WEIGHT_DISEASE_FAMILY = float(os.getenv("RAG_RERANK_WEIGHT_DISEASE_FAMILY", "0.12"))
    RERANK_PENALTY_MISMATCH = float(os.getenv("RAG_RERANK_PENALTY_MISMATCH", "0.18"))
    RERANK_PENALTY_PATHOLOGY = float(os.getenv("RAG_RERANK_PENALTY_PATHOLOGY", "0.22"))
    SEARCH_EXPANSION_MULTIPLIER = int(os.getenv("RAG_SEARCH_EXPANSION_MULTIPLIER", "100"))
    SEARCH_EXPANSION_MIN = int(os.getenv("RAG_SEARCH_EXPANSION_MIN", "500"))
    FAISS_NPROBE = int(os.getenv("RAG_FAISS_NPROBE", "16"))

    @staticmethod
    def _normalize_label(value: str) -> str:
        return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())

    @classmethod
    def configure_rerank_weights(
        cls,
        *,
        embedding: Optional[float] = None,
        symptom_overlap: Optional[float] = None,
        lexical: Optional[float] = None,
        feature_alignment: Optional[float] = None,
        lab_match: Optional[float] = None,
        demographic: Optional[float] = None,
        disease_family: Optional[float] = None,
        mismatch_penalty: Optional[float] = None,
        pathology_penalty: Optional[float] = None,
    ) -> None:
        if embedding is not None:
            cls.RERANK_WEIGHT_EMBEDDING = float(embedding)
        if symptom_overlap is not None:
            cls.RERANK_WEIGHT_SYMPTOM_OVERLAP = float(symptom_overlap)
        if lexical is not None:
            cls.RERANK_WEIGHT_LEXICAL = float(lexical)
        if feature_alignment is not None:
            cls.RERANK_WEIGHT_FEATURE_ALIGNMENT = float(feature_alignment)
        if lab_match is not None:
            cls.RERANK_WEIGHT_LAB_MATCH = float(lab_match)
        if demographic is not None:
            cls.RERANK_WEIGHT_DEMOGRAPHIC = float(demographic)
        if disease_family is not None:
            cls.RERANK_WEIGHT_DISEASE_FAMILY = float(disease_family)
        if mismatch_penalty is not None:
            cls.RERANK_PENALTY_MISMATCH = float(mismatch_penalty)
        if pathology_penalty is not None:
            cls.RERANK_PENALTY_PATHOLOGY = float(pathology_penalty)

    @classmethod
    def configure_search_expansion(
        cls,
        *,
        multiplier: Optional[int] = None,
        minimum: Optional[int] = None,
    ) -> None:
        if multiplier is not None:
            cls.SEARCH_EXPANSION_MULTIPLIER = max(1, int(multiplier))
        if minimum is not None:
            cls.SEARCH_EXPANSION_MIN = max(1, int(minimum))

    def __init__(self, index_dir: Path, *, allow_unsafe_pickle: bool = False) -> None:
        try:
            import importlib
            faiss = importlib.import_module("faiss")
        except Exception as exc:
            raise ImportError("MedicalCaseSearcher requires 'faiss-cpu'.") from exc

        index_dir = Path(index_dir)
        self._faiss = faiss
        self.index = self._faiss.read_index(str(index_dir / "medical_cases.index"))
        self._configure_index_search()
        self.metadata = self._load_metadata(index_dir, allow_unsafe_pickle=allow_unsafe_pickle)
        logger.info("FAISS index loaded: %d cases", self.index.ntotal)

    def _configure_index_search(self) -> None:
        if not hasattr(self.index, "nprobe"):
            return
        nlist = int(getattr(self.index, "nlist", self.FAISS_NPROBE) or self.FAISS_NPROBE)
        self.index.nprobe = max(1, min(self.FAISS_NPROBE, nlist))
        logger.info("Configured FAISS IVF nprobe=%s", self.index.nprobe)

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

    @staticmethod
    def _normalized_embedding_score(value: float) -> float:
        # FAISS inner-product scores for normalized embeddings are cosine-like
        # values in [-1, 1]. Clamp after mapping to a 0..1 feature.
        return max(0.0, min((float(value) + 1.0) / 2.0, 1.0))

    @classmethod
    def _extract_demographics(cls, text: str) -> tuple[Optional[int], Optional[str]]:
        age = None
        sex = None
        age_match = re.search(r"\bage\s*:\s*(\d{1,3})\b", text, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"\b(\d{1,3})\s*(?:year|yr)s?\s*old\b", text, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
        sex_match = re.search(r"\bsex\s*:\s*([MF])\b", text, re.IGNORECASE)
        if not sex_match:
            sex_match = re.search(r"\b(female|male|[MF])\b", text, re.IGNORECASE)
        if sex_match:
            sex = sex_match.group(1).upper()[0]
        return age, sex

    @classmethod
    def _demographic_alignment_score(cls, query_text: str, case_text: str) -> float:
        query_age, query_sex = cls._extract_demographics(query_text)
        case_age, case_sex = cls._extract_demographics(case_text)
        scores: list[float] = []
        if query_age is not None and case_age is not None:
            age_gap = abs(query_age - case_age)
            scores.append(max(0.0, 1.0 - (age_gap / 60.0)))
        if query_sex and case_sex:
            scores.append(1.0 if query_sex == case_sex else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    @classmethod
    def _lab_match_score(cls, query_text: str, pathology: str, case_text: str) -> float:
        normalized_query = query_text.lower()
        normalized_pathology = cls._normalize_label(pathology)
        normalized_case = case_text.lower()
        score = 0.0
        has_glucose_signal = any(
            term in normalized_query
            for term in ("marked_hyperglycemia", "elevated_glucose", "hyperglycemia", "glucose=")
        )
        if has_glucose_signal:
            if any(term in normalized_pathology for term in ("diabetes", "hyperglycemia", "prediabetes")):
                score = max(score, 1.0)
            elif "diabetes" in normalized_case:
                score = max(score, 0.4)
        if "low_hemoglobin" in normalized_query and "anemia" in normalized_pathology:
            score = max(score, 1.0)
        return score

    @classmethod
    def _disease_family_hint_score(
        cls,
        query_features: set[str],
        pathology: str,
        query_text: str,
    ) -> float:
        normalized_pathology = cls._normalize_label(pathology)
        normalized_query = query_text.lower()
        score = 0.0

        if {"chest_pain", "exertional", "rest_relief"}.issubset(query_features):
            if "stable angina" in normalized_pathology:
                score = max(score, 1.0)
            elif "unstable angina" in normalized_pathology:
                score = max(score, 0.35)

        bronchospasm_context = (
            "wheezing" in query_features
            and ("shortness_breath" in query_features or "chest_tightness" in query_features)
        )
        low_infection_context = "no_fever" in query_features or "no_productive_cough" in query_features
        if bronchospasm_context:
            if any(term in normalized_pathology for term in ("bronchospasm", "asthma")):
                score = max(score, 1.0 if low_infection_context else 0.8)
            elif "copd" in normalized_pathology and any(term in normalized_query for term in ("smoke", "copd", "older")):
                score = max(score, 0.45)

        if "headache" in query_features and "photophobia" in query_features:
            if any(term in normalized_pathology for term in ("migraine", "cluster headache")):
                score = max(score, 0.8)

        if {"dysuria", "urinary_frequency"}.issubset(query_features):
            if any(term in normalized_pathology for term in ("urinary tract infection", "cystitis", "uti")):
                score = max(score, 1.0)

        if {"thirst", "polyuria"}.issubset(query_features) or "hyperglycemia" in query_features:
            if any(term in normalized_pathology for term in ("diabetes", "hyperglycemia", "prediabetes")):
                score = max(score, 1.0)

        if {"sudden", "pleuritic", "leg_swelling"}.intersection(query_features) and "immobility" in query_features:
            if "pulmonary embolism" in normalized_pathology:
                score = max(score, 1.0)
        if "one_sided" in query_features and "sudden" in query_features and "shortness_breath" in query_features:
            if "spontaneous pneumothorax" in normalized_pathology:
                score = max(score, 1.0)
            elif "pulmonary embolism" in normalized_pathology:
                score = max(score, 0.3)
        if {"hemoptysis", "weight_loss"}.issubset(query_features):
            if "tuberculosis" in normalized_pathology and "night_sweats" in query_features:
                score = max(score, 1.0)
            elif "pulmonary neoplasm" in normalized_pathology:
                score = max(score, 0.75)
        if "dry cough" in normalized_query and "swollen lymph" in normalized_query:
            if "sarcoidosis" in normalized_pathology:
                score = max(score, 1.0)
        if {"sore_throat", "fever"}.issubset(query_features):
            if "viral pharyngitis" in normalized_pathology and any(
                term in normalized_query for term in ("tonsil", "daycare", "painful swallowing")
            ):
                score = max(score, 0.85)
            elif "urti" in normalized_pathology and "runny nose" in normalized_query:
                score = max(score, 0.8)
        if "hoarseness" in query_features:
            if "acute laryngitis" in normalized_pathology:
                score = max(score, 1.0)
        if "ear_pain" in query_features:
            if "acute otitis media" in normalized_pathology:
                score = max(score, 1.0)
        if "facial_pain" in query_features and "nasal_congestion" in query_features:
            if "chronic rhinosinusitis" in normalized_pathology and "chronic" in normalized_query:
                score = max(score, 1.0)
            elif "acute rhinosinusitis" in normalized_pathology:
                score = max(score, 1.0)
        if {"itchy_eyes", "hay_fever"}.intersection(query_features):
            if "allergic sinusitis" in normalized_pathology:
                score = max(score, 1.0)
        if {"stridor", "barking_cough"}.intersection(query_features):
            if "croup" in normalized_pathology and any(term in normalized_query for term in ("toddler", "child", "2 year", "infant")):
                score = max(score, 1.0)
            elif "epiglottitis" in normalized_pathology and "drooling" in query_features:
                score = max(score, 1.0)
        if "post_tussive_vomiting" in query_features:
            if "whooping cough" in normalized_pathology:
                score = max(score, 1.0)
        if "fever" in query_features and any(term in normalized_query for term in ("body aches", "myalgia", "abrupt")):
            if "influenza" in normalized_pathology:
                score = max(score, 1.0)
        if {"food_allergy", "shortness_breath"}.issubset(query_features):
            if "anaphylaxis" in normalized_pathology:
                score = max(score, 1.0)
        if {"fish_exposure", "flushing"}.issubset(query_features):
            if "scombroid" in normalized_pathology:
                score = max(score, 1.0)
        if "irregular_heartbeat" in query_features:
            if "atrial fibrillation" in normalized_pathology:
                score = max(score, 1.0)
        if "sudden_palpitations" in query_features:
            if "psvt" in normalized_pathology:
                score = max(score, 1.0)
        if "positional_chest_pain" in query_features:
            if "pericarditis" in normalized_pathology:
                score = max(score, 1.0)
        if "viral_infection" in query_features and "chest_pain" in query_features and "shortness_breath" in query_features:
            if "myocarditis" in normalized_pathology:
                score = max(score, 0.85)
        if "forceful_vomiting" in query_features and "chest_pain" in query_features:
            if "boerhaave" in normalized_pathology:
                score = max(score, 1.0)
        if {"abdominal_pain", "weight_loss"}.issubset(query_features) and "epigastric" in normalized_query:
            if "pancreatic neoplasm" in normalized_pathology:
                score = max(score, 1.0)
        if "groin_pain" in query_features:
            if "inguinal hernia" in normalized_pathology:
                score = max(score, 1.0)
        if "ascending_weakness" in query_features:
            if "guillain" in normalized_pathology:
                score = max(score, 1.0)
        if "bulbar_weakness" in query_features:
            if "myasthenia gravis" in normalized_pathology:
                score = max(score, 1.0)
        if "dystonia_medication" in query_features:
            if "acute dystonic" in normalized_pathology:
                score = max(score, 1.0)
        if "joint_pain" in query_features:
            if normalized_pathology == "sle":
                score = max(score, 0.9)

        return score

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
        if "no_fever" in features:
            features.discard("fever")
        if "no_productive_cough" in features:
            features.discard("productive_cough")
        return features

    @classmethod
    def detect_out_of_scope_signals(
        cls,
        query_text: str,
        query_symptoms: Optional[List[str]] = None,
    ) -> List[str]:
        query_features = cls._extract_feature_flags(query_text, query_symptoms)
        detected: list[str] = []
        if (
            {"thirst", "polyuria"}.issubset(query_features)
            or "hyperglycemia" in query_features
            or any(term in (query_text or "").lower() for term in ("fasting glucose", "hba1c"))
        ):
            detected.append("diabetes_hyperglycemia")
        if {"dysuria", "urinary_frequency"}.issubset(query_features) or (
            "urgency" in query_features and "suprapubic_pain" in query_features
        ):
            detected.append("uti_cystitis")
        return detected

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
    def _clinical_context_mismatch_penalty(
        cls,
        query_features: set[str],
        *,
        pathology: str,
        query_text: str,
    ) -> float:
        normalized_pathology = cls._normalize_label(pathology)
        normalized_query = query_text.lower()
        penalty = 0.0
        if "copd" in normalized_pathology:
            query_age, _ = cls._extract_demographics(query_text)
            has_copd_context = any(term in normalized_query for term in ("copd", "smoke", "smoker"))
            if query_age is not None and query_age < 40 and not has_copd_context:
                penalty += 0.22
            if "no_fever" in query_features and "no_productive_cough" in query_features and not has_copd_context:
                penalty += 0.18
        if "unstable angina" in normalized_pathology and "rest_relief" in query_features and "exertional" in query_features:
            penalty += 0.12
        if any(term in normalized_pathology for term in ("pulmonary neoplasm", "pancreatic neoplasm")):
            if "weight_loss" not in query_features and "chronic" not in normalized_query:
                penalty += 0.16
        diabetes_context = (
            {"thirst", "polyuria"}.issubset(query_features)
            or "hyperglycemia" in query_features
            or any(term in normalized_query for term in ("glucose", "fasting glucose", "hba1c"))
        )
        if diabetes_context and not any(
            term in normalized_pathology for term in ("diabetes", "hyperglycemia", "prediabetes")
        ):
            penalty += 0.36
        uti_context = {"dysuria", "urinary_frequency"}.issubset(query_features) or "urgency" in query_features
        if uti_context and not any(
            term in normalized_pathology for term in ("urinary tract infection", "cystitis", "uti")
        ):
            penalty += 0.32
        return min(1.0, penalty)

    @classmethod
    def _pathology_mismatch_penalty(
        cls,
        *,
        query_text: str,
        query_features: set[str],
        pathology: str,
        case_text: str,
    ) -> float:
        normalized_query = (query_text or "").lower()
        normalized_pathology = cls._normalize_label(pathology)
        normalized_case = (case_text or "").lower()
        penalty = 0.0

        if "ebola" in normalized_pathology:
            has_severe_viral_context = (
                ("fever" in query_features)
                and any(term in normalized_query for term in ("vomit", "diarrhea", "bleed", "hemorrhag", "travel"))
            )
            if not has_severe_viral_context:
                penalty += 0.46

        if "guillain" in normalized_pathology:
            if not any(term in normalized_query for term in ("weakness", "ascending", "tingling", "paresthesia")):
                penalty += 0.28

        if "larygospasm" in normalized_pathology:
            if not any(term in normalized_query for term in ("stridor", "high pitched", "breathing in", "inspiration")):
                penalty += 0.24

        if "chagas" in normalized_pathology:
            if not any(term in normalized_query for term in ("travel", "latin", "lymph node", "swollen")):
                penalty += 0.18

        if "myocarditis" in normalized_pathology:
            if any(term in normalized_query for term in ("one side", "one-sided", "unilateral")) and any(
                term in normalized_query for term in ("sudden", "suddenly")
            ):
                penalty += 0.18

        if "pulmonary neoplasm" in normalized_pathology:
            if "weight loss" not in normalized_query and "chronic" not in normalized_query:
                penalty += 0.14

        # Penalize retrieval rows that are largely encoded/noisy when query is natural.
        if normalized_query and cls._looks_like_encoded_symptoms(normalized_case):
            penalty += 0.10

        return min(1.0, penalty)

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

        def annotate(item: Dict[str, Any]) -> Dict[str, Any]:
            case_text = str(item.get("case_text", "") or item.get("symptoms", ""))
            embedding_score = float(item.get("similarity", 0.0))
            normalized_embedding = cls._normalized_embedding_score(embedding_score)
            symptom_overlap = cls._symptom_overlap_score(symptom_terms, case_text)
            lexical_overlap = cls._lexical_overlap_score(query_text, case_text)
            feature_alignment = cls._feature_alignment_score(query_features, case_text)
            lab_match = cls._lab_match_score(query_text, str(item.get("pathology", "")), case_text)
            demographic_alignment = cls._demographic_alignment_score(query_text, case_text)
            disease_family_hint = cls._disease_family_hint_score(
                query_features,
                str(item.get("pathology", "")),
                query_text,
            )
            mismatch_penalty = cls._feature_mismatch_penalty(query_features, case_text)
            pathology_penalty = cls._pathology_mismatch_penalty(
                query_text=query_text,
                query_features=query_features,
                pathology=str(item.get("pathology", "")),
                case_text=case_text,
            )
            clinical_context_penalty = cls._clinical_context_mismatch_penalty(
                query_features,
                pathology=str(item.get("pathology", "")),
                query_text=query_text,
            )
            blended = (
                (cls.RERANK_WEIGHT_EMBEDDING * normalized_embedding)
                + (cls.RERANK_WEIGHT_SYMPTOM_OVERLAP * symptom_overlap)
                + (cls.RERANK_WEIGHT_LEXICAL * lexical_overlap)
                + (cls.RERANK_WEIGHT_FEATURE_ALIGNMENT * feature_alignment)
                + (cls.RERANK_WEIGHT_LAB_MATCH * lab_match)
                + (cls.RERANK_WEIGHT_DEMOGRAPHIC * demographic_alignment)
                + (cls.RERANK_WEIGHT_DISEASE_FAMILY * disease_family_hint)
                - (cls.RERANK_PENALTY_MISMATCH * mismatch_penalty)
                - (cls.RERANK_PENALTY_PATHOLOGY * pathology_penalty)
                - clinical_context_penalty
            )
            annotated = dict(item)
            annotated.update(
                {
                    "embedding_score_normalized": round(normalized_embedding, 6),
                    "symptom_overlap": round(symptom_overlap, 6),
                    "lexical_overlap": round(lexical_overlap, 6),
                    "feature_alignment": round(feature_alignment, 6),
                    "lab_match": round(lab_match, 6),
                    "demographic_alignment": round(demographic_alignment, 6),
                    "disease_family_hint": round(disease_family_hint, 6),
                    "mismatch_penalty": round(mismatch_penalty, 6),
                    "pathology_penalty": round(pathology_penalty, 6),
                    "clinical_context_penalty": round(clinical_context_penalty, 6),
                    "rerank_score": round(max(0.0, min(blended, 1.0)), 6),
                }
            )
            return annotated

        annotated_results = [annotate(item) for item in results]
        reranked = sorted(
            annotated_results,
            key=lambda item: (
                float(item.get("rerank_score", 0.0)),
                float(item.get("disease_family_hint", 0.0)),
                float(item.get("feature_alignment", 0.0)),
                float(item.get("symptom_overlap", 0.0)),
            ),
            reverse=True,
        )
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
        search_k = min(
            max(k * self.SEARCH_EXPANSION_MULTIPLIER, self.SEARCH_EXPANSION_MIN),
            self.index.ntotal,
        )
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
    def __init__(self, provider: BaseModelProvider) -> None:
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

        model = None
        load_attempts = [
            {"local_files_only": True, "low_cpu_mem_usage": True},
            {"local_files_only": True},
            {"low_cpu_mem_usage": True},
            {},
        ]
        for kwargs in load_attempts:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    **kwargs,
                )
                break
            except TypeError:
                continue
        if model is None:
            raise TypeError(
                "Unable to load fine-tuned model with compatible from_pretrained kwargs."
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
        llm_provider: str = "gemini",
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openrouter_site_url: Optional[str] = None,
        openrouter_app_name: str = "GP Medical Analysis",
        openrouter_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-2.5-flash-lite",
    ) -> None:
        self.embedder = embedder
        self.searcher = searcher
        self._translate_arabic = translate_arabic
        _, self._provider, _ = create_model_provider(
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
            openrouter_base_url=openrouter_base_url,
            openrouter_site_url=openrouter_site_url,
            openrouter_app_name=openrouter_app_name,
            openrouter_api_key=openrouter_api_key,
        )
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
            "RAG retrieval completed, but the AI summary is currently unavailable. "
            f"The most similar case suggests {top_case.get('pathology', 'an unknown condition')}. "
            f"Top retrieved conditions: {related_conditions}."
        )

    @staticmethod
    def _build_confidence_metadata(
        retrieved_cases: List[Dict[str, Any]],
        *,
        detected_out_of_scope_signals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        detected_out_of_scope_signals = list(detected_out_of_scope_signals or [])
        if not retrieved_cases:
            return {
                "level": "none",
                "usable_for_fusion": False,
                "scope_status": "out_of_scope_or_low_confidence",
                "detected_out_of_scope_signals": detected_out_of_scope_signals,
                "top_rerank_score": 0.0,
                "score_gap": 0.0,
                "reason": "no retrieved cases",
            }

        top = retrieved_cases[0]
        second = retrieved_cases[1] if len(retrieved_cases) > 1 else {}
        top_rerank = float(top.get("rerank_score", top.get("similarity", 0.0)) or 0.0)
        second_rerank = float(second.get("rerank_score", second.get("similarity", 0.0)) or 0.0)
        score_gap = max(0.0, top_rerank - second_rerank)
        explicit_signal = max(
            float(top.get("symptom_overlap", 0.0) or 0.0),
            float(top.get("feature_alignment", 0.0) or 0.0),
            float(top.get("lab_match", 0.0) or 0.0),
            float(top.get("disease_family_hint", 0.0) or 0.0),
        )
        penalty = max(
            float(top.get("mismatch_penalty", 0.0) or 0.0),
            float(top.get("pathology_penalty", 0.0) or 0.0),
            float(top.get("clinical_context_penalty", 0.0) or 0.0),
        )

        if top_rerank >= 0.58 and explicit_signal >= 0.30 and penalty < 0.45:
            level = "high"
        elif top_rerank >= 0.44 and explicit_signal >= 0.18 and penalty < 0.60:
            level = "moderate"
        else:
            level = "low"
        scope_status = "supported_scope"
        if level == "low" or detected_out_of_scope_signals:
            scope_status = "out_of_scope_or_low_confidence"

        return {
            "level": level,
            "usable_for_fusion": level in {"high", "moderate"} and not detected_out_of_scope_signals,
            "scope_status": scope_status,
            "detected_out_of_scope_signals": detected_out_of_scope_signals,
            "top_rerank_score": round(top_rerank, 6),
            "score_gap": round(score_gap, 6),
            "explicit_signal": round(explicit_signal, 6),
            "max_penalty": round(penalty, 6),
            "reason": (
                "sufficient explicit clinical overlap"
                if level in {"high", "moderate"} and not detected_out_of_scope_signals
                else "detected out-of-scope clinical signals"
                if detected_out_of_scope_signals
                else "weak explicit overlap or high mismatch penalty"
            ),
        }

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
        detected_out_of_scope_signals = MedicalCaseSearcher.detect_out_of_scope_signals(
            query_text,
            query_symptoms,
        )
        confidence = self._build_confidence_metadata(
            cases,
            detected_out_of_scope_signals=detected_out_of_scope_signals,
        )
        fallback_response = self._build_retrieval_only_response(cases)
        return {
            "retrieved_cases": cases,
            "response": fallback_response + self.MEDICAL_DISCLAIMER,
            "rag_query_text": query_text,
            "rag_mode": "retrieval_only",
            "rag_confidence": confidence,
            "rag_scope_status": confidence.get("scope_status"),
            "detected_out_of_scope_signals": detected_out_of_scope_signals,
        }
