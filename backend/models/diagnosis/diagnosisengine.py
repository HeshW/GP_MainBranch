"""Production diagnosis engine orchestrating AI-first diagnosis with rule safety checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import logging

from models.common.language import detect_preferred_language, normalize_language
from models.common.provider_factory import create_model_provider
from .rag import (
    ArabicToEnglishTranslator,
    ClinicalBERTEmbedder,
    FineTunedDiagnosisClassifier,
    MedicalCaseSearcher,
    MedicalRAGAssistant,
)
from .rules import diagnose_from_labs, diagnose_from_symptoms
from .synthesis import DiagnosisResponseSynthesizer
from .text import build_combined_text

logger = logging.getLogger(__name__)


class DiagnosisEngine:
    """Orchestrates AI-first diagnosis with rule-based validation and escalation."""

    DISCLAIMER = "This is an AI-assisted medical decision support system. Consult a professional."
    CLASSIFIER_PRIMARY_THRESHOLD = 0.55
    CLASSIFIER_SUPPORT_THRESHOLD = 0.35
    RULE_GATING_AI_CONFIDENCE_THRESHOLD = 0.8
    CLARIFICATION_CONFIDENCE_THRESHOLD = 0.72
    CLARIFICATION_MARGIN_THRESHOLD = 0.12
    CLARIFICATION_OVERRIDE_MARGIN = 0.05
    CLARIFICATION_OVERRIDE_GAIN_THRESHOLD = 0.12
    CLARIFICATION_LEADER_MARGIN = 0.05
    SERIOUS_CLARIFICATION_CONFIDENCE_THRESHOLD = 0.88
    SERIOUS_CLARIFICATION_MARGIN_THRESHOLD = 0.18
    SERIOUS_RESPIRATORY_CONFIDENCE_THRESHOLD = 0.90
    SERIOUS_RESPIRATORY_MARGIN_THRESHOLD = 0.20
    SERIOUS_MIN_SIGNAL_COUNT = 3
    CLASSIFIER_OVERRIDE_MARGIN = 0.08
    PRE_DIAGNOSIS_SIGNAL_WEIGHT = 0.35
    PRE_DIAGNOSIS_RULE_ALIGNMENT_BONUS = 0.06
    PRE_DIAGNOSIS_GENERIC_PATTERN_PENALTY = 0.04
    MAX_CLARIFICATION_QUESTIONS = 3
    SERIOUS_DIAGNOSIS_KEYWORDS = (
        "pulmonary embolism",
        "spontaneous pneumothorax",
        "acute pulmonary edema",
        "possible nstemi",
        "possible stemi",
        "unstable angina",
        "myocarditis",
        "pericarditis",
        "tuberculosis",
        "pneumonia",
    )
    SERIOUS_RESPIRATORY_KEYWORDS = (
        "pulmonary embolism",
        "spontaneous pneumothorax",
        "acute pulmonary edema",
        "pneumonia",
        "tuberculosis",
    )
    GENERIC_RULE_PATTERN_DIRECT_MAP = {
        "possible gastroesophageal reflux pattern": "GERD",
        "possible anemia related symptom pattern": "Anemia",
        "possible upper respiratory tract infection pattern": "URTI",
    }
    GENERIC_RULE_PATTERN_FAMILY_KEYWORDS = {
        "possible lower respiratory infection pattern": (
            "pneumonia",
            "bronchitis",
            "bronchospasm",
            "acute asthma exacerbation",
            "asthma",
            "urti",
            "viral pharyngitis",
            "influenza",
            "whooping cough",
            "bronchiolitis",
            "bronchiectasis",
            "tuberculosis",
            "acute copd exacerbation",
        ),
        "possible acute viral illness pattern": (
            "influenza",
            "viral pharyngitis",
            "urti",
        ),
        "possible upper respiratory tract infection pattern": (
            "urti",
            "viral pharyngitis",
            "acute laryngitis",
            "influenza",
            "allergic sinusitis",
        ),
        "possible cardiopulmonary red flag symptom pattern": (
            "pulmonary embolism",
            "stable angina",
            "unstable angina",
            "possible nstemi",
            "possible stemi",
            "pericarditis",
            "myocarditis",
            "acute pulmonary edema",
            "spontaneous pneumothorax",
            "atrial fibrillation",
            "psvt",
            "panic attack",
        ),
    }
    CONFIDENCE_SCORES = {
        "very high": 0.95,
        "high": 0.85,
        "moderate": 0.65,
        "medium": 0.65,
        "low": 0.45,
        "very low": 0.25,
    }
    NON_DIAGNOSIS_TERMS = {
        "fatigue",
        "thirst",
        "fever",
        "cough",
        "pain",
        "headache",
        "nausea",
        "vomiting",
        "diarrhea",
        "dizziness",
        "weakness",
        "palpitations",
        "shortness of breath",
        "dyspnea",
        "chest pain",
        "abdominal pain",
        "sore throat",
    }
    FOLLOW_UP_QUESTION_BANK = (
        {
            "keywords": ("gerd", "reflux", "gastroesophageal"),
            "question": "Do your symptoms get worse after meals or when lying down, with a sour or acidic taste in the mouth?",
            "question_ar": "هل تزيد الأعراض بعد الأكل أو عند الاستلقاء مع طعم حامضي أو حرقان بالفم؟",
            "signals": ("after meals", "lying down", "sour taste", "acid", "heartburn"),
            "type": "yes_no",
        },
        {
            "keywords": ("larygospasm",),
            "question": "Do you get sudden episodes of difficulty breathing or a high-pitched sound when breathing in?",
            "question_ar": "هل تحدث نوبات مفاجئة من صعوبة التنفس أو صوت صفير/حدة عند الشهيق؟",
            "signals": ("high-pitched", "breathing in", "stridor", "sudden episode"),
            "type": "yes_no",
        },
        {
            "keywords": ("urti", "viral pharyngitis", "allergic sinusitis", "influenza", "bronchitis", "pneumonia"),
            "question": "Do you also have fever, cough, sore throat, or nasal congestion?",
            "question_ar": "هل لديك أيضاً حمى أو كحة أو ألم بالحلق أو احتقان بالأنف؟",
            "signals": ("fever", "cough", "sore throat", "nasal congestion", "runny nose"),
            "type": "multi_select",
        },
        {
            "keywords": ("bronchospasm", "acute asthma exacerbation", "asthma"),
            "question": "Are symptoms mainly wheeze/chest tightness without fever or productive sputum, and do bronchodilators help?",
            "question_ar": "هل الأعراض أساساً صفير/ضيق صدر بدون حمى أو بلغم صديدي، وهل تتحسن مع موسعات الشعب؟",
            "signals": ("wheezing", "chest tightness", "without fever", "no productive cough", "bronchodilator"),
            "type": "multi_select",
        },
        {
            "keywords": ("pericarditis", "unstable angina", "stable angina", "pulmonary embolism", "atrial fibrillation", "myocarditis"),
            "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
            "question_ar": "هل ألم الصدر مرتبط بالمجهود أو التنفس العميق أو خفقان/عدم انتظام ضربات القلب؟",
            "signals": ("exertion", "deep breathing", "irregular heartbeat", "palpitations", "pleuritic"),
            "type": "multi_select",
        },
        {
            "keywords": ("pulmonary embolism",),
            "question": "Did the shortness of breath start suddenly, or was there recent immobility, leg swelling, or chest pain that worsens with breathing?",
            "question_ar": "هل بدأ ضيق التنفس فجأة؟ وهل كان هناك قلة حركة مؤخراً أو تورم بالساق أو ألم صدر يزيد مع التنفس؟",
            "signals": ("suddenly", "immobility", "leg swelling", "worsens with breathing"),
            "type": "multi_select",
        },
        {
            "keywords": ("diabetes", "hyperglycemia", "prediabetes"),
            "question": "Have you noticed increased thirst, frequent urination, weight loss, or blurred vision?",
            "question_ar": "هل لاحظت زيادة في العطش أو كثرة التبول أو فقدان وزن أو زغللة في النظر؟",
            "signals": ("thirst", "frequent urination", "weight loss", "blurred vision"),
            "type": "multi_select",
        },
        {
            "keywords": ("myasthenia gravis", "guillain-barr", "acute dystonic"),
            "question": "Do you have drooping eyelids, double vision, difficulty speaking or swallowing, or worsening weakness over the day?",
            "question_ar": "هل لديك تدلي بالجفن أو ازدواج بالرؤية أو صعوبة بالكلام أو البلع أو ضعف يزداد خلال اليوم؟",
            "signals": ("drooping eyelids", "double vision", "difficulty speaking", "difficulty swallowing", "worsening weakness"),
            "type": "multi_select",
        },
        {
            "keywords": ("anemia",),
            "question": "Are you also having dizziness, shortness of breath on exertion, paleness, or unusual fatigue?",
            "question_ar": "هل لديك أيضاً دوخة أو ضيق تنفس مع المجهود أو شحوب أو تعب غير معتاد؟",
            "signals": ("dizziness", "shortness of breath", "pale", "fatigue"),
            "type": "multi_select",
        },
    )
    CLARIFICATION_PAIR_BANK = (
        {
            "left_keywords": ("atrial fibrillation",),
            "right_keywords": ("psvt",),
            "counterpart_gap": 0.10,
            "question": "Are the palpitations irregular and uneven, or mostly sudden fast episodes that start and stop abruptly?",
            "question_ar": "هل الخفقان غير منتظم ومتفاوت، أم نوبات سريعة مفاجئة تبدأ وتنتهي بشكل مفاجئ؟",
            "signals": ("irregular", "sudden", "abrupt"),
            "type": "multi_select",
        },
        {
            "left_keywords": ("pneumonia",),
            "right_keywords": ("bronchospasm", "acute asthma exacerbation", "asthma"),
            "question": "Do you have clear infection signs (fever with productive cough), or mostly wheeze/chest tightness without infection features?",
            "question_ar": "هل لديك علامات عدوى واضحة (حمى مع كحة ببلغم)، أم صفير/ضيق صدر بدون علامات عدوى؟",
            "signals": ("fever", "productive cough", "wheezing", "chest tightness"),
            "type": "multi_select",
        },
        {
            "left_keywords": ("pneumonia",),
            "right_keywords": ("bronchitis",),
            "question": "Do you have pleuritic chest pain or higher-fever infection signs (favoring pneumonia), or mostly lingering cough after a recent cold (favoring bronchitis)?",
            "question_ar": "هل لديك ألم صدر يزيد مع التنفس أو علامات عدوى أشد (ترجّح الالتهاب الرئوي)، أم كحة ممتدة بعد نزلة برد (ترجّح التهاب الشعب الهوائية)؟",
            "signals": ("pleuritic", "productive cough", "fever", "recent cold"),
            "type": "multi_select",
        },
        {
            "left_keywords": ("unstable angina",),
            "right_keywords": ("stable angina",),
            "question": "Is the chest pain appearing at rest or worsening recently, versus mainly with exertion and improving with rest?",
            "question_ar": "هل ألم الصدر يظهر أثناء الراحة أو يزداد مؤخراً، أم يحدث أساساً مع المجهود ويتحسن بالراحة؟",
            "signals": ("at rest", "worsening", "exertion", "improves with rest"),
            "type": "multi_select",
        },
        {
            "left_keywords": ("pulmonary embolism",),
            "right_keywords": ("bronchitis",),
            "question": "Was the breathing problem sudden with pleuritic pain or leg swelling, or did it follow a gradual upper respiratory infection?",
            "question_ar": "هل بدأت مشكلة التنفس فجأة مع ألم يزيد بالتنفس أو تورم ساق، أم جاءت تدريجياً بعد عدوى تنفسية علوية؟",
            "signals": ("sudden", "pleuritic", "leg swelling", "gradual", "upper respiratory"),
            "type": "multi_select",
        },
        {
            "left_keywords": ("myasthenia gravis",),
            "right_keywords": ("guillain barr", "guillain barre"),
            "question": "Are symptoms mainly fluctuating eye/bulbar weakness (ptosis, speech/swallow fatigue), or more ascending limb weakness with tingling?",
            "question_ar": "هل الأعراض أساساً ضعفاً متذبذباً بالعين/البلع والكلام، أم ضعفاً صاعداً بالأطراف مع تنميل؟",
            "signals": ("ptosis", "fatigable", "ascending", "tingling"),
            "type": "multi_select",
        },
    )
    DIAGNOSTIC_SIGNAL_BANK = {
        "gerd": {
            "positive": ("after meals", "lying down", "sour taste", "reflux", "heartburn", "burning"),
            "negative": ("high-pitched", "stridor"),
        },
        "larygospasm": {
            "positive": ("high-pitched", "stridor", "breathing in", "sudden episode"),
            "negative": ("after meals", "sour taste", "heartburn"),
        },
        "viral pharyngitis": {
            "positive": ("sore throat", "mild fever", "nasal congestion", "cold symptoms", "hoarseness"),
            "negative": ("irregular palpitations", "chest pain"),
        },
        "urti": {
            "positive": ("sore throat", "nasal congestion", "runny nose", "hoarseness", "recent cold"),
            "negative": ("pleuritic", "leg swelling", "immobility"),
        },
        "acute laryngitis": {
            "positive": ("hoarseness", "painful voice use", "voice", "recent cold", "upper respiratory"),
            "negative": ("primary chest symptoms",),
        },
        "pulmonary embolism": {
            "positive": ("sudden", "pleuritic", "leg swelling", "immobility", "worse when i breathe"),
            "negative": ("gradual", "mild infection"),
        },
        "spontaneous pneumothorax": {
            "positive": ("sudden", "one sided", "sharp", "shortness of breath", "pleuritic"),
            "negative": ("productive cough", "fever", "gradual"),
        },
        "myocarditis": {
            "positive": ("viral", "fatigue", "worse when lying down", "myalgia"),
            "negative": ("sudden", "pleuritic", "leg swelling", "immobility"),
        },
        "pericarditis": {
            "positive": ("worse when lying down", "better sitting up", "pleuritic"),
            "negative": ("leg swelling", "immobility"),
        },
        "bronchitis": {
            "positive": ("recent upper respiratory", "persistent cough", "recent cold", "mild shortness of breath"),
            "negative": ("major weight loss", "irregular heartbeat", "pleuritic", "productive cough"),
        },
        "pulmonary neoplasm": {
            "positive": ("chronic cough", "weight loss", "progressive", "worsening breathing"),
            "negative": ("short infection",),
        },
        "pancreatic neoplasm": {
            "positive": ("epigastric", "poor appetite", "weight loss", "progressive"),
            "negative": ("brief stomach upset",),
        },
        "sarcoidosis": {
            "positive": ("dry cough", "gradual", "persistent", "not productive"),
            "negative": ("productive cough", "acute infection"),
        },
        "pneumonia": {
            "positive": ("fever", "productive cough", "pleuritic", "infection"),
            "negative": ("isolated wheezing",),
        },
        "bronchospasm acute asthma exacerbation": {
            "positive": ("wheezing", "chest tightness", "airways are tightening", "bronchospasm", "asthma"),
            "negative": ("high fever", "productive cough", "infection"),
        },
        "unstable angina": {
            "positive": ("at rest", "worsening", "diaphoresis", "sweating", "nausea"),
            "negative": ("improves with rest", "exertion only"),
        },
        "stable angina": {
            "positive": ("exertion", "improves with rest", "relief with rest"),
            "negative": ("at rest", "worsening"),
        },
        "spontaneous pneumothorax": {
            "positive": ("sudden", "one-sided", "unilateral", "acute shortness of breath"),
            "negative": ("productive cough", "fever"),
        },
        "scombroid food poisoning": {
            "positive": ("after eating fish", "soon after eating fish", "flushing", "food-triggered"),
            "negative": (),
        },
        "atrial fibrillation": {
            "positive": ("irregular heartbeat", "irregular rhythm", "atrial fibrillation", "palpitations"),
            "negative": ("sudden episodes", "comes in attacks", "settle"),
        },
        "psvt": {
            "positive": ("sudden episodes", "rapid heartbeat", "attacks", "psvt"),
            "negative": ("irregular heartbeat", "irregular rhythm", "atrial fibrillation"),
        },
        "myasthenia gravis": {
            "positive": ("ptosis", "drooping eyelids", "difficulty speaking", "fatigable", "worsens with exertion"),
            "negative": (),
        },
        "guillain barr": {
            "positive": ("ascending weakness", "tingling", "reflexes"),
            "negative": ("ptosis", "fatigable", "difficulty speaking"),
        },
    }
    RESPIRATORY_LABEL_KEYWORDS = (
        "possible lower respiratory infection pattern",
        "pneumonia",
        "bronchitis",
        "bronchospasm",
        "acute asthma exacerbation",
        "asthma",
        "sarcoidosis",
        "urti",
        "viral pharyngitis",
        "influenza",
        "larygospasm",
        "pulmonary",
    )
    ANSWER_SIGNAL_ALIASES = {
        "fever": ("fever", "temperature", "high temperature", "حمى", "حرارة", "سخونية", "سخونه"),
        "productive cough": ("productive cough", "sputum", "phlegm", "بلغم", "كحة ببلغم", "كحه ببلغم"),
        "cough": ("cough", "coughing", "كحة", "كحه", "سعال"),
        "wheezing": ("wheezing", "wheeze", "صفير", "ازيز", "أزيز"),
        "chest tightness": ("chest tightness", "tight chest", "ضيق صدر", "كتمة", "كتمه"),
        "shortness of breath": (
            "shortness of breath",
            "short of breath",
            "dyspnea",
            "breathlessness",
            "ضيق تنفس",
            "نهجان",
            "صعوبة التنفس",
            "صعوبة في التنفس",
        ),
        "palpitations": ("palpitations", "rapid heartbeat", "خفقان", "ضربات قلب سريعة", "رفرفة"),
        "irregular heartbeat": (
            "irregular heartbeat",
            "irregular rhythm",
            "arrhythmia",
            "عدم انتظام ضربات القلب",
            "نبض غير منتظم",
        ),
        "exertion": ("exertion", "exercise", "effort", "مجهود", "مع المجهود"),
        "improves with rest": (
            "improves with rest",
            "better with rest",
            "relief with rest",
            "يتحسن مع الراحة",
            "يرتاح مع الراحة",
            "يخف مع الراحة",
        ),
        "at rest": ("at rest", "rest pain", "في الراحة", "اثناء الراحة", "أثناء الراحة"),
        "sudden": ("sudden", "abrupt", "suddenly", "فجأة", "مفاجئ"),
        "gradual": ("gradual", "progressive", "تدريجي", "بالتدريج"),
        "pleuritic": (
            "pleuritic",
            "worse when breathing",
            "worsens with breathing",
            "يزيد مع التنفس",
            "ألم مع التنفس",
        ),
        "leg swelling": ("leg swelling", "swollen leg", "تورم الساق", "ورم الساق"),
        "immobility": ("immobility", "prolonged sitting", "قلة الحركة", "عدم الحركة"),
        "one sided": ("one sided", "one-sided", "unilateral", "جهة واحدة", "جانب واحد"),
        "dry cough": ("dry cough", "كحة جافة", "كحه جافة", "سعال جاف"),
    }

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        clinicalbert_model_dir: Optional[Path | str] = None,
        allow_unsafe_pickle_metadata: bool = False,
        llm_provider: str = "gemini",
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openrouter_site_url: Optional[str] = None,
        openrouter_app_name: str = "GP Medical Analysis",
        openrouter_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-2.5-flash-lite",
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
        self._response_synthesizer: Optional[DiagnosisResponseSynthesizer] = None
        self._classifier_translate_arabic = classifier_translate_arabic
        self._rag_top_k = rag_top_k

        if use_rag:
            if not faiss_index_dir:
                raise ValueError("RAG requires faiss_index_dir")
            self._rag_assistant = MedicalRAGAssistant(
                embedder=ClinicalBERTEmbedder(model_dir=clinicalbert_model_dir),
                searcher=MedicalCaseSearcher(
                    Path(faiss_index_dir),
                    allow_unsafe_pickle=allow_unsafe_pickle_metadata,
                ),
                translate_arabic=rag_translate_arabic,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model_name=llm_model_name,
                openrouter_base_url=openrouter_base_url,
                openrouter_site_url=openrouter_site_url,
                openrouter_app_name=openrouter_app_name,
                openrouter_api_key=openrouter_api_key,
                gemini_api_key=gemini_api_key,
                gemini_model_name=gemini_model_name,
            )

        if use_finetuned_classifier:
            if not finetuned_model_dir:
                raise ValueError("Fine-tuned classifier requires finetuned_model_dir")
            self._classifier = FineTunedDiagnosisClassifier(
                model_dir=finetuned_model_dir,
                max_length=classifier_max_length,
            )
            _, translator_provider, _ = create_model_provider(
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
            if classifier_translate_arabic and translator_provider:
                self._classifier_translator = ArabicToEnglishTranslator(
                    translator_provider
                )
            elif classifier_translate_arabic:
                logger.warning(
                    "classifier_translate_arabic=True but no LLM API key is configured; classifier will use raw input text."
                )

        _, synthesis_provider, _ = create_model_provider(
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
        if synthesis_provider:
            self._response_synthesizer = DiagnosisResponseSynthesizer(
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model_name=llm_model_name,
                openrouter_base_url=openrouter_base_url,
                openrouter_site_url=openrouter_site_url,
                openrouter_app_name=openrouter_app_name,
                gemini_api_key=gemini_api_key,
                gemini_model_name=gemini_model_name,
                openrouter_api_key=openrouter_api_key,
            )

    @staticmethod
    def _build_safety(
        findings: list[Dict[str, Any]],
        *,
        response_language: str = "en",
    ) -> Dict[str, Any]:
        language = normalize_language(response_language)
        arabic_mode = language == "ar"
        severity_order = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "info": 0}
        severity_labels_ar = {
            "critical": "حرجة",
            "high": "مرتفعة",
            "moderate": "متوسطة",
            "low": "منخفضة",
            "info": "معلوماتية",
        }
        highest_severity = "info"
        reasons: list[str] = []

        for finding in findings:
            severity = str(finding.get("severity", "info")).lower()
            if severity_order.get(severity, 0) > severity_order.get(highest_severity, 0):
                highest_severity = severity
            if severity in {"critical", "high"}:
                condition = str(finding.get("condition", "Unknown finding"))
                if arabic_mode:
                    reason = f"تم تصنيف {condition} بدرجة خطورة {severity_labels_ar.get(severity, severity)}."
                else:
                    reason = f"{condition} marked as {severity} severity."
                reasons.append(
                    reason
                )

        clinician_review_required = bool(findings)
        emergency_attention_recommended = highest_severity == "critical"

        if not findings:
            reasons.append(
                "لم يتم اكتشاف نتائج غير طبيعية بواسطة قواعد التحليل."
                if arabic_mode
                else "No abnormal findings were detected by the rule engine."
            )
        elif clinician_review_required and not reasons:
            reasons.append(
                "يوصى بمراجعة سريرية لأي نتيجة غير طبيعية."
                if arabic_mode
                else "Clinical review is recommended for any abnormal finding."
            )

        return {
            "clinician_review_required": clinician_review_required,
            "emergency_attention_recommended": emergency_attention_recommended,
            "highest_rule_severity": highest_severity,
            "critical_findings_count": sum(
                1 for finding in findings if str(finding.get("severity", "")).lower() == "critical"
            ),
            "reasons": reasons,
        }

    @staticmethod
    def _normalize_confidence(confidence: Any) -> float:
        if isinstance(confidence, (int, float)):
            return max(0.0, min(float(confidence), 1.0))
        normalized = str(confidence or "").strip().lower()
        return DiagnosisEngine.CONFIDENCE_SCORES.get(normalized, 0.0)

    @staticmethod
    def _labels_overlap(left: str, right: str) -> bool:
        left_normalized = str(left or "").strip().lower()
        right_normalized = str(right or "").strip().lower()
        if not left_normalized or not right_normalized:
            return False
        return (
            left_normalized in right_normalized
            or right_normalized in left_normalized
        )

    @classmethod
    def _merge_candidate(
        cls,
        merged: Dict[str, Dict[str, Any]],
        *,
        label: str,
        confidence: float,
        source: str,
        reasoning: str,
        evidence: list[str],
    ) -> None:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return
        existing = merged.get(normalized_label)
        payload = {
            "label": label,
            "confidence": confidence,
            "sources": [source],
            "reasoning": reasoning,
            "evidence": list(dict.fromkeys(evidence)),
        }
        if existing is None:
            merged[normalized_label] = payload
            return
        if confidence > existing["confidence"]:
            existing["label"] = label
            existing["confidence"] = confidence
            existing["reasoning"] = reasoning
        existing["sources"] = list(dict.fromkeys(existing["sources"] + [source]))
        existing["evidence"] = list(dict.fromkeys(existing["evidence"] + evidence))

    @classmethod
    def _collect_diagnostic_candidates(
        cls,
        *,
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        report_text: str = "",
        rag_out: Optional[Dict[str, Any]] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        rule_conditions = [str(item.get("condition", "")).strip() for item in findings if item.get("condition")]
        rag_structured = (rag_out or {}).get("structured_diagnosis") or {}
        rag_findings = rag_structured.get("findings") or []
        rag_summary = str(rag_structured.get("assessment_summary", "")).strip()
        rag_confidence = (rag_out or {}).get("rag_confidence") or {}
        rag_usable_for_fusion = bool(rag_confidence.get("usable_for_fusion", bool(rag_out)))

        if classifier_prediction:
            for item in classifier_prediction.get("top_predictions", []) or []:
                label = str(item.get("label", "")).strip()
                confidence = cls._normalize_confidence(item.get("confidence"))
                if label and not cls._is_symptom_like_label(label, patient_symptoms):
                    cls._merge_candidate(
                        merged,
                        label=label,
                        confidence=confidence,
                        source="classifier",
                        reasoning=f"Fine-tuned classifier prediction with confidence {confidence:.2f}.",
                        evidence=[f"Classifier candidate: {label}"],
                    )

        if rag_usable_for_fusion:
            for item in rag_findings:
                label = str(item.get("condition", "")).strip()
                confidence = cls._normalize_confidence(item.get("confidence"))
                evidence = str(item.get("evidence", "")).strip() or "RAG finding extracted from similar cases."
                if label and not cls._is_symptom_like_label(label, patient_symptoms):
                    cls._merge_candidate(
                        merged,
                        label=label,
                        confidence=confidence,
                        source="rag",
                        reasoning=rag_summary or "RAG structured assessment based on retrieved clinical cases.",
                        evidence=[evidence],
                    )

        seen_rag_labels: set[str] = set()
        if rag_usable_for_fusion:
            for case in (rag_out or {}).get("retrieved_cases", []) or []:
                label = str(case.get("pathology", "")).strip()
                if not label or cls._is_symptom_like_label(label, patient_symptoms):
                    continue
                normalized_label = cls._normalize_label(label)
                if normalized_label in seen_rag_labels:
                    continue
                seen_rag_labels.add(normalized_label)
                retrieval_score = float(case.get("rerank_score", case.get("similarity", 0.0)) or 0.0)
                similarity = max(0.3, min(retrieval_score, 0.8))
                cls._merge_candidate(
                    merged,
                    label=label,
                    confidence=similarity,
                    source="rag_retrieval",
                    reasoning=(
                        "Nearest-neighbor retrieval from similar indexed medical cases "
                        f"(RAG confidence: {rag_confidence.get('level', 'unknown')})."
                    ),
                    evidence=[f"Top retrieved case pathology: {label}"],
                )

        for finding in findings:
            label = str(finding.get("condition", "")).strip()
            if not label:
                continue
            cls._merge_candidate(
                merged,
                label=label,
                confidence=cls._normalize_confidence(finding.get("confidence")),
                source=str(finding.get("source", "rules")),
                reasoning="Deterministic rule-based clinical signal.",
                evidence=[str(finding.get("evidence", "")).strip() or f"Rule finding: {label}"],
            )

        expanded = cls._expand_base_diagnostic_candidates(
            list(merged.values()),
            report_text=report_text,
        )
        reranked = cls._rerank_base_candidates(
            candidates=expanded,
            findings=findings,
            patient_symptoms=patient_symptoms,
            report_text=report_text,
        )

        for candidate in reranked:
            if rule_conditions and "rule_alignment" not in candidate:
                candidate["rule_alignment"] = cls._diagnosis_aligns_with_rules(
                    str(candidate.get("label", "")),
                    rule_conditions,
                )
            elif "rule_alignment" not in candidate:
                candidate["rule_alignment"] = False

        return reranked

    @classmethod
    def _expand_base_diagnostic_candidates(
        cls,
        candidates: list[Dict[str, Any]],
        report_text: str = "",
    ) -> list[Dict[str, Any]]:
        if not candidates:
            return candidates

        candidate_map: Dict[str, Dict[str, Any]] = {}
        for item in candidates[:10]:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            candidate_map[cls._normalize_label(label)] = dict(item)

        normalized_context = cls._normalize_label(report_text)
        top_confidence = max(
            (float(item.get("confidence", 0.0)) for item in candidate_map.values()),
            default=0.0,
        )
        has_chest_discomfort = any(
            term in normalized_context
            for term in ("chest pain", "chest discomfort", "chest pressure")
        )
        has_short_breath = any(
            term in normalized_context
            for term in ("shortness of breath", "short of breath", "dyspnea")
        )
        pe_risk_markers = any(
            term in normalized_context
            for term in (
                "leg swelling",
                "calf swelling",
                "immobility",
                "recent surgery",
                "after surgery",
                "long flight",
            )
        )
        unstable_angina_context = (
            has_chest_discomfort
            and any(
                term in normalized_context
                for term in ("at rest", "rest pain", "worsening", "getting worse", "sweating", "diaphoresis")
            )
        )
        stable_angina_context = (
            has_chest_discomfort
            and any(term in normalized_context for term in ("exertion", "exercise", "effort"))
            and any(
                term in normalized_context
                for term in ("improves with rest", "better with rest", "relief with rest")
            )
        )
        spontaneous_pneumothorax_context = (
            any(term in normalized_context for term in ("sudden", "suddenly"))
            and any(term in normalized_context for term in ("one side", "one sided", "unilateral"))
            and has_short_breath
            and not pe_risk_markers
        )
        pulmonary_neoplasm_context = (
            "cough" in normalized_context
            and "weight loss" in normalized_context
            and any(term in normalized_context for term in ("chronic", "persistent", "progressive", "worsening"))
        )
        pancreatic_neoplasm_context = (
            any(term in normalized_context for term in ("epigastric", "abdominal pain", "stomach pain"))
            and "weight loss" in normalized_context
            and any(term in normalized_context for term in ("poor appetite", "loss of appetite", "reduced appetite"))
        )
        viral_pharyngitis_context = (
            "sore throat" in normalized_context
            and "fever" in normalized_context
            and any(term in normalized_context for term in ("hoarseness", "cold", "nasal congestion", "runny nose"))
        )
        acute_laryngitis_context = (
            "hoarseness" in normalized_context
            and any(term in normalized_context for term in ("voice", "recent cold", "sore throat"))
        )
        upper_respiratory_context = any(
            term in normalized_context
            for term in ("sore throat", "nasal congestion", "runny nose", "hoarseness", "recent cold", "upper respiratory")
        )
        respiratory_infection_context = any(
            term in normalized_context
            for term in ("fever", "productive cough", "pleuritic", "infection", "chills", "sputum")
        )
        wheeze_dominant_context = (
            any(term in normalized_context for term in ("wheez", "chest tightness"))
            and not respiratory_infection_context
        )

        if unstable_angina_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Unstable angina",
                confidence=max(0.62, top_confidence - 0.02),
                source="cardiopulmonary_pattern_expansion",
            )
        if stable_angina_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Stable angina",
                confidence=max(0.58, top_confidence - 0.03),
                source="cardiopulmonary_pattern_expansion",
            )
        if spontaneous_pneumothorax_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Spontaneous pneumothorax",
                confidence=max(0.62, top_confidence - 0.02),
                source="cardiopulmonary_pattern_expansion",
            )
        if pulmonary_neoplasm_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Pulmonary neoplasm",
                confidence=max(0.63, top_confidence - 0.01),
                source="clinical_context_expansion",
            )
        if pancreatic_neoplasm_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Pancreatic neoplasm",
                confidence=max(0.60, top_confidence - 0.02),
                source="clinical_context_expansion",
            )
        if viral_pharyngitis_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Viral pharyngitis",
                confidence=max(0.58, top_confidence - 0.03),
                source="clinical_context_expansion",
            )
        if acute_laryngitis_context:
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Acute laryngitis",
                confidence=max(0.56, top_confidence - 0.04),
                source="clinical_context_expansion",
            )

        lower_respiratory_pattern = next(
            (
                item for item in candidate_map.values()
                if cls._normalize_label(str(item.get("label", ""))) == "possible lower respiratory infection pattern"
            ),
            None,
        )
        if lower_respiratory_pattern is not None:
            pattern_conf = float(lower_respiratory_pattern.get("confidence", 0.35))
            bronchitis_confidence = max(0.40, pattern_conf + (0.07 if respiratory_infection_context else 0.03))
            pneumonia_confidence = max(0.34, pattern_conf + (0.06 if respiratory_infection_context else -0.01))
            bronchospasm_confidence = max(0.39, pattern_conf + (0.07 if wheeze_dominant_context else 0.04))

            cls._inject_or_promote_candidate(
                candidate_map,
                label="Bronchitis",
                confidence=bronchitis_confidence,
                source="respiratory_pattern_expansion",
            )
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Pneumonia",
                confidence=pneumonia_confidence,
                source="respiratory_pattern_expansion",
            )
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Bronchospasm / acute asthma exacerbation",
                confidence=bronchospasm_confidence,
                source="respiratory_pattern_expansion",
            )
            if upper_respiratory_context:
                cls._inject_or_promote_candidate(
                    candidate_map,
                    label="URTI",
                    confidence=max(0.41, pattern_conf + 0.04),
                    source="respiratory_pattern_expansion",
                )
                cls._inject_or_promote_candidate(
                    candidate_map,
                    label="Viral pharyngitis",
                    confidence=max(0.40, pattern_conf + 0.03),
                    source="respiratory_pattern_expansion",
                )
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Sarcoidosis",
                confidence=max(0.33, pattern_conf - 0.02),
                source="respiratory_pattern_expansion",
            )

            expanded_probe = sorted(candidate_map.values(), key=lambda item: item.get("confidence", 0.0), reverse=True)
            top_label = str(expanded_probe[0].get("label", "")) if expanded_probe else ""
            if expanded_probe and not cls._is_respiratory_label(top_label):
                target_floor = max(0.46, float(expanded_probe[0].get("confidence", 0.0)) - 0.03)
                for item in expanded_probe:
                    label = str(item.get("label", ""))
                    if not cls._is_respiratory_label(label):
                        continue
                    cls._inject_or_promote_candidate(
                        candidate_map,
                        label=label,
                        confidence=max(float(item.get("confidence", 0.0)), target_floor),
                        source="respiratory_context_rebalance",
                    )

        cardiopulmonary_pattern = next(
            (
                item for item in candidate_map.values()
                if cls._normalize_label(str(item.get("label", ""))) == "possible cardiopulmonary red flag symptom pattern"
            ),
            None,
        )
        if cardiopulmonary_pattern is not None:
            pattern_conf = float(cardiopulmonary_pattern.get("confidence", 0.35))
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Pulmonary embolism",
                confidence=max(0.44, pattern_conf + 0.04),
                source="cardiopulmonary_pattern_expansion",
            )

        return sorted(candidate_map.values(), key=lambda item: item.get("confidence", 0.0), reverse=True)

    @classmethod
    def _pre_diagnosis_context_boost(cls, label: str, context_text: str) -> float:
        normalized_label = cls._normalize_label(label)
        normalized_context = cls._normalize_label(context_text)
        if not normalized_label or not normalized_context:
            return 0.0

        has_infection = any(term in normalized_context for term in ("fever", "productive cough", "infection"))
        has_wheeze = "wheez" in normalized_context or "chest tightness" in normalized_context
        upper_respiratory_context = any(
            term in normalized_context
            for term in ("sore throat", "nasal congestion", "runny nose", "hoarseness", "recent cold", "upper respiratory")
        )
        no_fever = cls._is_negated_signal(normalized_context, "fever") or "without fever" in normalized_context
        no_productive_cough = (
            cls._is_negated_signal(normalized_context, "productive cough")
            or "without productive" in normalized_context
            or "no productive" in normalized_context
        )
        pe_context = (
            (
                "sudden" in normalized_context
                and (
                    "pleuritic" in normalized_context
                    or "worse when i breathe" in normalized_context
                    or "worsens with breathing" in normalized_context
                )
            )
            or "leg swelling" in normalized_context
            or "immobility" in normalized_context
        )
        pe_risk_markers = any(
            term in normalized_context
            for term in (
                "leg swelling",
                "calf swelling",
                "immobility",
                "recent surgery",
                "after surgery",
                "long flight",
            )
        )
        af_context = "irregular" in normalized_context and (
            "palpit" in normalized_context or "heartbeat" in normalized_context
        )
        psvt_context = any(
            phrase in normalized_context
            for phrase in ("sudden fast", "abrupt", "start and stop", "starts and stops")
        )
        unstable_angina_context = (
            any(term in normalized_context for term in ("at rest", "rest pain", "worsening", "getting worse"))
            and any(term in normalized_context for term in ("chest pain", "chest pressure", "chest discomfort"))
        )
        stable_angina_context = (
            any(term in normalized_context for term in ("exertion", "exercise", "effort"))
            and any(term in normalized_context for term in ("improves with rest", "better with rest", "relief with rest"))
            and any(term in normalized_context for term in ("chest pain", "chest pressure", "chest discomfort"))
        )
        spontaneous_pneumothorax_context = (
            any(term in normalized_context for term in ("sudden", "suddenly"))
            and any(term in normalized_context for term in ("one side", "one sided", "unilateral"))
            and any(term in normalized_context for term in ("shortness of breath", "short of breath", "dyspnea"))
            and not pe_risk_markers
        )
        pulmonary_neoplasm_context = (
            "cough" in normalized_context
            and "weight loss" in normalized_context
            and any(term in normalized_context for term in ("chronic", "persistent", "progressive", "worsening"))
        )
        pancreatic_neoplasm_context = (
            any(term in normalized_context for term in ("epigastric", "abdominal pain", "stomach pain"))
            and "weight loss" in normalized_context
            and any(term in normalized_context for term in ("poor appetite", "loss of appetite", "reduced appetite"))
        )
        viral_pharyngitis_context = (
            "sore throat" in normalized_context
            and "fever" in normalized_context
            and any(term in normalized_context for term in ("hoarseness", "cold", "nasal congestion", "runny nose"))
        )
        acute_laryngitis_context = (
            "hoarseness" in normalized_context
            and any(term in normalized_context for term in ("voice", "recent cold", "sore throat"))
        )
        chronic_dry_cough = (
            "dry cough" in normalized_context
            and ("gradual" in normalized_context or "persistent" in normalized_context)
        )

        boost = 0.0
        if "pulmonary embolism" in normalized_label and pe_context:
            boost += 0.18
        if any(term in normalized_label for term in ("myocarditis", "pericarditis")) and pe_context:
            boost -= 0.10

        if "pneumonia" in normalized_label:
            if has_infection:
                boost += 0.12
            if has_wheeze and no_fever and no_productive_cough:
                boost -= 0.09
            if upper_respiratory_context and not has_infection:
                boost -= 0.10

        if "bronchospasm" in normalized_label or "asthma" in normalized_label:
            if has_wheeze:
                boost += 0.10
            if no_fever or no_productive_cough:
                boost += 0.07

        if "bronchitis" in normalized_label:
            if "pleuritic" in normalized_context and has_infection:
                boost -= 0.08
            if has_wheeze and (no_fever or no_productive_cough):
                boost -= 0.08

        if "urti" in normalized_label and upper_respiratory_context:
            boost += 0.16

        if "viral pharyngitis" in normalized_label and upper_respiratory_context:
            boost += 0.08

        if "atrial fibrillation" in normalized_label and af_context:
            boost += 0.12
        if "psvt" in normalized_label and psvt_context:
            boost += 0.10

        if "sarcoidosis" in normalized_label:
            if chronic_dry_cough:
                boost += 0.14
            if cls._is_negated_signal(normalized_context, "acute infection"):
                boost += 0.06

        if "myocarditis" in normalized_label and chronic_dry_cough:
            boost -= 0.16
        if "myocarditis" in normalized_label and unstable_angina_context:
            boost -= 0.18
        if "myocarditis" in normalized_label and spontaneous_pneumothorax_context:
            boost -= 0.22

        if "unstable angina" in normalized_label and unstable_angina_context:
            boost += 0.24
        if "stable angina" in normalized_label and stable_angina_context:
            boost += 0.18
        if "spontaneous pneumothorax" in normalized_label and spontaneous_pneumothorax_context:
            boost += 0.26
        if "spontaneous pneumothorax" in normalized_label and pe_risk_markers:
            boost -= 0.14
        if "pulmonary neoplasm" in normalized_label and pulmonary_neoplasm_context:
            boost += 0.22
        if "pancreatic neoplasm" in normalized_label and pancreatic_neoplasm_context:
            boost += 0.22
        if "viral pharyngitis" in normalized_label and viral_pharyngitis_context:
            boost += 0.20
        if "acute laryngitis" in normalized_label and acute_laryngitis_context:
            boost += 0.16

        if "ebola" in normalized_label and viral_pharyngitis_context:
            boost -= 0.32
        if "larygospasm" in normalized_label and pulmonary_neoplasm_context:
            boost -= 0.20
        if ("bronchospasm" in normalized_label or "asthma" in normalized_label) and spontaneous_pneumothorax_context:
            boost -= 0.20

        return boost

    @classmethod
    def _rerank_base_candidates(
        cls,
        *,
        candidates: list[Dict[str, Any]],
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        report_text: str,
    ) -> list[Dict[str, Any]]:
        if not candidates:
            return candidates

        rule_conditions = [str(item.get("condition", "")).strip() for item in findings if item.get("condition")]
        context_text = " ".join(
            part for part in (
                report_text,
                " ".join(patient_symptoms),
            )
            if str(part).strip()
        )
        normalized_context = cls._normalize_label(context_text)
        has_concrete_candidates = any(
            not cls._is_generic_rule_pattern_label(str(item.get("label", "")))
            for item in candidates
        )

        reranked: list[Dict[str, Any]] = []
        for item in candidates:
            candidate = dict(item)
            label = str(candidate.get("label", "")).strip()
            base_confidence = cls._normalize_confidence(candidate.get("confidence"))

            signal_score = cls._signal_match_score(label, normalized_context) if normalized_context else 0.0
            context_boost = cls._pre_diagnosis_context_boost(label, normalized_context) if normalized_context else 0.0
            rule_alignment = cls._diagnosis_aligns_with_rules(label, rule_conditions)
            rule_bonus = (
                cls.PRE_DIAGNOSIS_RULE_ALIGNMENT_BONUS
                if rule_alignment and not cls._is_generic_rule_pattern_label(label)
                else 0.0
            )
            generic_penalty = (
                cls.PRE_DIAGNOSIS_GENERIC_PATTERN_PENALTY
                if has_concrete_candidates and cls._is_generic_rule_pattern_label(label)
                else 0.0
            )

            adjusted_confidence = (
                base_confidence
                + (cls.PRE_DIAGNOSIS_SIGNAL_WEIGHT * signal_score)
                + context_boost
                + rule_bonus
                - generic_penalty
            )
            candidate["confidence"] = max(0.05, min(round(adjusted_confidence, 4), 0.98))
            candidate["rule_alignment"] = rule_alignment
            reranked.append(candidate)

        candidates = sorted(
            reranked,
            key=lambda item: (
                item["confidence"],
                "classifier" in item["sources"] and any(
                    source in {"rag", "rag_retrieval"}
                    for source in item["sources"]
                ),
                "classifier" in item["sources"],
                "rag" in item["sources"] or "rag_retrieval" in item["sources"],
            ),
            reverse=True,
        )
        return candidates

    @classmethod
    def _clarification_reasons(
        cls,
        *,
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        candidates: list[Dict[str, Any]],
        final_diagnosis: Optional[Dict[str, Any]],
    ) -> list[str]:
        reasons: list[str] = []
        if not candidates and not findings:
            return reasons

        lab_rule_findings = [
            item for item in findings
            if str(item.get("source", "")).strip().lower() == "lab_rules"
        ]
        ai_candidates = [
            item for item in candidates
            if any(source in {"classifier", "rag", "rag_retrieval"} for source in item.get("sources", []))
        ]
        if (
            final_diagnosis
            and str(final_diagnosis.get("source", "")).strip().lower() == "rules_fallback"
            and lab_rule_findings
        ):
            return reasons
        if (
            final_diagnosis
            and str(final_diagnosis.get("source", "")).strip().lower() == "rules_fallback"
            and not ai_candidates
            and len(candidates) <= 1
        ):
            return reasons
        if not final_diagnosis:
            reasons.append("No reliable final diagnosis could be selected from the available evidence.")
        else:
            final_confidence = cls._normalize_confidence(final_diagnosis.get("confidence"))
            final_source = str(final_diagnosis.get("source", "")).strip().lower()
            final_label = str(final_diagnosis.get("diagnosis", "")).strip()
            candidate_margin = 0.0
            serious_follow_up_required = False
            canonicalized_from = cls._normalize_label(str(final_diagnosis.get("canonicalized_from", "")))
            if (
                final_source == "rules_fallback"
                and bool(final_diagnosis.get("rule_alignment"))
                and canonicalized_from in cls.GENERIC_RULE_PATTERN_DIRECT_MAP
            ):
                return reasons
            if len(candidates) >= 2:
                top_confidence = cls._normalize_confidence(candidates[0].get("confidence"))
                second_confidence = cls._normalize_confidence(candidates[1].get("confidence"))
                candidate_margin = top_confidence - second_confidence

            serious_follow_up_required = cls._requires_serious_follow_up(
                label=final_label,
                confidence=final_confidence,
                candidate_margin=candidate_margin,
                findings=findings,
                patient_symptoms=patient_symptoms,
                final_diagnosis=final_diagnosis,
            )
            if serious_follow_up_required:
                reasons.append(
                    "Serious or high-risk diagnosis requires targeted follow-up before final confirmation."
                )

            if len(candidates) >= 2:
                if (
                    final_confidence >= 0.70
                    and candidate_margin >= 0.18
                    and bool(final_diagnosis.get("rule_alignment"))
                    and not serious_follow_up_required
                ):
                    return reasons
                if (
                    final_confidence >= 0.90
                    and candidate_margin >= 0.14
                    and final_source in {
                        "classifier",
                        "classifier_rag_consensus",
                        "rag",
                        "rag_retrieval",
                        "cardiopulmonary_pattern_expansion",
                        "respiratory_pattern_expansion",
                    }
                    and not serious_follow_up_required
                ):
                    return reasons
            if (
                final_source == "respiratory_pattern_expansion"
                and final_confidence >= 0.80
                and bool(final_diagnosis.get("rule_alignment"))
                and not serious_follow_up_required
                and cls._label_matches_keywords(
                    final_label,
                    ("bronchitis", "bronchospasm", "acute asthma exacerbation", "asthma"),
                )
            ):
                return reasons
            if final_confidence < cls.CLARIFICATION_CONFIDENCE_THRESHOLD:
                reasons.append("Current diagnosis confidence is below the clarification threshold.")
            if findings and not final_diagnosis.get("rule_alignment"):
                # Avoid low-information follow-ups when AI confidence and margin are already strong.
                if not (final_confidence >= 0.88 and candidate_margin >= 0.12):
                    reasons.append("Rule-based safety signals do not clearly align with the current AI diagnosis.")

        if len(candidates) >= 2:
            top_conf = candidates[0]["confidence"]
            second_conf = candidates[1]["confidence"]
            if (
                cls._normalize_label(candidates[0]["label"]) != cls._normalize_label(candidates[1]["label"])
                and abs(top_conf - second_conf) <= cls.CLARIFICATION_MARGIN_THRESHOLD
            ):
                high_confidence_rule_aligned = (
                    bool(final_diagnosis)
                    and bool((final_diagnosis or {}).get("rule_alignment"))
                    and cls._normalize_confidence((final_diagnosis or {}).get("confidence")) >= 0.82
                )
                if not high_confidence_rule_aligned:
                    reasons.append("Top candidate diseases are close in score and need discrimination.")

        if len(patient_symptoms) < 2 and not findings:
            reasons.append("The first-turn symptom summary is sparse, so more discriminative details are needed.")
        return reasons

    @classmethod
    def _inject_or_promote_candidate(
        cls,
        candidate_map: Dict[str, Dict[str, Any]],
        *,
        label: str,
        confidence: float,
        source: str,
    ) -> None:
        normalized = cls._normalize_label(label)
        if not normalized:
            return
        existing = candidate_map.get(normalized)
        payload = {
            "label": label,
            "confidence": max(0.0, min(confidence, 0.95)),
            "sources": [source],
            "reasoning": f"Clarification candidate expansion: {source}.",
            "evidence": [f"Clarification expansion candidate: {label}"],
            "rule_alignment": False,
        }
        if existing is None:
            candidate_map[normalized] = payload
            return

        existing_confidence = float(existing.get("confidence", 0.0))
        if payload["confidence"] > existing_confidence or (
            str(existing.get("label", "")) == str(existing.get("label", "")).lower()
            and label != label.lower()
        ):
            existing["label"] = label
        existing["confidence"] = max(existing_confidence, payload["confidence"])
        existing["sources"] = list(dict.fromkeys(list(existing.get("sources", [])) + payload["sources"]))
        if payload["evidence"][0] not in existing.get("evidence", []):
            existing["evidence"] = list(existing.get("evidence", [])) + payload["evidence"]

    @classmethod
    def _expand_clarification_candidates(cls, candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if not candidates:
            return candidates

        candidate_map: Dict[str, Dict[str, Any]] = {}
        for item in candidates[:8]:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            candidate_map[cls._normalize_label(label)] = dict(item)

        ranked = sorted(candidate_map.values(), key=lambda item: item.get("confidence", 0.0), reverse=True)
        anchor = ranked[0] if ranked else None
        anchor_label = str(anchor.get("label", "")) if anchor else ""
        anchor_conf = float(anchor.get("confidence", 0.0)) if anchor else 0.0

        for template in cls.CLARIFICATION_PAIR_BANK:
            left_keywords = tuple(template.get("left_keywords", ()))
            right_keywords = tuple(template.get("right_keywords", ()))
            promotion_gap = float(template.get("counterpart_gap", 0.32))

            if cls._label_matches_keywords(anchor_label, left_keywords):
                counterpart = next(
                    (
                        item for item in ranked
                        if cls._label_matches_keywords(str(item.get("label", "")), right_keywords)
                    ),
                    None,
                )
                counterpart_label = str(counterpart.get("label", "")) if counterpart else str(right_keywords[0])
                counterpart_conf = float(counterpart.get("confidence", 0.0)) if counterpart else 0.0
                promoted = max(counterpart_conf, anchor_conf - promotion_gap, 0.26)
                cls._inject_or_promote_candidate(
                    candidate_map,
                    label=counterpart_label,
                    confidence=promoted,
                    source="ambiguity_pair_expansion",
                )

            if cls._label_matches_keywords(anchor_label, right_keywords):
                counterpart = next(
                    (
                        item for item in ranked
                        if cls._label_matches_keywords(str(item.get("label", "")), left_keywords)
                    ),
                    None,
                )
                counterpart_label = str(counterpart.get("label", "")) if counterpart else str(left_keywords[0])
                counterpart_conf = float(counterpart.get("confidence", 0.0)) if counterpart else 0.0
                promoted = max(counterpart_conf, anchor_conf - promotion_gap, 0.26)
                cls._inject_or_promote_candidate(
                    candidate_map,
                    label=counterpart_label,
                    confidence=promoted,
                    source="ambiguity_pair_expansion",
                )

        lower_respiratory_pattern = next(
            (
                item for item in candidate_map.values()
                if cls._normalize_label(str(item.get("label", ""))) == "possible lower respiratory infection pattern"
            ),
            None,
        )
        if lower_respiratory_pattern is not None:
            pattern_conf = float(lower_respiratory_pattern.get("confidence", 0.35))
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Bronchitis",
                confidence=max(0.40, pattern_conf + 0.05),
                source="respiratory_pattern_expansion",
            )
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Pneumonia",
                confidence=max(0.38, pattern_conf + 0.03),
                source="respiratory_pattern_expansion",
            )
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Bronchospasm / acute asthma exacerbation",
                confidence=max(0.36, pattern_conf + 0.01),
                source="respiratory_pattern_expansion",
            )

            expanded_probe = sorted(candidate_map.values(), key=lambda item: item.get("confidence", 0.0), reverse=True)
            top_label = str(expanded_probe[0].get("label", "")) if expanded_probe else ""
            if expanded_probe and not cls._is_respiratory_label(top_label):
                target_floor = max(0.44, float(expanded_probe[0].get("confidence", 0.0)) - 0.02)
                for item in expanded_probe:
                    label = str(item.get("label", ""))
                    if not cls._is_respiratory_label(label):
                        continue
                    cls._inject_or_promote_candidate(
                        candidate_map,
                        label=label,
                        confidence=max(float(item.get("confidence", 0.0)), target_floor),
                        source="respiratory_context_rebalance",
                    )

        cardiopulmonary_pattern = next(
            (
                item for item in candidate_map.values()
                if cls._normalize_label(str(item.get("label", ""))) == "possible cardiopulmonary red flag symptom pattern"
            ),
            None,
        )
        if cardiopulmonary_pattern is not None:
            pattern_conf = float(cardiopulmonary_pattern.get("confidence", 0.35))
            cls._inject_or_promote_candidate(
                candidate_map,
                label="Pulmonary embolism",
                confidence=max(0.42, pattern_conf + 0.02),
                source="cardiopulmonary_pattern_expansion",
            )

        expanded = sorted(candidate_map.values(), key=lambda item: item.get("confidence", 0.0), reverse=True)
        return expanded

    @classmethod
    def _is_respiratory_label(cls, label: str) -> bool:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return False
        return any(keyword in normalized_label for keyword in cls.RESPIRATORY_LABEL_KEYWORDS)

    @classmethod
    def _is_serious_label(cls, label: str) -> bool:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return False
        return any(keyword in normalized_label for keyword in cls.SERIOUS_DIAGNOSIS_KEYWORDS)

    @classmethod
    def _is_serious_respiratory_label(cls, label: str) -> bool:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return False
        return any(keyword in normalized_label for keyword in cls.SERIOUS_RESPIRATORY_KEYWORDS)

    @classmethod
    def _has_sparse_supporting_signals(
        cls,
        *,
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        final_diagnosis: Optional[Dict[str, Any]],
    ) -> bool:
        supporting_evidence = (final_diagnosis or {}).get("supporting_evidence", []) or []
        signal_count = len({cls._normalize_label(item) for item in patient_symptoms if str(item).strip()})
        signal_count += len(findings)
        signal_count += sum(1 for item in supporting_evidence if str(item).strip())
        return signal_count < cls.SERIOUS_MIN_SIGNAL_COUNT

    @classmethod
    def _requires_serious_follow_up(
        cls,
        *,
        label: str,
        confidence: float,
        candidate_margin: float,
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        final_diagnosis: Optional[Dict[str, Any]],
    ) -> bool:
        if not cls._is_serious_label(label):
            return False

        confidence_threshold = cls.SERIOUS_CLARIFICATION_CONFIDENCE_THRESHOLD
        margin_threshold = cls.SERIOUS_CLARIFICATION_MARGIN_THRESHOLD
        if cls._is_serious_respiratory_label(label):
            confidence_threshold = cls.SERIOUS_RESPIRATORY_CONFIDENCE_THRESHOLD
            margin_threshold = cls.SERIOUS_RESPIRATORY_MARGIN_THRESHOLD

        if confidence < confidence_threshold:
            return True
        if candidate_margin < margin_threshold:
            return True
        if cls._has_sparse_supporting_signals(
            findings=findings,
            patient_symptoms=patient_symptoms,
            final_diagnosis=final_diagnosis,
        ):
            return True
        return False

    @classmethod
    def _label_matches_keywords(cls, label: str, keywords: tuple[str, ...]) -> bool:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return False
        return any(
            cls._labels_overlap(normalized_label, cls._normalize_label(keyword))
            for keyword in keywords
            if str(keyword).strip()
        )

    @classmethod
    def _diagnosis_aligns_with_rules(cls, diagnosis_label: str, rule_conditions: list[str]) -> bool:
        normalized_label = cls._normalize_label(diagnosis_label)
        if not normalized_label or not rule_conditions:
            return False

        for condition in rule_conditions:
            normalized_condition = cls._normalize_label(condition)
            if not normalized_condition:
                continue
            if cls._labels_overlap(normalized_label, normalized_condition):
                return True
            family_keywords = cls.GENERIC_RULE_PATTERN_FAMILY_KEYWORDS.get(normalized_condition, ())
            if family_keywords and cls._label_matches_keywords(normalized_label, family_keywords):
                return True
        return False

    @classmethod
    def _build_clarification(
        cls,
        *,
        report: Dict[str, Any],
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        candidates: list[Dict[str, Any]],
        final_diagnosis: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        reasons = cls._clarification_reasons(
            findings=findings,
            patient_symptoms=patient_symptoms,
            candidates=candidates,
            final_diagnosis=final_diagnosis,
        )
        if not reasons:
            return None

        reported_terms = cls._normalize_label(
            " ".join(
                [
                    str(report.get("raw_text", "") or ""),
                    " ".join(patient_symptoms),
                ]
            )
        )
        arabic_mode = cls._is_arabic_text(str(report.get("raw_text", "") or ""))
        questions: list[Dict[str, Any]] = []
        used_questions: set[str] = set()
        clarification_candidates = cls._expand_clarification_candidates(candidates)
        has_lower_respiratory_pattern = any(
            cls._normalize_label(str(item.get("label", ""))) == "possible lower respiratory infection pattern"
            for item in clarification_candidates
        )
        respiratory_candidate_count = sum(
            int(cls._is_respiratory_label(str(item.get("label", ""))))
            for item in clarification_candidates[:5]
        )

        chronic_dry_cough_context = (
            "dry cough" in reported_terms
            and any(
                cue in reported_terms
                for cue in ("gradual", "persistent", "slowly worsening", "not productive")
            )
        )
        if (has_lower_respiratory_pattern or respiratory_candidate_count >= 2) and chronic_dry_cough_context:
            clarification_map: Dict[str, Dict[str, Any]] = {
                cls._normalize_label(str(item.get("label", ""))): dict(item)
                for item in clarification_candidates
                if str(item.get("label", "")).strip()
            }
            top_respiratory_confidence = max(
                (
                    float(item.get("confidence", 0.0))
                    for item in clarification_candidates
                    if cls._is_respiratory_label(str(item.get("label", "")))
                ),
                default=0.0,
            )
            cls._inject_or_promote_candidate(
                clarification_map,
                label="Sarcoidosis",
                confidence=max(0.46, top_respiratory_confidence - 0.03),
                source="respiratory_pattern_expansion",
            )
            clarification_candidates = sorted(
                clarification_map.values(),
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )

        exertional_angina_context = (
            any(term in reported_terms for term in ("exertion", "with exertion", "exercise", "effort", "exert"))
            and any(
                term in reported_terms
                for term in ("improves with rest", "improves when i rest", "relief with rest", "rest")
            )
            and any(term in reported_terms for term in ("chest pressure", "chest pain", "chest discomfort"))
        )
        if exertional_angina_context:
            clarification_map = {
                cls._normalize_label(str(item.get("label", ""))): dict(item)
                for item in clarification_candidates
                if str(item.get("label", "")).strip()
            }
            top_cardio_confidence = max(
                (float(item.get("confidence", 0.0)) for item in clarification_candidates),
                default=0.0,
            )
            cls._inject_or_promote_candidate(
                clarification_map,
                label="Stable angina",
                confidence=max(0.52, top_cardio_confidence - 0.03),
                source="cardiopulmonary_pattern_expansion",
            )
            clarification_candidates = sorted(
                clarification_map.values(),
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )

        spontaneous_pneumothorax_context = (
            "sudden" in reported_terms
            and any(term in reported_terms for term in ("one sided", "one side", "unilateral"))
            and any(term in reported_terms for term in ("shortness of breath", "dyspnea"))
        )
        if spontaneous_pneumothorax_context:
            clarification_map = {
                cls._normalize_label(str(item.get("label", ""))): dict(item)
                for item in clarification_candidates
                if str(item.get("label", "")).strip()
            }
            top_cardio_confidence = max(
                (float(item.get("confidence", 0.0)) for item in clarification_candidates),
                default=0.0,
            )
            cls._inject_or_promote_candidate(
                clarification_map,
                label="Spontaneous pneumothorax",
                confidence=max(0.48, top_cardio_confidence - 0.08),
                source="cardiopulmonary_pattern_expansion",
            )
            clarification_candidates = sorted(
                clarification_map.values(),
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )

        selected_candidates = clarification_candidates
        if has_lower_respiratory_pattern and not exertional_angina_context:
            respiratory_candidates = [
                item for item in clarification_candidates
                if cls._is_respiratory_label(str(item.get("label", "")))
            ]
            respiratory_candidates.sort(
                key=lambda item: (
                    cls._is_generic_rule_pattern_label(str(item.get("label", ""))),
                    -float(item.get("confidence", 0.0)),
                )
            )
            non_respiratory_candidates = [
                item for item in clarification_candidates
                if not cls._is_respiratory_label(str(item.get("label", "")))
            ]
            selected_candidates = respiratory_candidates + non_respiratory_candidates

        if exertional_angina_context:
            cardiopulmonary_candidates = [
                item for item in selected_candidates
                if cls._label_matches_keywords(
                    str(item.get("label", "")),
                    (
                        "stable angina",
                        "unstable angina",
                        "pulmonary embolism",
                        "myocarditis",
                        "pericarditis",
                        "atrial fibrillation",
                        "psvt",
                    ),
                )
            ]
            cardiopulmonary_candidates.sort(
                key=lambda item: (
                    not cls._label_matches_keywords(str(item.get("label", "")), ("stable angina", "unstable angina")),
                    -float(item.get("confidence", 0.0)),
                )
            )
            if cardiopulmonary_candidates:
                non_cardiopulmonary_candidates = [
                    item for item in selected_candidates
                    if item not in cardiopulmonary_candidates
                ]
                selected_candidates = cardiopulmonary_candidates + non_cardiopulmonary_candidates

            if any(cls._is_serious_label(str(item.get("label", ""))) for item in selected_candidates[:5]):
                selected_candidates = sorted(
                    selected_candidates,
                    key=lambda item: (
                        not cls._is_serious_label(str(item.get("label", ""))),
                        -float(item.get("confidence", 0.0)),
                    ),
                )

        candidate_labels = [candidate["label"] for candidate in selected_candidates[:3]]
        top_confidence = float(selected_candidates[0].get("confidence", 0.0)) if selected_candidates else 0.0
        second_confidence = float(selected_candidates[1].get("confidence", 0.0)) if len(selected_candidates) > 1 else 0.0
        top_margin = top_confidence - second_confidence
        serious_candidate_present = any(cls._is_serious_label(label) for label in candidate_labels)
        serious_respiratory_present = any(cls._is_serious_respiratory_label(label) for label in candidate_labels)
        question_budget = cls.MAX_CLARIFICATION_QUESTIONS
        if top_confidence >= 0.84 and top_margin >= 0.16:
            question_budget = 1
        elif top_confidence >= 0.74 and top_margin >= 0.10:
            question_budget = 2
        if any("close in score" in reason.lower() for reason in reasons):
            question_budget = max(question_budget, 2)
        if serious_candidate_present:
            question_budget = max(question_budget, 2)
        if serious_respiratory_present:
            question_budget = cls.MAX_CLARIFICATION_QUESTIONS
        question_budget = max(1, min(question_budget, cls.MAX_CLARIFICATION_QUESTIONS))

        if len(candidate_labels) >= 2:
            top_left = candidate_labels[0]
            top_right = candidate_labels[1]
            if has_lower_respiratory_pattern:
                respiratory_seed = [
                    str(item.get("label", ""))
                    for item in selected_candidates
                    if cls._is_respiratory_label(str(item.get("label", "")))
                ]
                if len(respiratory_seed) >= 2:
                    pneumonia_label = next(
                        (label for label in respiratory_seed if cls._label_matches_keywords(label, ("pneumonia",))),
                        "",
                    )
                    bronchospasm_label = next(
                        (
                            label for label in respiratory_seed
                            if cls._label_matches_keywords(label, ("bronchospasm", "acute asthma exacerbation", "asthma"))
                        ),
                        "",
                    )
                    bronchitis_label = next(
                        (label for label in respiratory_seed if cls._label_matches_keywords(label, ("bronchitis",))),
                        "",
                    )
                    if pneumonia_label and bronchospasm_label:
                        top_left, top_right = pneumonia_label, bronchospasm_label
                    elif pneumonia_label and bronchitis_label:
                        top_left, top_right = pneumonia_label, bronchitis_label
                    else:
                        top_left, top_right = respiratory_seed[0], respiratory_seed[1]
            for template in cls.CLARIFICATION_PAIR_BANK:
                left_keywords = tuple(template.get("left_keywords", ()))
                right_keywords = tuple(template.get("right_keywords", ()))
                direct_match = (
                    cls._label_matches_keywords(top_left, left_keywords)
                    and cls._label_matches_keywords(top_right, right_keywords)
                )
                reverse_match = (
                    cls._label_matches_keywords(top_left, right_keywords)
                    and cls._label_matches_keywords(top_right, left_keywords)
                )
                if not (direct_match or reverse_match):
                    continue
                if template["question"] in used_questions:
                    continue
                signal_terms = [
                    cls._normalize_label(item)
                    for item in template.get("signals", ())
                    if str(item).strip()
                ]
                if signal_terms and all(signal in reported_terms for signal in signal_terms):
                    continue

                questions.append(
                    {
                        "question": template["question_ar"] if arabic_mode and template.get("question_ar") else template["question"],
                        "type": template.get("type", "multi_select"),
                        "target_conditions": [top_left, top_right],
                        "reason": f"Directly discriminates between {top_left} and {top_right}.",
                    }
                )
                used_questions.add(template["question"])
                break

        prioritized_candidates = selected_candidates

        for candidate in prioritized_candidates[:3]:
            normalized_label = cls._normalize_label(candidate["label"])
            for template in cls.FOLLOW_UP_QUESTION_BANK:
                if not any(keyword in normalized_label for keyword in template["keywords"]):
                    continue
                if template["question"] in used_questions:
                    continue
                if all(signal in reported_terms for signal in template["signals"]):
                    continue
                questions.append(
                    {
                        "question": template["question_ar"] if arabic_mode and template.get("question_ar") else template["question"],
                        "type": template["type"],
                        "target_conditions": [candidate["label"]],
                        "reason": f"Helps distinguish whether the presentation fits {candidate['label']}.",
                    }
                )
                used_questions.add(template["question"])
                if len(questions) >= question_budget:
                    break
            if len(questions) >= question_budget:
                break

        if not questions:
            generic_candidates = ", ".join(candidate_labels[:3]) or "the top candidate conditions"
            generic_questions = [
                "ما العرض الأكثر وضوحاً الآن: الألم أم صعوبة التنفس أم الحمى أم الضعف؟" if arabic_mode else "Which symptom is most prominent right now: pain, breathing trouble, fever, or weakness?",
                "هل لاحظت أي علامة خطورة مثل الإغماء أو ضيق تنفس شديد أو ألم يزداد بسرعة؟" if arabic_mode else "Have you noticed any red-flag symptom such as fainting, severe shortness of breath, or rapidly worsening pain?",
            ]
            for question in generic_questions:
                if question in used_questions:
                    continue
                questions.append(
                    {
                        "question": question,
                        "type": "free_text",
                        "target_conditions": candidate_labels[:3],
                        "reason": f"Provides extra detail to separate {generic_candidates}.",
                    }
                )
                used_questions.add(question)
                if len(questions) >= question_budget:
                    break

        return {
            "needed": True,
            "mode": "follow_up_questions",
            "reasons": reasons,
            "questions": questions[:question_budget],
            "candidate_diseases": [
                {
                    "label": candidate["label"],
                    "confidence": round(candidate["confidence"], 2),
                    "sources": candidate["sources"],
                }
                for candidate in selected_candidates[:3]
            ],
        }

    @classmethod
    def _canonicalize_answer_signals(cls, answer_text: str) -> str:
        normalized = cls._normalize_label(answer_text)
        if not normalized:
            return ""

        canonicalized = f" {normalized} "
        for canonical_signal, aliases in cls.ANSWER_SIGNAL_ALIASES.items():
            canonical = cls._normalize_label(canonical_signal)
            if not canonical:
                continue
            for alias in aliases:
                normalized_alias = cls._normalize_label(alias)
                if not normalized_alias:
                    continue
                canonicalized = re.sub(
                    rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
                    canonical,
                    canonicalized,
                    flags=re.IGNORECASE,
                )

        return " ".join(canonicalized.split())

    @classmethod
    def _is_negated_signal(cls, answer_text: str, signal: str) -> bool:
        raw_answer = cls._normalize_label(answer_text)
        normalized_answer = cls._canonicalize_answer_signals(answer_text)
        normalized_signal = cls._normalize_label(signal)
        if not raw_answer or not normalized_signal:
            return False
        if normalized_signal not in normalized_answer and normalized_signal not in raw_answer:
            return False

        negation_prefixes = (
            "no",
            "not",
            "without",
            "denies",
            "deny",
            "negative for",
            "absence of",
            "no evidence of",
            "لا",
            "لا يوجد",
            "بدون",
            "ينفي",
            "مش",
            "مافي",
            "ما في",
            "ما عندي",
            "ليس",
        )
        for prefix in negation_prefixes:
            pattern = re.compile(
                rf"\b{re.escape(prefix)}\b(?:\s+\w+){{0,8}}\s+{re.escape(normalized_signal)}\b",
                re.IGNORECASE,
            )
            if pattern.search(normalized_answer):
                return True

        if any(
            token in normalized_answer
            for token in (
                f"لا {normalized_signal}",
                f"بدون {normalized_signal}",
                f"لا يوجد {normalized_signal}",
                f"مافي {normalized_signal}",
                f"ما في {normalized_signal}",
            )
        ):
            return True

        postfix_pattern = re.compile(
            rf"\b{re.escape(normalized_signal)}\b(?:\s+\w+){{0,4}}\s+(?:absent|negative|not\s+present|ruled\s+out)\b",
            re.IGNORECASE,
        )
        return bool(postfix_pattern.search(normalized_answer))

    @classmethod
    def _comparative_label_bias(cls, normalized_label: str, normalized_answer: str) -> float:
        if not normalized_label or not normalized_answer:
            return 0.0

        bonus_patterns = (
            f"more like {normalized_label}",
            f"sounds like {normalized_label}",
            f"consistent with {normalized_label}",
            f"fits {normalized_label}",
        )
        penalty_patterns = (
            f"rather than {normalized_label}",
            f"instead of {normalized_label}",
            f"than {normalized_label}",
            f"not {normalized_label}",
        )

        score = 0.0
        if any(pattern in normalized_answer for pattern in bonus_patterns):
            score += 0.28
        if any(pattern in normalized_answer for pattern in penalty_patterns):
            score -= 0.28
        return score

    @classmethod
    def _signal_match_score(cls, label: str, answer_text: str) -> float:
        normalized_label = cls._normalize_label(label)
        normalized_answer = cls._canonicalize_answer_signals(answer_text)
        score = 0.0
        score += cls._comparative_label_bias(normalized_label, normalized_answer)
        if normalized_label and normalized_label in normalized_answer:
            score += 0.45
        for template in cls.FOLLOW_UP_QUESTION_BANK:
            if not any(keyword in normalized_label for keyword in template["keywords"]):
                continue
            matched_signals = 0
            negated_signals = 0
            for signal in template["signals"]:
                normalized_signal = cls._normalize_label(signal)
                if not normalized_signal or normalized_signal not in normalized_answer:
                    continue
                if cls._is_negated_signal(normalized_answer, normalized_signal):
                    negated_signals += 1
                else:
                    matched_signals += 1
            score += min(0.24, 0.08 * matched_signals)
            score -= min(0.12, 0.04 * negated_signals)
        for bank_label, profile in cls.DIAGNOSTIC_SIGNAL_BANK.items():
            if bank_label not in normalized_label:
                continue
            positive_hits = 0
            positive_negated_hits = 0
            negative_hits = 0
            negative_negated_hits = 0

            for signal in profile.get("positive", ()):
                normalized_signal = cls._normalize_label(signal)
                if not normalized_signal or normalized_signal not in normalized_answer:
                    continue
                if cls._is_negated_signal(normalized_answer, normalized_signal):
                    positive_negated_hits += 1
                else:
                    positive_hits += 1

            for signal in profile.get("negative", ()):
                normalized_signal = cls._normalize_label(signal)
                if not normalized_signal or normalized_signal not in normalized_answer:
                    continue
                if cls._is_negated_signal(normalized_answer, normalized_signal):
                    negative_negated_hits += 1
                else:
                    negative_hits += 1

            score += min(0.4, 0.11 * positive_hits)
            score -= min(0.35, 0.11 * negative_hits)
            score -= min(0.18, 0.06 * positive_negated_hits)
            score += min(0.12, 0.04 * negative_negated_hits)
            break

        infection_over_wheeze = (
            "infection" in normalized_answer
            and "wheez" in normalized_answer
            and any(phrase in normalized_answer for phrase in ("more than", "rather than", "instead of"))
        )
        if "pneumonia" in normalized_label:
            if infection_over_wheeze:
                score += 0.24
            if "pleuritic" in normalized_answer and "productive cough" in normalized_answer:
                score += 0.12
        if "bronchospasm" in normalized_label or "asthma" in normalized_label:
            if infection_over_wheeze:
                score -= 0.24
        if "bronchitis" in normalized_label:
            if infection_over_wheeze and "pleuritic" in normalized_answer:
                score -= 0.16

        stable_angina_over_arrhythmia = (
            "stable angina" in normalized_answer
            and any(
                term in normalized_answer
                for term in ("arrhythmia", "irregular heartbeat", "atrial fibrillation", "psvt")
            )
            and any(phrase in normalized_answer for phrase in ("more than", "rather than", "instead of"))
        )
        exertion_with_rest_relief = (
            any(term in normalized_answer for term in ("exertion", "exercise", "effort", "exert"))
            and any(
                phrase in normalized_answer
                for phrase in ("improves with rest", "better with rest", "relief with rest")
            )
        )

        if "stable angina" in normalized_label:
            if stable_angina_over_arrhythmia:
                score += 0.34
            if exertion_with_rest_relief:
                score += 0.16

        if any(term in normalized_label for term in ("atrial fibrillation", "psvt", "myocarditis")):
            if stable_angina_over_arrhythmia:
                score -= 0.24
            if exertion_with_rest_relief and "myocarditis" in normalized_label:
                score -= 0.10

        if "unstable angina" in normalized_label and exertion_with_rest_relief:
            score -= 0.12
        if "pulmonary embolism" in normalized_label and stable_angina_over_arrhythmia:
            score -= 0.18
        if "bronchitis" in normalized_label and stable_angina_over_arrhythmia:
            score -= 0.16

        scombroid_food_triggered = (
            any(
                term in normalized_answer
                for term in ("after eating fish", "soon after eating fish", "food triggered", "flushing")
            )
            and any(term in normalized_answer for term in ("palpitations", "headache", "chest discomfort"))
        )
        if "scombroid food poisoning" in normalized_label and scombroid_food_triggered:
            score += 0.32
        if any(term in normalized_label for term in ("atrial fibrillation", "psvt")) and scombroid_food_triggered:
            score -= 0.14
        return score

    @classmethod
    def _is_generic_rule_pattern_label(cls, label: str) -> bool:
        normalized = cls._normalize_label(label)
        return normalized.startswith("possible ") and " pattern" in normalized

    @classmethod
    def _collect_canonicalization_candidates(cls, diagnosis_payload: Dict[str, Any]) -> list[dict[str, Any]]:
        pool: Dict[str, Dict[str, Any]] = {}

        def _push_candidate(label: str, confidence: Any) -> None:
            clean_label = str(label).strip()
            if not clean_label or cls._is_generic_rule_pattern_label(clean_label):
                return
            normalized = cls._normalize_label(clean_label)
            if not normalized:
                return
            candidate_confidence = cls._normalize_confidence(confidence)
            existing = pool.get(normalized)
            if existing is None or candidate_confidence > existing["confidence"]:
                pool[normalized] = {
                    "label": clean_label,
                    "confidence": candidate_confidence,
                }

        for item in diagnosis_payload.get("diagnostic_candidates", []) or []:
            _push_candidate(item.get("label", ""), item.get("confidence", 0.0))

        classifier_prediction = diagnosis_payload.get("classifier_prediction", {}) or {}
        for item in classifier_prediction.get("top_predictions", []) or []:
            _push_candidate(item.get("label", ""), item.get("confidence", 0.0))

        return sorted(pool.values(), key=lambda entry: entry["confidence"], reverse=True)

    @classmethod
    def _label_matches_any_keyword(cls, label: str, keywords: tuple[str, ...]) -> bool:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return False
        return any(
            cls._normalize_label(keyword) in normalized_label
            for keyword in keywords
            if str(keyword).strip()
        )

    @classmethod
    def _canonicalize_generic_rule_label(cls, label: str, diagnosis_payload: Dict[str, Any]) -> str:
        normalized = cls._normalize_label(label)
        if not cls._is_generic_rule_pattern_label(label):
            return label

        direct = cls.GENERIC_RULE_PATTERN_DIRECT_MAP.get(normalized)
        if direct:
            return direct

        keywords = cls.GENERIC_RULE_PATTERN_FAMILY_KEYWORDS.get(normalized)
        if not keywords:
            return label

        candidates = cls._collect_canonicalization_candidates(diagnosis_payload)
        for candidate in candidates:
            if cls._label_matches_any_keyword(candidate["label"], keywords):
                return candidate["label"]
        return label

    @classmethod
    def _apply_label_canonicalization(cls, diagnosis_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(diagnosis_payload, dict):
            return diagnosis_payload

        final = diagnosis_payload.get("final_diagnosis", {}) or {}
        label = str(final.get("diagnosis", "")).strip()
        if not label:
            return diagnosis_payload

        canonical_label = cls._canonicalize_generic_rule_label(label, diagnosis_payload)
        if canonical_label == label:
            return diagnosis_payload

        updated = dict(diagnosis_payload)
        updated_final = dict(final)
        updated_final["diagnosis"] = canonical_label
        updated_final["canonicalized_from"] = label
        reasoning = str(updated_final.get("reasoning", "")).strip()
        canonical_note = f" Diagnostic label canonicalized from '{label}' to '{canonical_label}'."
        if canonical_note not in reasoning:
            updated_final["reasoning"] = (reasoning + canonical_note).strip()
        updated["final_diagnosis"] = updated_final
        return updated

    @classmethod
    def apply_follow_up_scoring(
        cls,
        diagnosis: Dict[str, Any],
        *,
        answers: list[str],
        prior_diagnosis: Optional[Dict[str, Any]] = None,
        normalized_follow_up_text: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(diagnosis, dict):
            return diagnosis

        rescored = dict(diagnosis)
        normalized_answers = [str(item).strip() for item in answers if str(item).strip()]
        normalized_follow_up = cls._canonicalize_answer_signals(normalized_follow_up_text)
        if not normalized_answers and not normalized_follow_up:
            return rescored

        candidate_map: Dict[str, Dict[str, Any]] = {}
        for item in diagnosis.get("diagnostic_candidates", []) or []:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            candidate_map[cls._normalize_label(label)] = {
                "label": label,
                "confidence": cls._normalize_confidence(item.get("confidence")),
                "sources": list(item.get("sources", []) or []),
            }

        prior = prior_diagnosis or {}
        prior_clarification = prior.get("clarification", {}) or {}
        for item in prior_clarification.get("candidate_diseases", []) or []:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            normalized_label = cls._normalize_label(label)
            existing = candidate_map.get(normalized_label)
            confidence = cls._normalize_confidence(item.get("confidence"))
            if existing is None or confidence > existing["confidence"]:
                candidate_map[normalized_label] = {
                    "label": label,
                    "confidence": confidence,
                    "sources": list(item.get("sources", []) or []),
                }

        if not candidate_map:
            final_label = str((diagnosis.get("final_diagnosis", {}) or {}).get("diagnosis", "")).strip()
            if final_label:
                candidate_map[cls._normalize_label(final_label)] = {
                    "label": final_label,
                    "confidence": cls._normalize_confidence((diagnosis.get("final_diagnosis", {}) or {}).get("confidence")),
                    "sources": [str((diagnosis.get("final_diagnosis", {}) or {}).get("source", "unknown"))],
                }

        question_target_answers: Dict[str, list[str]] = {}
        question_pair_bonus: Dict[str, float] = {}
        for idx, question in enumerate(prior_clarification.get("questions", []) or []):
            answer = normalized_answers[idx] if idx < len(normalized_answers) else ""
            if not answer:
                continue
            normalized_targets: list[str] = []
            target_scores: Dict[str, float] = {}
            for label in question.get("target_conditions", []) or []:
                normalized_label = cls._normalize_label(label)
                if not normalized_label:
                    continue
                question_target_answers.setdefault(normalized_label, []).append(answer)
                if normalized_label not in normalized_targets:
                    normalized_targets.append(normalized_label)
                    target_scores[normalized_label] = cls._signal_match_score(label, answer)

            if len(normalized_targets) >= 2:
                ranked_scores = sorted(
                    target_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                best_label, best_score = ranked_scores[0]
                second_label, second_score = ranked_scores[1]
                score_margin = best_score - second_score
                if score_margin >= 0.08:
                    question_pair_bonus[best_label] = question_pair_bonus.get(best_label, 0.0) + min(0.45, score_margin * 0.7)
                    question_pair_bonus[second_label] = question_pair_bonus.get(second_label, 0.0) - min(0.25, score_margin * 0.35)

        rescored_candidates = []
        answers_blob = " ".join(normalized_answers)
        if normalized_follow_up:
            answers_blob = " ".join(part for part in (answers_blob, normalized_follow_up) if part)
        for normalized_label, candidate in candidate_map.items():
            base_confidence = float(candidate["confidence"])
            targeted_answers = question_target_answers.get(normalized_label, [])
            targeted_blob = " ".join(targeted_answers)
            if normalized_follow_up:
                targeted_blob = " ".join(part for part in (targeted_blob, normalized_follow_up) if part)

            adjusted = base_confidence
            overall_signal_score = cls._signal_match_score(candidate["label"], answers_blob)
            if targeted_answers:
                adjusted += 0.03 * min(len(targeted_answers), 3)
                targeted_signal_score = cls._signal_match_score(candidate["label"], targeted_blob)
                adjusted += targeted_signal_score * 0.6
                adjusted += overall_signal_score * 0.2
            else:
                adjusted += overall_signal_score * 0.55

            if normalized_follow_up:
                follow_up_signal_score = cls._signal_match_score(candidate["label"], normalized_follow_up)
                adjusted += follow_up_signal_score * 0.15

            adjusted += question_pair_bonus.get(normalized_label, 0.0)

            clarification_gain = adjusted - base_confidence
            rescored_candidates.append(
                {
                    "label": candidate["label"],
                    "confidence": round(max(0.0, min(adjusted, 0.99)), 2),
                    "sources": candidate["sources"],
                    "_clarification_gain": clarification_gain,
                    "_signal_score": overall_signal_score,
                }
            )

        rescored_candidates.sort(
            key=lambda item: (item["confidence"], item.get("_clarification_gain", 0.0)),
            reverse=True,
        )
        rescored["diagnostic_candidates"] = [
            {
                "label": item["label"],
                "confidence": item["confidence"],
                "sources": item["sources"],
            }
            for item in rescored_candidates
        ]

        best_candidate_label = ""
        best_candidate_confidence = 0.0
        best_candidate_gain = 0.0
        override_applied = False

        if rescored_candidates:
            best = rescored_candidates[0]
            best_candidate_label = best["label"]
            best_candidate_confidence = best["confidence"]
            best_candidate_gain = float(best.get("_clarification_gain", 0.0))
            final = dict(rescored.get("final_diagnosis", {}) or {})
            previous_label = str(final.get("diagnosis", "")).strip()
            previous_confidence = cls._normalize_confidence(final.get("confidence"))
            changed_label = cls._normalize_label(previous_label) != cls._normalize_label(best["label"])
            confidence_gap = best["confidence"] - previous_confidence
            previous_signal_score = cls._signal_match_score(previous_label, answers_blob) if previous_label else 0.0
            best_signal_score = float(best.get("_signal_score", 0.0))
            signal_advantage = best_signal_score - previous_signal_score
            leader_margin = (
                best["confidence"] - rescored_candidates[1]["confidence"]
                if len(rescored_candidates) > 1
                else best["confidence"]
            )
            if not previous_label:
                should_override = True
            elif not changed_label:
                should_override = best["confidence"] > previous_confidence + 0.01
            else:
                robust_override = (
                    confidence_gap >= max(cls.CLARIFICATION_OVERRIDE_MARGIN, 0.08)
                    and best_candidate_gain >= cls.CLARIFICATION_OVERRIDE_GAIN_THRESHOLD
                    and leader_margin >= cls.CLARIFICATION_LEADER_MARGIN
                )
                signal_supported_override = (
                    confidence_gap >= 0.02
                    and best_candidate_gain >= 0.20
                    and best_signal_score >= 0.20
                    and signal_advantage >= 0.22
                )
                should_override = (
                    robust_override
                    or signal_supported_override
                )
            if should_override:
                override_applied = True
                final.update(
                    {
                        "diagnosis": best["label"],
                        "confidence": best["confidence"],
                        "source": "clarification_rerank",
                        "mode": "interactive_refinement",
                        "reasoning": "Final diagnosis updated after scoring the follow-up answers against the clarification candidates.",
                    }
                )
                rescored["final_diagnosis"] = final
                rescored["summary"] = (
                    f"After clarification, the leading diagnosis is {best['label']} "
                    f"(confidence {best['confidence']:.2f})."
                )

        clarification = rescored.get("clarification", {}) or {}
        if clarification:
            # Treat a submitted clarification round as closed to avoid re-asking
            # the same follow-up questions on the next client turn.
            clarification["needed"] = False
            clarification["completed"] = True
            clarification["applied"] = True
            clarification["answers_used"] = normalized_answers
            clarification["rerank_top_label"] = best_candidate_label
            clarification["rerank_top_confidence"] = round(best_candidate_confidence, 2)
            clarification["rerank_top_gain"] = round(best_candidate_gain, 3)
            clarification["override_applied"] = override_applied
            rescored["clarification"] = clarification
        return cls._apply_label_canonicalization(rescored)

    @classmethod
    def _normalize_label(cls, value: str) -> str:
        return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())

    @staticmethod
    def _is_arabic_text(text: str) -> bool:
        if not text or not str(text).strip():
            return False
        raw = str(text)
        arabic_chars = sum(1 for char in raw if "\u0600" <= char <= "\u06ff")
        return arabic_chars / max(len(raw), 1) > 0.2

    @classmethod
    def _is_symptom_like_label(cls, label: str, patient_symptoms: list[str]) -> bool:
        normalized_label = cls._normalize_label(label)
        if not normalized_label:
            return True
        normalized_symptoms = {cls._normalize_label(item) for item in patient_symptoms if str(item).strip()}
        if normalized_label in normalized_symptoms:
            return True
        if normalized_label in cls.NON_DIAGNOSIS_TERMS:
            return True
        return any(
            normalized_label == symptom
            or normalized_label in symptom
            or symptom in normalized_label
            for symptom in normalized_symptoms
            if symptom
        )

    @classmethod
    def _build_rules_fallback_diagnosis(cls, rule_conditions: list[str]) -> Optional[Dict[str, Any]]:
        if not rule_conditions:
            return None
        unique_conditions = list(dict.fromkeys(rule_conditions))
        return {
            "diagnosis": unique_conditions[0],
            "confidence": 0.58,
            "source": "rules_fallback",
            "mode": "rules_fallback",
            "reasoning": "AI diagnostic layers were inconclusive or conflicted with rule-based clinical signals, so the system fell back to deterministic medical rules.",
            "supporting_evidence": [
                "Rule findings: " + ", ".join(unique_conditions),
            ],
            "rule_alignment": True,
        }

    @classmethod
    def _should_prefer_rules_over_ai(
        cls,
        *,
        findings: list[Dict[str, Any]],
        selected_label: str,
        selected_confidence: float,
        classifier_prediction: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not findings:
            return False

        selected_label_normalized = cls._normalize_label(selected_label)
        lab_rule_findings = [
            item for item in findings
            if str(item.get("source", "")).strip().lower() == "lab_rules"
        ]
        if lab_rule_findings:
            lab_rule_labels = [
                cls._normalize_label(str(item.get("condition", "")).strip())
                for item in lab_rule_findings
                if str(item.get("condition", "")).strip()
            ]
            lab_rule_conflicts = not any(
                cls._labels_overlap(selected_label_normalized, rule_label)
                for rule_label in lab_rule_labels
                if rule_label
            )
            if lab_rule_conflicts:
                return True

        symptom_rule_findings = [
            item for item in findings
            if str(item.get("source", "")).strip().lower() == "symptom_rules"
        ]
        if not symptom_rule_findings:
            return False

        rule_labels = [
            cls._normalize_label(str(item.get("condition", "")).strip())
            for item in symptom_rule_findings
            if str(item.get("condition", "")).strip()
        ]
        rule_conflicts = not any(
            cls._labels_overlap(selected_label_normalized, rule_label)
            for rule_label in rule_labels
            if rule_label
        )
        if not rule_conflicts:
            return False

        generic_pattern_labels = [
            rule_label
            for rule_label in rule_labels
            if rule_label in cls.GENERIC_RULE_PATTERN_FAMILY_KEYWORDS
        ]
        for generic_label in generic_pattern_labels:
            family_keywords = cls.GENERIC_RULE_PATTERN_FAMILY_KEYWORDS.get(generic_label, ())
            if cls._label_matches_keywords(selected_label_normalized, family_keywords):
                return False

        max_rule_severity = max(
            (
                {"critical": 4, "high": 3, "moderate": 2, "low": 1, "info": 0}.get(
                    str(item.get("severity", "info")).strip().lower(),
                    0,
                )
                for item in symptom_rule_findings
            ),
            default=0,
        )
        classifier_confidence = (
            cls._normalize_confidence(classifier_prediction.get("confidence"))
            if classifier_prediction else 0.0
        )

        return (
            max_rule_severity >= 2
            and selected_confidence < cls.RULE_GATING_AI_CONFIDENCE_THRESHOLD
            and classifier_confidence < cls.CLASSIFIER_PRIMARY_THRESHOLD
        )

    @classmethod
    def _build_final_diagnosis(
        cls,
        *,
        findings: list[Dict[str, Any]],
        patient_symptoms: list[str],
        report_text: str = "",
        rag_out: Optional[Dict[str, Any]] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        rule_conditions = [str(item.get("condition", "")) for item in findings if item.get("condition")]
        candidates = cls._collect_diagnostic_candidates(
            findings=findings,
            patient_symptoms=patient_symptoms,
            report_text=report_text,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
        )
        ai_candidates = [
            candidate
            for candidate in candidates
            if not cls._is_generic_rule_pattern_label(str(candidate.get("label", "")))
            and any(
                source in {
                    "classifier",
                    "rag",
                    "rag_retrieval",
                    "respiratory_pattern_expansion",
                    "respiratory_context_rebalance",
                    "cardiopulmonary_pattern_expansion",
                }
                for source in candidate["sources"]
            )
        ]

        classifier_label = (
            str(classifier_prediction.get("predicted_label", "")).strip()
            if classifier_prediction
            else ""
        )
        rag_label = ""
        for candidate in ai_candidates:
            if any(source in {"rag", "rag_retrieval"} for source in candidate["sources"]):
                rag_label = candidate["label"]
                break

        classifier_primary = (
            classifier_prediction is not None
            and cls._normalize_confidence(classifier_prediction.get("confidence")) >= cls.CLASSIFIER_PRIMARY_THRESHOLD
        )
        classifier_supportive = (
            classifier_prediction is not None
            and cls._normalize_confidence(classifier_prediction.get("confidence")) >= cls.CLASSIFIER_SUPPORT_THRESHOLD
        )
        rag_agrees_with_classifier = cls._labels_overlap(rag_label, classifier_label)

        if rag_agrees_with_classifier and classifier_supportive and classifier_label:
            best_confidence = max(
                cls._normalize_confidence(classifier_prediction.get("confidence")),
                next(
                    (
                        candidate["confidence"]
                        for candidate in ai_candidates
                        if any(source in {"rag", "rag_retrieval"} for source in candidate["sources"])
                    ),
                    0.0,
                ),
            )
            supporting_evidence = [
                "Classifier and RAG converged on the same diagnosis family.",
            ]
            if rule_conditions:
                supporting_evidence.append(
                    "Rule safety layer flagged: " + ", ".join(dict.fromkeys(rule_conditions))
                )
            if cls._should_prefer_rules_over_ai(
                findings=findings,
                selected_label=classifier_label,
                selected_confidence=best_confidence,
                classifier_prediction=classifier_prediction,
            ):
                return cls._build_rules_fallback_diagnosis(rule_conditions)
            return {
                "diagnosis": classifier_label,
                "confidence": round(min(best_confidence + 0.08, 0.98), 2),
                "source": "classifier_rag_consensus",
                "mode": "ai_primary",
                "reasoning": "Final diagnosis selected from agreement between the fine-tuned classifier and RAG evidence.",
                "supporting_evidence": supporting_evidence,
                "rule_alignment": cls._diagnosis_aligns_with_rules(classifier_label, rule_conditions),
            }

        if ai_candidates:
            classifier_candidate = next(
                (
                    item for item in ai_candidates
                    if "classifier" in item["sources"] and cls._labels_overlap(item["label"], classifier_label)
                ),
                None,
            )
            rag_candidate = next(
                (
                    item for item in ai_candidates
                    if any(source in {"rag", "rag_retrieval"} for source in item["sources"])
                ),
                None,
            )
            ranked_candidates = sorted(
                ai_candidates,
                key=lambda item: (
                    item["confidence"],
                    "classifier" in item["sources"],
                    "rag" in item["sources"] or "rag_retrieval" in item["sources"],
                ),
                reverse=True,
            )
            selected = ranked_candidates[0]
            if classifier_candidate and classifier_primary:
                selected = classifier_candidate
                if rag_candidate and not cls._labels_overlap(classifier_candidate["label"], rag_candidate["label"]):
                    rag_advantage = rag_candidate["confidence"] - classifier_candidate["confidence"]
                    if rag_advantage >= cls.CLASSIFIER_OVERRIDE_MARGIN and not any(
                        cls._labels_overlap(classifier_candidate["label"], condition)
                        for condition in rule_conditions
                    ):
                        selected = rag_candidate
            elif classifier_candidate and classifier_supportive:
                selected = ranked_candidates[0]
                if cls._labels_overlap(selected["label"], classifier_candidate["label"]):
                    selected = classifier_candidate
                elif (
                    selected["confidence"] - classifier_candidate["confidence"]
                    <= cls.CLASSIFIER_OVERRIDE_MARGIN
                ):
                    selected = classifier_candidate
            elif "classifier" in selected["sources"] and not classifier_primary and rag_candidate is not None:
                selected = rag_candidate

            direct_rule_candidates = [
                candidate for candidate in candidates
                if "symptom_rules" in candidate["sources"]
                and not cls._is_generic_rule_pattern_label(str(candidate.get("label", "")))
                and not cls._labels_overlap(str(candidate.get("label", "")), str(selected.get("label", "")))
            ]
            if direct_rule_candidates:
                severity_by_label = {
                    cls._normalize_label(str(item.get("condition", ""))): str(item.get("severity", "info")).lower()
                    for item in findings
                    if str(item.get("condition", "")).strip()
                }
                severity_score = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "info": 0}

                def rule_candidate_rank(item: Dict[str, Any]) -> tuple[float, int]:
                    label_key = cls._normalize_label(str(item.get("label", "")))
                    severity = severity_score.get(severity_by_label.get(label_key, "info"), 0)
                    return (float(item.get("confidence", 0.0)), severity)

                best_rule_candidate = max(direct_rule_candidates, key=rule_candidate_rank)
                best_rule_confidence, best_rule_severity = rule_candidate_rank(best_rule_candidate)
                selected_confidence = float(selected.get("confidence", 0.0))
                if (
                    best_rule_confidence >= selected_confidence + 0.03
                    or (
                        best_rule_severity >= 3
                        and selected_confidence < 0.85
                        and best_rule_confidence + 0.10 >= selected_confidence
                    )
                    or (
                        best_rule_severity >= 2
                        and selected_confidence < cls.CLASSIFIER_PRIMARY_THRESHOLD
                        and best_rule_confidence + 0.08 >= selected_confidence
                    )
                ):
                    selected = best_rule_candidate

            supporting_evidence = list(dict.fromkeys(selected["evidence"]))
            if rule_conditions:
                supporting_evidence.append(
                    "Rule safety findings: " + ", ".join(dict.fromkeys(rule_conditions))
                )
            if cls._should_prefer_rules_over_ai(
                findings=findings,
                selected_label=selected["label"],
                selected_confidence=selected["confidence"],
                classifier_prediction=classifier_prediction,
            ):
                return cls._build_rules_fallback_diagnosis(rule_conditions)
            return {
                "diagnosis": selected["label"],
                "confidence": round(selected["confidence"], 2),
                "source": selected["sources"][0],
                "mode": "ai_primary",
                "reasoning": selected["reasoning"],
                "supporting_evidence": supporting_evidence,
                "rule_alignment": cls._diagnosis_aligns_with_rules(selected["label"], rule_conditions),
            }

        if findings:
            return cls._build_rules_fallback_diagnosis(rule_conditions)

        return None

    @classmethod
    def _build_decision_fusion(
        cls,
        findings: list[Dict[str, Any]],
        *,
        rag_out: Optional[Dict[str, Any]] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None,
        final_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        finding_sources = [str(item.get("source", "lab_rules")) for item in findings]
        lab_rule_findings_count = sum(1 for source in finding_sources if source == "lab_rules")
        symptom_rule_findings_count = sum(1 for source in finding_sources if source == "symptom_rules")
        rule_conditions = [str(item.get("condition", "")).lower() for item in findings]
        classifier_label = (
            str(classifier_prediction.get("predicted_label", "")).lower()
            if classifier_prediction
            else ""
        )
        classifier_agrees = None
        if classifier_label:
            classifier_agrees = any(
                classifier_label in condition or condition in classifier_label
                for condition in rule_conditions
                if condition
            )

        primary_source = str((final_diagnosis or {}).get("source") or "rules")
        supporting_sources: list[str] = []
        if rag_out:
            supporting_sources.append("rag")
        if classifier_prediction:
            supporting_sources.append("classifier")
        if lab_rule_findings_count:
            supporting_sources.append("lab_rules")
        if symptom_rule_findings_count:
            supporting_sources.append("symptom_rules")
        if primary_source not in supporting_sources:
            supporting_sources.insert(0, primary_source)

        rule_validation_status = "not_available"
        if findings and final_diagnosis:
            if final_diagnosis.get("rule_alignment"):
                rule_validation_status = "aligned"
            else:
                rule_validation_status = "safety_flagged"
        elif findings:
            rule_validation_status = "fallback_only"

        return {
            "primary_source": primary_source,
            "supporting_sources": list(dict.fromkeys(supporting_sources)),
            "rule_findings_count": len(findings),
            "lab_rule_findings_count": lab_rule_findings_count,
            "symptom_rule_findings_count": symptom_rule_findings_count,
            "rag_used": bool(rag_out),
            "classifier_used": bool(classifier_prediction),
            "classifier_agrees_with_rules": classifier_agrees,
            "rule_validation_status": rule_validation_status,
            "rag_confidence": (rag_out or {}).get("rag_confidence") if rag_out else None,
            "rag_scope_status": (rag_out or {}).get("rag_scope_status") if rag_out else None,
            "rag_usable_for_fusion": (
                ((rag_out or {}).get("rag_confidence") or {}).get("usable_for_fusion")
                if rag_out
                else None
            ),
            "final_assessment_basis": (
                "Fine-tuned model predictions and RAG evidence drive the final diagnosis whenever available. "
                "Clinical rules act as a deterministic safety and escalation layer."
            ),
        }

    @classmethod
    def _build_summary(
        cls,
        findings_payload: list[Dict[str, Any]],
        *,
        final_diagnosis: Optional[Dict[str, Any]] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None,
        clarification: Optional[Dict[str, Any]] = None,
        response_language: str = "en",
    ) -> str:
        language = normalize_language(response_language)
        arabic_mode = language == "ar"

        if clarification and clarification.get("needed"):
            candidate_labels = [item.get("label", "") for item in clarification.get("candidate_diseases", [])]
            if candidate_labels:
                joined = ", ".join(candidate_labels[:3])
                if arabic_mode:
                    return (
                        "ما زال التقييم الأولي غير محسوم. "
                        f"الاحتمالات الأبرز حالياً هي: {joined}. "
                        "الإجابة عن أسئلة المتابعة ستساعد على تحسين دقة التشخيص."
                    )
                return (
                    "The first-pass assessment is still uncertain. "
                    f"Current leading possibilities are {joined}. "
                    "Answering the follow-up questions will help refine the diagnosis."
                )
            return (
                "ما زال التقييم الأولي غير مؤكد. "
                "نحتاج إلى أسئلة متابعة قبل إصدار استنتاج تشخيصي أقوى."
                if arabic_mode
                else "The first-pass assessment is still uncertain. "
                "Follow-up questions are needed before making a stronger diagnostic claim."
            )

        if final_diagnosis:
            diagnosis = final_diagnosis.get("diagnosis", "an undetermined condition")
            confidence = cls._normalize_confidence(final_diagnosis.get("confidence"))
            source = str(final_diagnosis.get("source", "ai"))
            if findings_payload:
                if arabic_mode:
                    return (
                        f"يشير التقييم المدعوم بالذكاء الاصطناعي إلى {diagnosis} "
                        f"(الثقة {confidence:.2f}، المصدر: {source}) مع إرفاق فحوصات الأمان المبنية على القواعد."
                    )
                return (
                    f"AI-assisted assessment suggests {diagnosis} "
                    f"(confidence {confidence:.2f}, source: {source}) with rule-based safety checks attached."
                )
            if arabic_mode:
                return (
                    f"يشير التقييم المدعوم بالذكاء الاصطناعي إلى {diagnosis} "
                    f"(الثقة {confidence:.2f}، المصدر: {source})."
                )
            return (
                f"AI-assisted assessment suggests {diagnosis} "
                f"(confidence {confidence:.2f}, source: {source})."
            )

        if findings_payload:
            unique_conditions = ", ".join(
                dict.fromkeys(str(item.get("condition", "Unknown finding")) for item in findings_payload)
            )
            if any(item.get("source") == "symptom_rules" for item in findings_payload):
                if arabic_mode:
                    return (
                        "لم يتم اكتشاف نتائج غير طبيعية في قواعد التحاليل المخبرية، "
                        f"لكن قواعد الأعراض تشير إلى: {unique_conditions}."
                    )
                return (
                    "No abnormal lab-rule findings were detected, but symptom-based rules suggest: "
                    f"{unique_conditions}."
                )
            if arabic_mode:
                return f"تم رصد {len(findings_payload)} نتائج محتملة: {unique_conditions}."
            return f"Detected {len(findings_payload)} potential findings: {unique_conditions}."

        if classifier_prediction:
            predicted_label = classifier_prediction.get("predicted_label", "unknown")
            confidence = float(classifier_prediction.get("confidence", 0.0))
            if confidence >= cls.CLASSIFIER_SUPPORT_THRESHOLD:
                if arabic_mode:
                    return (
                        "لم ترصد قواعد التحليل نتائج غير طبيعية، "
                        f"لكن تصنيف الذكاء الاصطناعي يشير إلى {predicted_label} "
                        f"(الثقة {confidence:.2f})."
                    )
                return (
                    "Rule engine found no abnormal findings, but AI classification suggests "
                    f"{predicted_label} "
                    f"(confidence {confidence:.2f})."
                )

        return "لم يتم اكتشاف نتائج ذات دلالة سريرية مهمة." if arabic_mode else "No clinically significant findings detected."

    async def diagnose(self, report: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")

        response_language = detect_preferred_language(
            report.get("raw_text_original"),
            report.get("raw_text"),
            report.get("follow_up_answers"),
            report.get("sections"),
            report.get("symptoms"),
            default="en",
        )

        labs = report.get("labs", {}) or {}
        symptoms = report.get("symptoms", []) or []
        findings = diagnose_from_labs(labs)
        symptom_findings = diagnose_from_symptoms(
            symptoms,
            raw_text=str(report.get("raw_text", "") or ""),
        ) if symptoms else []
        merged_findings = [
            ("lab_rules", finding) for finding in findings
        ] + [
            ("symptom_rules", finding) for finding in symptom_findings
        ]
        findings_payload = [
            {
                "condition": finding.condition,
                "confidence": finding.confidence,
                "evidence": finding.evidence,
                "severity": finding.severity,
                "source": source,
            }
            for source, finding in merged_findings
        ]
        result = {
            "findings": findings_payload,
            "disclaimer": self.DISCLAIMER,
            "response_language": response_language,
        }

        combined = build_combined_text(report)
        patient_symptoms = [str(item).strip().lower() for item in symptoms if str(item).strip()]
        rag_out: Optional[Dict[str, Any]] = None
        classifier_prediction: Optional[Dict[str, Any]] = None

        if self._rag_assistant:
            rag_out = await self._rag_assistant.query(
                combined,
                top_k=self._rag_top_k,
                query_symptoms=patient_symptoms,
            )
            result["rag_response"] = rag_out["response"]
            result["retrieved_cases"] = rag_out["retrieved_cases"]
            result["rag_metadata"] = {
                "query_text": rag_out.get("rag_query_text", combined),
                "mode": rag_out.get("rag_mode", "retrieval_only"),
                "confidence": rag_out.get("rag_confidence"),
                "rag_scope_status": rag_out.get("rag_scope_status"),
                "usable_for_fusion": (rag_out.get("rag_confidence") or {}).get("usable_for_fusion"),
                "detected_out_of_scope_signals": rag_out.get("detected_out_of_scope_signals", []),
            }
            if "structured_diagnosis" in rag_out:
                result["structured_rag_diagnosis"] = rag_out["structured_diagnosis"]

        if self._classifier:
            classifier_input_text = str(report.get("raw_text", "") or "").strip() or combined
            classifier_query_text = classifier_input_text
            translated_from_arabic = False
            if (
                self._classifier_translate_arabic
                and self._classifier_translator
                and self._classifier_translator.is_arabic(classifier_input_text)
            ):
                classifier_query_text = await self._classifier_translator.translate(classifier_input_text)
                translated_from_arabic = classifier_query_text != classifier_input_text
            classifier_prediction = self._classifier.predict(classifier_query_text)
            classifier_prediction["input_text"] = classifier_input_text
            classifier_prediction["query_text"] = classifier_query_text
            classifier_prediction["translated_from_arabic"] = translated_from_arabic
            result["classifier_prediction"] = classifier_prediction

        final_diagnosis = self._build_final_diagnosis(
            findings=findings_payload,
            patient_symptoms=patient_symptoms,
            report_text=combined,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
        )
        if final_diagnosis:
            result["final_diagnosis"] = final_diagnosis

        candidates = self._collect_diagnostic_candidates(
            findings=findings_payload,
            patient_symptoms=patient_symptoms,
            report_text=combined,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
        )
        if candidates:
            result["diagnostic_candidates"] = [
                {
                    "label": candidate["label"],
                    "confidence": round(candidate["confidence"], 2),
                    "sources": candidate["sources"],
                }
                for candidate in candidates[:5]
            ]

        result = self._apply_label_canonicalization(result)
        final_diagnosis = result.get("final_diagnosis", final_diagnosis)

        clarification = self._build_clarification(
            report=report,
            findings=findings_payload,
            patient_symptoms=patient_symptoms,
            candidates=candidates,
            final_diagnosis=final_diagnosis,
        )
        if clarification:
            result["clarification"] = clarification

        result["summary"] = self._build_summary(
            findings_payload,
            final_diagnosis=final_diagnosis,
            classifier_prediction=classifier_prediction,
            clarification=clarification,
            response_language=response_language,
        )

        result["decision_fusion"] = self._build_decision_fusion(
            findings_payload,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
            final_diagnosis=final_diagnosis,
        )
        result["safety"] = self._build_safety(
            findings_payload,
            response_language=response_language,
        )

        if self._response_synthesizer:
            synthesis = await self._response_synthesizer.synthesize(
                report,
                result,
                response_language=response_language,
            )
            result["ai_response"] = synthesis["response_text"]
            result["ai_response_metadata"] = synthesis["metadata"]
            result["gemini_response"] = synthesis["response_text"]
            result["gemini_response_metadata"] = synthesis["metadata"]
            if synthesis.get("structured_response") is not None:
                result["structured_ai_response"] = synthesis["structured_response"]
                result["structured_gemini_response"] = synthesis["structured_response"]

        return result


async def diagnose(report: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return await DiagnosisEngine(**kwargs).diagnose(report)
