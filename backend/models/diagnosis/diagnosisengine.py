"""Production diagnosis engine orchestrating AI-first diagnosis with rule safety checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import logging

from models.common.ai_provider import GeminiProvider
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
    CLASSIFIER_OVERRIDE_MARGIN = 0.08
    MAX_CLARIFICATION_QUESTIONS = 3
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

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        clinicalbert_model_dir: Optional[Path | str] = None,
        allow_unsafe_pickle_metadata: bool = False,
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
                gemini_api_key=gemini_api_key,
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

        if gemini_api_key:
            self._response_synthesizer = DiagnosisResponseSynthesizer(gemini_api_key=gemini_api_key)

    @staticmethod
    def _build_safety(findings: list[Dict[str, Any]]) -> Dict[str, Any]:
        severity_order = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "info": 0}
        highest_severity = "info"
        reasons: list[str] = []

        for finding in findings:
            severity = str(finding.get("severity", "info")).lower()
            if severity_order.get(severity, 0) > severity_order.get(highest_severity, 0):
                highest_severity = severity
            if severity in {"critical", "high"}:
                reasons.append(
                    f"{finding.get('condition', 'Unknown finding')} marked as {severity} severity."
                )

        clinician_review_required = bool(findings)
        emergency_attention_recommended = highest_severity == "critical"

        if not findings:
            reasons.append("No abnormal findings were detected by the rule engine.")
        elif clinician_review_required and not reasons:
            reasons.append("Clinical review is recommended for any abnormal finding.")

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
        rag_out: Optional[Dict[str, Any]] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        rule_conditions = [str(item.get("condition", "")).strip() for item in findings if item.get("condition")]
        rag_structured = (rag_out or {}).get("structured_diagnosis") or {}
        rag_findings = rag_structured.get("findings") or []
        rag_summary = str(rag_structured.get("assessment_summary", "")).strip()

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
        for case in (rag_out or {}).get("retrieved_cases", []) or []:
            label = str(case.get("pathology", "")).strip()
            if not label or cls._is_symptom_like_label(label, patient_symptoms):
                continue
            normalized_label = cls._normalize_label(label)
            if normalized_label in seen_rag_labels:
                continue
            seen_rag_labels.add(normalized_label)
            similarity = max(0.3, min(float(case.get("similarity", 0.0)), 0.8))
            cls._merge_candidate(
                merged,
                label=label,
                confidence=similarity,
                source="rag_retrieval",
                reasoning="Nearest-neighbor retrieval from similar indexed medical cases.",
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

        candidates = sorted(
            merged.values(),
            key=lambda item: (
                item["confidence"] + (0.03 if "classifier" in item["sources"] else 0.0),
                "classifier" in item["sources"],
                "rag" in item["sources"] or "rag_retrieval" in item["sources"],
            ),
            reverse=True,
        )

        for candidate in candidates:
            if rule_conditions:
                candidate["rule_alignment"] = any(
                    cls._labels_overlap(candidate["label"], condition)
                    for condition in rule_conditions
                )
            else:
                candidate["rule_alignment"] = False
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
            if final_confidence < cls.CLARIFICATION_CONFIDENCE_THRESHOLD:
                reasons.append("Current diagnosis confidence is below the clarification threshold.")
            if findings and not final_diagnosis.get("rule_alignment"):
                reasons.append("Rule-based safety signals do not clearly align with the current AI diagnosis.")

        if len(candidates) >= 2:
            top_conf = candidates[0]["confidence"]
            second_conf = candidates[1]["confidence"]
            if (
                cls._normalize_label(candidates[0]["label"]) != cls._normalize_label(candidates[1]["label"])
                and abs(top_conf - second_conf) <= cls.CLARIFICATION_MARGIN_THRESHOLD
            ):
                reasons.append("Top candidate diseases are close in score and need discrimination.")

        if len(patient_symptoms) < 2 and not findings:
            reasons.append("The first-turn symptom summary is sparse, so more discriminative details are needed.")
        return reasons

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
        candidate_labels = [candidate["label"] for candidate in candidates[:3]]

        for candidate in candidates[:3]:
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
                if len(questions) >= cls.MAX_CLARIFICATION_QUESTIONS:
                    break
            if len(questions) >= cls.MAX_CLARIFICATION_QUESTIONS:
                break

        if len(questions) < cls.MAX_CLARIFICATION_QUESTIONS:
            generic_candidates = ", ".join(candidate_labels[:2]) if candidate_labels else "the current differential diagnosis"
            generic_questions = [
                "هل بدأت الأعراض فجأة أم تدريجياً؟ وهل تزداد سوءاً؟" if arabic_mode else "Did the symptoms start suddenly or gradually, and are they getting worse?",
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
                if len(questions) >= cls.MAX_CLARIFICATION_QUESTIONS:
                    break

        return {
            "needed": True,
            "mode": "follow_up_questions",
            "reasons": reasons,
            "questions": questions[: cls.MAX_CLARIFICATION_QUESTIONS],
            "candidate_diseases": [
                {
                    "label": candidate["label"],
                    "confidence": round(candidate["confidence"], 2),
                    "sources": candidate["sources"],
                }
                for candidate in candidates[:3]
            ],
        }

    @classmethod
    def _signal_match_score(cls, label: str, answer_text: str) -> float:
        normalized_label = cls._normalize_label(label)
        lowered_answer = answer_text.lower()
        score = 0.0
        if normalized_label and normalized_label in cls._normalize_label(answer_text):
            score += 0.35
        for template in cls.FOLLOW_UP_QUESTION_BANK:
            if not any(keyword in normalized_label for keyword in template["keywords"]):
                continue
            matched_signals = [signal for signal in template["signals"] if signal in lowered_answer]
            score += min(0.3, 0.1 * len(matched_signals))
        return score

    @classmethod
    def apply_follow_up_scoring(
        cls,
        diagnosis: Dict[str, Any],
        *,
        answers: list[str],
        prior_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(diagnosis, dict):
            return diagnosis

        rescored = dict(diagnosis)
        normalized_answers = [str(item).strip() for item in answers if str(item).strip()]
        if not normalized_answers:
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

        question_targets: Dict[str, float] = {}
        for idx, question in enumerate(prior_clarification.get("questions", []) or []):
            answer = normalized_answers[idx] if idx < len(normalized_answers) else ""
            if not answer:
                continue
            for label in question.get("target_conditions", []) or []:
                normalized_label = cls._normalize_label(label)
                question_targets[normalized_label] = question_targets.get(normalized_label, 0.0) + 0.08

        rescored_candidates = []
        answers_blob = " ".join(normalized_answers)
        for normalized_label, candidate in candidate_map.items():
            adjusted = float(candidate["confidence"])
            adjusted += question_targets.get(normalized_label, 0.0)
            adjusted += cls._signal_match_score(candidate["label"], answers_blob)
            rescored_candidates.append(
                {
                    "label": candidate["label"],
                    "confidence": round(min(adjusted, 0.99), 2),
                    "sources": candidate["sources"],
                }
            )

        rescored_candidates.sort(key=lambda item: item["confidence"], reverse=True)
        rescored["diagnostic_candidates"] = rescored_candidates

        if rescored_candidates:
            best = rescored_candidates[0]
            final = dict(rescored.get("final_diagnosis", {}) or {})
            previous_label = str(final.get("diagnosis", "")).strip()
            previous_confidence = cls._normalize_confidence(final.get("confidence"))
            should_override = (
                not previous_label
                or cls._normalize_label(previous_label) != cls._normalize_label(best["label"])
                or best["confidence"] > previous_confidence
            )
            if should_override:
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
            clarification["applied"] = True
            clarification["answers_used"] = normalized_answers
            rescored["clarification"] = clarification
        return rescored

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
        rag_out: Optional[Dict[str, Any]] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        rule_conditions = [str(item.get("condition", "")) for item in findings if item.get("condition")]
        candidates = cls._collect_diagnostic_candidates(
            findings=findings,
            patient_symptoms=patient_symptoms,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
        )
        ai_candidates = [
            candidate
            for candidate in candidates
            if any(source in {"classifier", "rag", "rag_retrieval"} for source in candidate["sources"])
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
                "rule_alignment": any(
                    cls._labels_overlap(classifier_label, condition)
                    for condition in rule_conditions
                ),
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
                    "classifier" in item["sources"],
                    item["confidence"],
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
                selected = classifier_candidate
                if rag_candidate and not cls._labels_overlap(classifier_candidate["label"], rag_candidate["label"]):
                    rag_advantage = rag_candidate["confidence"] - classifier_candidate["confidence"]
                    if rag_advantage >= cls.CLASSIFIER_OVERRIDE_MARGIN * 1.5:
                        selected = rag_candidate
            elif "classifier" in selected["sources"] and not classifier_primary and rag_candidate is not None:
                selected = rag_candidate

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
                "rule_alignment": any(
                    cls._labels_overlap(selected["label"], condition)
                    for condition in rule_conditions
                ),
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
    ) -> str:
        if clarification and clarification.get("needed"):
            candidate_labels = [item.get("label", "") for item in clarification.get("candidate_diseases", [])]
            if candidate_labels:
                return (
                    "The first-pass assessment is still uncertain. "
                    f"Current leading possibilities are {', '.join(candidate_labels[:3])}. "
                    "Answering the follow-up questions will help refine the diagnosis."
                )
            return (
                "The first-pass assessment is still uncertain. "
                "Follow-up questions are needed before making a stronger diagnostic claim."
            )

        if final_diagnosis:
            diagnosis = final_diagnosis.get("diagnosis", "an undetermined condition")
            confidence = cls._normalize_confidence(final_diagnosis.get("confidence"))
            source = str(final_diagnosis.get("source", "ai"))
            if findings_payload:
                return (
                    f"AI-assisted assessment suggests {diagnosis} "
                    f"(confidence {confidence:.2f}, source: {source}) with rule-based safety checks attached."
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
                return (
                    "No abnormal lab-rule findings were detected, but symptom-based rules suggest: "
                    f"{unique_conditions}."
                )
            return f"Detected {len(findings_payload)} potential findings: {unique_conditions}."

        if classifier_prediction:
            predicted_label = classifier_prediction.get("predicted_label", "unknown")
            confidence = float(classifier_prediction.get("confidence", 0.0))
            if confidence >= cls.CLASSIFIER_SUPPORT_THRESHOLD:
                return (
                    "Rule engine found no abnormal findings, but AI classification suggests "
                    f"{predicted_label} "
                    f"(confidence {confidence:.2f})."
                )

        return "No clinically significant findings detected."

    async def diagnose(self, report: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")

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
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
        )
        if final_diagnosis:
            result["final_diagnosis"] = final_diagnosis

        candidates = self._collect_diagnostic_candidates(
            findings=findings_payload,
            patient_symptoms=patient_symptoms,
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
        )

        result["decision_fusion"] = self._build_decision_fusion(
            findings_payload,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
            final_diagnosis=final_diagnosis,
        )
        result["safety"] = self._build_safety(findings_payload)

        if self._response_synthesizer:
            synthesis = await self._response_synthesizer.synthesize(report, result)
            result["gemini_response"] = synthesis["response_text"]
            result["gemini_response_metadata"] = synthesis["metadata"]
            if synthesis.get("structured_response") is not None:
                result["structured_gemini_response"] = synthesis["structured_response"]

        return result


async def diagnose(report: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return await DiagnosisEngine(**kwargs).diagnose(report)
