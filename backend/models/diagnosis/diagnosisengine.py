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

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        clinicalbert_model_dir: Optional[Path | str] = None,
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
                searcher=MedicalCaseSearcher(Path(faiss_index_dir)),
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
    def _normalize_label(cls, value: str) -> str:
        return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())

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
        rag_structured = (rag_out or {}).get("structured_diagnosis") or {}
        rag_findings = rag_structured.get("findings") or []
        rag_summary = str(rag_structured.get("assessment_summary", "")).strip()

        candidates: list[Dict[str, Any]] = []

        if classifier_prediction:
            label = str(classifier_prediction.get("predicted_label", "")).strip()
            confidence = cls._normalize_confidence(classifier_prediction.get("confidence"))
            if label and not cls._is_symptom_like_label(label, patient_symptoms):
                candidates.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "source": "classifier",
                        "reasoning": f"Fine-tuned classifier prediction with confidence {confidence:.2f}.",
                        "evidence": [
                            f"Classifier top label: {label}",
                        ],
                    }
                )

        if rag_findings:
            rag_diagnosis_findings = [
                item for item in rag_findings
                if not cls._is_symptom_like_label(str(item.get("condition", "")).strip(), patient_symptoms)
            ]
            if rag_diagnosis_findings:
                rag_best = max(
                    rag_diagnosis_findings,
                    key=lambda item: cls._normalize_confidence(item.get("confidence")),
                )
                rag_label = str(rag_best.get("condition", "")).strip()
                rag_confidence = cls._normalize_confidence(rag_best.get("confidence"))
                candidates.append(
                    {
                        "label": rag_label,
                        "confidence": rag_confidence,
                        "source": "rag",
                        "reasoning": rag_summary or "RAG structured assessment based on retrieved clinical cases.",
                        "evidence": [
                            str(rag_best.get("evidence", "")).strip() or "RAG finding extracted from similar cases.",
                        ],
                    }
                )

        if rag_out and rag_out.get("retrieved_cases"):
            top_case = rag_out["retrieved_cases"][0]
            rag_label = str(top_case.get("pathology", "")).strip()
            rag_confidence = max(0.3, min(float(top_case.get("similarity", 0.0)), 0.8))
            if rag_label and not cls._is_symptom_like_label(rag_label, patient_symptoms):
                candidates.append(
                    {
                        "label": rag_label,
                        "confidence": rag_confidence,
                        "source": "rag_retrieval",
                        "reasoning": "Nearest-neighbor retrieval from similar indexed medical cases.",
                        "evidence": [
                            f"Top retrieved case pathology: {rag_label}",
                        ],
                    }
                )

        classifier_label = (
            str(classifier_prediction.get("predicted_label", "")).strip()
            if classifier_prediction
            else ""
        )
        rag_label = ""
        for candidate in candidates:
            if candidate["source"] in {"rag", "rag_retrieval"}:
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
                        for candidate in candidates
                        if candidate["source"] in {"rag", "rag_retrieval"}
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

        if candidates:
            ranked_candidates = sorted(
                candidates,
                key=lambda item: (
                    item["confidence"] + (0.03 if item["source"] == "rag" else 0.0),
                    item["source"] == "rag",
                ),
                reverse=True,
            )
            selected = ranked_candidates[0]
            if selected["source"] == "classifier" and not classifier_primary:
                rag_candidate = next(
                    (item for item in ranked_candidates if item["source"] in {"rag", "rag_retrieval"}),
                    None,
                )
                if rag_candidate is not None:
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
                "source": selected["source"],
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
    ) -> str:
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

        final_diagnosis = self._build_final_diagnosis(
            findings=findings_payload,
            patient_symptoms=patient_symptoms,
            rag_out=rag_out,
            classifier_prediction=classifier_prediction,
        )
        if final_diagnosis:
            result["final_diagnosis"] = final_diagnosis

        result["summary"] = self._build_summary(
            findings_payload,
            final_diagnosis=final_diagnosis,
            classifier_prediction=classifier_prediction,
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
