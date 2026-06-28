import hashlib
import importlib
import json
import pickle

import pytest

from models.diagnosis.rag import MedicalCaseSearcher, MedicalRAGAssistant
from scripts.evaluate_rag_retrieval_quality import evaluate_thresholds


def test_rerank_results_prefers_feature_aligned_case():
    results = [
        {
            "similarity": 0.74,
            "pathology": "Myocarditis",
            "case_text": (
                "Patient: 34 year old M. Presenting symptoms: chest pain, shortness of breath, "
                "viral infection, fatigue."
            ),
        },
        {
            "similarity": 0.61,
            "pathology": "Diabetes",
            "case_text": (
                "Patient: 50 year old F. Presenting symptoms: fatigue, thirst, polyuria, "
                "weight loss."
            ),
        },
    ]

    reranked = MedicalCaseSearcher._rerank_results(
        results,
        query_text="Fatigue and increased thirst for two weeks.",
        query_symptoms=["fatigue", "thirst"],
    )

    assert reranked[0]["pathology"] == "Diabetes"


def test_feature_mismatch_penalty_detects_unrequested_cardiac_features():
    query_features = MedicalCaseSearcher._extract_feature_flags(
        "Fatigue and increased thirst for two weeks.",
        ["fatigue", "thirst"],
    )

    penalty = MedicalCaseSearcher._feature_mismatch_penalty(
        query_features,
        "Patient reports chest pain, shortness of breath, and recent viral infection with fatigue.",
    )

    assert penalty > 0


class _FakeIndex:
    ntotal = 3
    nlist = 32
    nprobe = 1


class _FakeFaiss:
    @staticmethod
    def read_index(path):
        return _FakeIndex()


def _patch_fake_faiss(monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "faiss":
            return _FakeFaiss()
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def test_medical_case_searcher_prefers_json_metadata(monkeypatch, tmp_path):
    _patch_fake_faiss(monkeypatch)

    (tmp_path / "metadata_mapping.json").write_text(
        json.dumps({"patient_ids": ["p1"], "pathologies": ["Dx"], "symptoms": ["cough"]}),
        encoding="utf-8",
    )

    searcher = MedicalCaseSearcher(tmp_path)

    assert searcher.metadata["patient_ids"] == ["p1"]
    assert searcher.index.nprobe == 16


def test_medical_case_searcher_rejects_pickle_without_hash(monkeypatch, tmp_path):
    _patch_fake_faiss(monkeypatch)

    payload = {"patient_ids": ["p1"], "pathologies": ["Dx"], "symptoms": ["cough"]}
    with (tmp_path / "metadata_mapping.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="without hash verification"):
        MedicalCaseSearcher(tmp_path)


def test_medical_case_searcher_accepts_verified_pickle(monkeypatch, tmp_path):
    _patch_fake_faiss(monkeypatch)

    payload = {"patient_ids": ["p2"], "pathologies": ["Dx2"], "symptoms": ["fever"]}
    pickle_path = tmp_path / "metadata_mapping.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle)

    digest = hashlib.sha256(pickle_path.read_bytes()).hexdigest()
    (tmp_path / "metadata_mapping.pkl.sha256").write_text(digest, encoding="utf-8")

    searcher = MedicalCaseSearcher(tmp_path)

    assert searcher.metadata["patient_ids"] == ["p2"]


def test_medical_case_searcher_allows_explicit_unsafe_pickle_override(monkeypatch, tmp_path):
    _patch_fake_faiss(monkeypatch)

    payload = {"patient_ids": ["p3"], "pathologies": ["Dx3"], "symptoms": ["fatigue"]}
    with (tmp_path / "metadata_mapping.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    searcher = MedicalCaseSearcher(tmp_path, allow_unsafe_pickle=True)

    assert searcher.metadata["patient_ids"] == ["p3"]


def test_rerank_results_adds_debug_scores_and_prefers_family_hint():
    results = [
        {
            "similarity": 0.35,
            "pathology": "Acute COPD exacerbation / infection",
            "case_text": "Patient: 70 year old M. Presenting symptoms: shortness of breath, wheezing, productive cough.",
        },
        {
            "similarity": 0.31,
            "pathology": "Bronchospasm / acute asthma exacerbation",
            "case_text": "Patient: 24 year old F. Presenting symptoms: shortness of breath, wheezing, chest tightness.",
        },
    ]

    reranked = MedicalCaseSearcher._rerank_results(
        results,
        query_text=(
            "age: 24\nsex: F\nclinical_context: shortness of breath with wheezing "
            "and chest tightness without fever\nnegative_symptoms: fever"
        ),
        query_symptoms=["shortness of breath", "wheezing", "chest tightness"],
    )

    assert reranked[0]["pathology"] == "Bronchospasm / acute asthma exacerbation"
    assert "rerank_score" in reranked[0]
    assert "symptom_overlap" in reranked[0]
    assert reranked[0]["disease_family_hint"] > 0


def test_out_of_scope_signals_disable_rag_fusion():
    signals = MedicalCaseSearcher.detect_out_of_scope_signals(
        "age: 50\nclinical_context: fatigue, thirst, polyuria, fasting glucose is elevated",
        ["fatigue", "thirst", "polyuria"],
    )
    confidence = MedicalRAGAssistant._build_confidence_metadata(
        [
            {
                "pathology": "Pulmonary neoplasm",
                "rerank_score": 0.62,
                "symptom_overlap": 0.5,
                "feature_alignment": 0.5,
                "lab_match": 0.0,
                "disease_family_hint": 0.0,
                "mismatch_penalty": 0.0,
                "pathology_penalty": 0.0,
                "clinical_context_penalty": 0.36,
            }
        ],
        detected_out_of_scope_signals=signals,
    )

    assert signals == ["diabetes_hyperglycemia"]
    assert confidence["scope_status"] == "out_of_scope_or_low_confidence"
    assert confidence["usable_for_fusion"] is False
    assert confidence["detected_out_of_scope_signals"] == ["diabetes_hyperglycemia"]


def test_evaluation_thresholds_are_optional_and_explicit():
    passing = evaluate_thresholds(
        {
            "top_5_accuracy": 0.9,
            "mrr": 0.75,
            "out_of_scope_low_confidence_rate": 1.0,
        }
    )
    failing = evaluate_thresholds(
        {
            "top_5_accuracy": 0.7,
            "mrr": 0.75,
            "out_of_scope_low_confidence_rate": 1.0,
        }
    )

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["checks"]["top_5_accuracy"]["passed"] is False
