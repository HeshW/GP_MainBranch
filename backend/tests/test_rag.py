import hashlib
import importlib
import json
import pickle

import pytest

from models.diagnosis.rag import MedicalCaseSearcher


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
