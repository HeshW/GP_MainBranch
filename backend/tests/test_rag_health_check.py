from pathlib import Path
from types import SimpleNamespace

from scripts import rag_health_check


class _FakeIndex:
    ntotal = 2
    nprobe = 16


class _FakeSearcher:
    def __init__(self, index_dir, allow_unsafe_pickle=False):
        self.index = _FakeIndex()
        self.metadata = {
            "patient_ids": ["p1", "p2"],
            "pathologies": ["Stable angina", "Bronchospasm / acute asthma exacerbation"],
        }

    def search(self, embedding, k=5, query_text="", query_symptoms=None):
        if "glucose" in query_text.lower() or "thirst" in query_text.lower():
            return [
                {
                    "pathology": "Pulmonary neoplasm",
                    "rerank_score": 0.2,
                    "symptom_overlap": 0.2,
                    "feature_alignment": 0.2,
                    "lab_match": 0.0,
                    "disease_family_hint": 0.0,
                    "mismatch_penalty": 0.0,
                    "pathology_penalty": 0.0,
                    "clinical_context_penalty": 0.36,
                }
            ]
        return [
            {
                "pathology": "Stable angina",
                "rerank_score": 0.7,
                "symptom_overlap": 0.7,
                "feature_alignment": 0.7,
                "lab_match": 0.0,
                "disease_family_hint": 1.0,
                "mismatch_penalty": 0.0,
                "pathology_penalty": 0.0,
                "clinical_context_penalty": 0.0,
            }
        ]

    @staticmethod
    def detect_out_of_scope_signals(query_text, query_symptoms=None):
        if "glucose" in query_text.lower() or "thirst" in query_text.lower():
            return ["diabetes_hyperglycemia"]
        return []


class _FakeEmbedder:
    model_source = "fake-model"

    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    def encode_text(self, text):
        return object()


def test_rag_health_check_payload_structure(monkeypatch):
    monkeypatch.setattr(rag_health_check, "MedicalCaseSearcher", _FakeSearcher)
    monkeypatch.setattr(rag_health_check, "ClinicalBERTEmbedder", _FakeEmbedder)
    monkeypatch.setattr(
        rag_health_check,
        "get_settings",
        lambda: SimpleNamespace(
            faiss_index_dir="fake-faiss",
            clinicalbert_model_dir="fake-clinicalbert",
            allow_unsafe_pickle_metadata=False,
            use_finetuned_classifier=False,
            finetuned_model_dir=None,
            classifier_max_length=256,
        ),
    )

    payload = rag_health_check.build_health_payload(
        SimpleNamespace(
            faiss_index_dir=Path("fake-faiss"),
            clinicalbert_model_dir=Path("fake-clinicalbert"),
            top_k=5,
        )
    )

    assert payload["status"] == "ok"
    assert payload["index_loaded"] is True
    assert payload["metadata_loaded"] is True
    assert payload["faiss_vector_count"] == 2
    assert payload["metadata_row_count"] == 2
    assert payload["unique_pathologies"] == 2
    assert payload["faiss_nprobe"] == 16
    assert payload["clinicalbert_model_loaded"] is True
    assert payload["classifier"]["enabled"] is False
    assert payload["in_scope_smoke_query"]["top_pathology"] == "Stable angina"
    assert payload["out_of_scope_safety_query"]["rag_confidence"]["usable_for_fusion"] is False
