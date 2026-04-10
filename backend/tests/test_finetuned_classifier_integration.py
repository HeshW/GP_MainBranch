import asyncio
import json
import shutil
from pathlib import Path

import models.diagnosis.diagnosisengine as diagnosisengine
from models.diagnosis.rag import FineTunedDiagnosisClassifier


def test_finetuned_classifier_prediction_is_included(monkeypatch):
    class StubClassifier:
        def __init__(self, model_dir, max_length=256, device=None):
            self.model_dir = model_dir
            self.max_length = max_length

        def predict(self, text):
            assert isinstance(text, str)
            assert text
            return {
                "predicted_label": "Influenza",
                "confidence": 0.91,
                "top_predictions": [
                    {"label": "Influenza", "confidence": 0.91},
                    {"label": "Bronchitis", "confidence": 0.06},
                ],
            }

    monkeypatch.setattr(diagnosisengine, "FineTunedDiagnosisClassifier", StubClassifier)

    engine = diagnosisengine.DiagnosisEngine(
        use_finetuned_classifier=True,
        finetuned_model_dir="fake-model-dir",
    )

    result = asyncio.run(
        engine.diagnose(
            {
                "raw_text": "Patient with fever, cough, and sore throat",
                "labs": {},
            }
        )
    )

    assert "classifier_prediction" in result
    assert result["classifier_prediction"]["predicted_label"] == "Influenza"
    assert result["classifier_prediction"]["confidence"] == 0.91
    assert result["final_diagnosis"]["diagnosis"] == "Influenza"
    assert result["decision_fusion"]["primary_source"] == "classifier"


def test_finetuned_classifier_translates_arabic_before_prediction(monkeypatch):
    class StubClassifier:
        def __init__(self, model_dir, max_length=256, device=None):
            pass

        def predict(self, text):
            assert text == "fever and cough"
            return {
                "predicted_label": "Influenza",
                "confidence": 0.88,
                "top_predictions": [{"label": "Influenza", "confidence": 0.88}],
            }

    class StubTranslator:
        def __init__(self, provider):
            pass

        @staticmethod
        def is_arabic(text):
            return True

        async def translate(self, arabic_text):
            assert arabic_text == "عندي حرارة وكحة"
            return "fever and cough"

    class StubProvider:
        def __init__(self, api_key, model_name="gemini-2.5-flash"):
            pass

    monkeypatch.setattr(diagnosisengine, "FineTunedDiagnosisClassifier", StubClassifier)
    monkeypatch.setattr(diagnosisengine, "ArabicToEnglishTranslator", StubTranslator)
    monkeypatch.setattr(diagnosisengine, "GeminiProvider", StubProvider)

    engine = diagnosisengine.DiagnosisEngine(
        use_finetuned_classifier=True,
        finetuned_model_dir="fake-model-dir",
        gemini_api_key="fake-gemini-key",
        classifier_translate_arabic=True,
    )

    result = asyncio.run(engine.diagnose({"raw_text": "عندي حرارة وكحة", "labs": {}}))

    assert result["classifier_prediction"]["predicted_label"] == "Influenza"
    assert result["classifier_prediction"]["query_text"] == "fever and cough"
    assert result["classifier_prediction"]["translated_from_arabic"] is True
    assert result["final_diagnosis"]["diagnosis"] == "Influenza"


def test_finetuned_classifier_accepts_id2label_format(monkeypatch):
    temp_dir = Path(__file__).resolve().parent / "_classifier_map_test"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)
    model_dir = temp_dir / "model"
    model_dir.mkdir(exist_ok=True)
    try:
        (model_dir / "label_map.json").write_text(
            json.dumps(
                {
                    "id2label": {"0": "Influenza"},
                    "label2id": {"Influenza": 0},
                }
            ),
            encoding="utf-8",
        )

        class DummyTensor:
            def to(self, device):
                return self

        class DummyInputs(dict):
            def to(self, device):
                return self

        class DummyTokenizer:
            @classmethod
            def from_pretrained(cls, model_dir):
                return cls()

            def __call__(self, text, truncation=True, padding=True, max_length=256, return_tensors="pt"):
                return DummyInputs({"input_ids": DummyTensor(), "attention_mask": DummyTensor()})

        class DummyLogits:
            shape = [1]

            def __getitem__(self, idx):
                return self

        class DummyOutput:
            logits = DummyLogits()

        class DummyModel:
            @classmethod
            def from_pretrained(cls, model_dir):
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **inputs):
                return DummyOutput()

        class DummyTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return False

            @staticmethod
            def device(name):
                return name

            @staticmethod
            def no_grad():
                class _NoGrad:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                return _NoGrad()

            @staticmethod
            def softmax(logits, dim=0):
                class _Prob:
                    shape = [1]

                    def __getitem__(self, idx):
                        class _Score:
                            def item(self):
                                return 0.95

                        return _Score()

                return _Prob()

            @staticmethod
            def argmax(probs):
                class _Idx:
                    def item(self):
                        return 0

                return _Idx()

            @staticmethod
            def topk(probs, k=1):
                class _Score:
                    def item(self):
                        return 0.95

                class _Idx:
                    def item(self):
                        return 0

                return [_Score()], [_Idx()]

        class DummyTransformers:
            AutoTokenizer = DummyTokenizer
            AutoModelForSequenceClassification = DummyModel

        def fake_import_module(name):
            if name == "torch":
                return DummyTorch
            if name == "transformers":
                return DummyTransformers
            raise ImportError(name)

        monkeypatch.setattr("importlib.import_module", fake_import_module)

        classifier = FineTunedDiagnosisClassifier(model_dir=model_dir)
        prediction = classifier.predict("fever and cough")

        assert classifier.id_to_label[0] == "Influenza"
        assert prediction["predicted_label"] == "Influenza"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
