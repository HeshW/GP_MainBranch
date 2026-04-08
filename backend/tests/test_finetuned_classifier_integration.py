import asyncio

import models.diagnosis.diagnosisengine as diagnosisengine


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
