from __future__ import annotations

from inspect import isawaitable

from manager.diagnosis_adapter import adapt_and_diagnose


def test_adapt_and_diagnose_returns_concrete_result(monkeypatch):
    async def fake_diagnose(report, **kwargs):
        return {"status": "ok", "report": report, "kwargs": kwargs}

    monkeypatch.setattr("manager.diagnosis_adapter.diagnose", fake_diagnose)

    result = adapt_and_diagnose({"raw_text": "fatigue"})

    assert result["status"] == "ok"
    assert result["report"]["raw_text"] == "fatigue"
    assert not isawaitable(result)


def test_adapt_and_diagnose_forwards_runtime_flags(monkeypatch):
    async def fake_diagnose(report, **kwargs):
        return {"report": report, "kwargs": kwargs}

    monkeypatch.setattr("manager.diagnosis_adapter.diagnose", fake_diagnose)

    result = adapt_and_diagnose(
        {"raw_text": "test"},
        use_rag=True,
        rag_top_k=9,
        classifier_translate_arabic=False,
    )

    assert result["kwargs"]["use_rag"] is True
    assert result["kwargs"]["rag_top_k"] == 9
    assert result["kwargs"]["classifier_translate_arabic"] is False
