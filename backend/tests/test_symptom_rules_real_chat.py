from models.diagnosis.rules import diagnose_from_symptoms


def _conditions(findings):
    return {item.condition for item in findings}


def test_real_chat_anaphylaxis_rule():
    findings = diagnose_from_symptoms(
        ["dizziness", "wheezing", "flushing"],
        raw_text="Right after eating peanuts my throat started to tighten.",
    )

    assert "Anaphylaxis" in _conditions(findings)


def test_real_chat_panic_rule():
    findings = diagnose_from_symptoms(
        ["palpitations", "tingling"],
        raw_text="I suddenly felt choking and scared I might die.",
    )

    assert "Panic attack" in _conditions(findings)


def test_real_chat_neoplasm_and_neuro_rules():
    pancreatic = diagnose_from_symptoms(
        ["abdominal pain", "loss of appetite", "weight loss"],
        raw_text="Upper stomach pain, poor appetite, and losing weight over the past months.",
    )
    myasthenia = diagnose_from_symptoms(
        ["ptosis", "double vision"],
        raw_text="My eyelids droop later in the day and I get double vision.",
    )

    assert "Pancreatic neoplasm" in _conditions(pancreatic)
    assert "Myasthenia gravis" in _conditions(myasthenia)
