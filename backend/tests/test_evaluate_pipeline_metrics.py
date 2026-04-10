from scripts.evaluate_pipeline_end_to_end import clinically_matches, is_in_supported_scope


def test_clinically_matches_diabetes_family_rule_output():
    assert clinically_matches(
        "Diabetes Mellitus (suspected)",
        "Possible hyperglycemia / diabetes symptom pattern",
    )


def test_clinically_matches_anemia_family_variants():
    assert clinically_matches("Anemia", "Moderate Anemia")


def test_is_in_supported_scope_uses_normalized_labels():
    supported = {"Viral pharyngitis", "GERD"}
    assert is_in_supported_scope("viral pharyngitis", supported)
    assert not is_in_supported_scope("Diabetes Mellitus (suspected)", supported)
