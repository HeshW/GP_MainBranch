from scripts.build_discussion_evaluation import (
    build_label_metrics,
    classifier_confusion_pairs,
    retraining_recommendation,
)


def test_build_label_metrics_computes_macro_and_weighted_scores():
    rows = [
        {"true_label": "A", "predicted_label": "A"},
        {"true_label": "A", "predicted_label": "B"},
        {"true_label": "B", "predicted_label": "B"},
    ]

    report_rows, averages = build_label_metrics(rows)
    by_label = {row["label"]: row for row in report_rows}

    assert by_label["A"]["recall"] == 0.5
    assert by_label["B"]["precision"] == 0.5
    assert round(averages["macro_f1"], 4) == 0.6667
    assert round(averages["weighted_f1"], 4) == 0.6667


def test_classifier_confusion_pairs_counts_misclassifications():
    rows = [
        {"true_label": "Acute", "predicted_label": "Chronic"},
        {"true_label": "Acute", "predicted_label": "Chronic"},
        {"true_label": "Acute", "predicted_label": "Acute"},
    ]

    assert classifier_confusion_pairs(rows) == [
        {"true_label": "Acute", "predicted_label": "Chronic", "count": 2}
    ]


def test_retraining_recommendation_avoids_retraining_for_strong_existing_artifacts():
    classifier = {
        "available": True,
        "accuracy_from_predictions": 0.989,
        "macro_f1_from_predictions": 0.979,
    }
    rag = {"available": True, "metadata_loaded": True}

    assert retraining_recommendation(classifier, rag).startswith("No retraining is needed")
