from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize pipeline failures from an end-to-end summary JSON.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/archive/final_cleanup/misc/pipeline_failure_analysis.json"),
    )
    return parser.parse_args()


def categorize(detail: dict) -> str:
    if detail.get("top_1_clinical_match"):
        return "clinically_close_but_not_exact"
    prediction = str(detail.get("top_1_prediction", "")).lower()
    if prediction.startswith("possible "):
        return "rule_pattern_too_generic"
    candidate_labels = [str(item).lower() for item in detail.get("clarification_candidate_diseases", [])]
    expected = str(detail.get("primary_expected_condition", "")).lower()
    if expected and any(expected in item or item in expected for item in candidate_labels):
        return "candidate_list_contains_truth_but_rank_failed"
    if detail.get("clarification_needed") and not detail.get("clarification_applied"):
        return "needs_follow_up_answers"
    if detail.get("primary_source", "").startswith("rag"):
        return "rag_misfire"
    if detail.get("primary_source") == "classifier":
        return "classifier_misfire"
    return "other"


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    details = summary.get("details", [])
    failures = [item for item in details if not item.get("top_1_correct")]
    categorized = []
    counter = Counter()
    for item in failures:
        category = categorize(item)
        counter[category] += 1
        categorized.append(
            {
                "case_id": item.get("case_id"),
                "expected": item.get("primary_expected_condition"),
                "top_1_prediction": item.get("top_1_prediction"),
                "primary_source": item.get("primary_source"),
                "category": category,
                "clarification_candidate_diseases": item.get("clarification_candidate_diseases", []),
            }
        )

    payload = {
        "summary_file": str(args.summary),
        "num_failures": len(failures),
        "category_distribution": dict(counter),
        "cases": categorized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
