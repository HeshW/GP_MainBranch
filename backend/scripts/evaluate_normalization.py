from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.symptom_parser import parse_symptoms
from manager.symptom_validator import validate_parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate symptom normalization and parsing.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/evaluation/normalization_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/normalization_summary.json"),
    )
    return parser.parse_args()


def _safe_list(value: Any) -> list[str]:
    return [str(item).strip().lower() for item in (value or []) if str(item).strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    details = []
    tp = fp = fn = 0
    exact_matches = 0
    negation_exact_matches = 0
    context_hits = 0
    context_total = 0

    for case in cases:
        parsed = parse_symptoms(case["text"])
        validated = validate_parsed(parsed)
        predicted = set(_safe_list(validated.get("symptoms")))
        expected = set(_safe_list(case.get("expected_symptoms")))
        predicted_negated = set(_safe_list(validated.get("negated_symptoms")))
        expected_negated = set(_safe_list(case.get("expected_negated_symptoms")))

        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

        if predicted == expected:
            exact_matches += 1
        if predicted_negated == expected_negated:
            negation_exact_matches += 1

        expected_context = case.get("expected_context", {}) or {}
        predicted_context = validated.get("context", {}) or {}
        case_context_hits = 0
        case_context_total = 0
        for key, expected_values in expected_context.items():
            expected_norm = set(_safe_list(expected_values))
            predicted_norm = set(_safe_list(predicted_context.get(key)))
            case_context_hits += len(expected_norm & predicted_norm)
            case_context_total += len(expected_norm)
        context_hits += case_context_hits
        context_total += case_context_total

        details.append(
            {
                "case_id": case.get("id"),
                "predicted_symptoms": sorted(predicted),
                "expected_symptoms": sorted(expected),
                "predicted_negated_symptoms": sorted(predicted_negated),
                "expected_negated_symptoms": sorted(expected_negated),
                "predicted_context": predicted_context,
                "expected_context": expected_context,
                "symptom_exact_match": predicted == expected,
                "negation_exact_match": predicted_negated == expected_negated,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    summary = {
        "num_cases": len(cases),
        "symptom_precision": precision,
        "symptom_recall": recall,
        "symptom_f1": f1,
        "normalization_exact_match_rate": exact_matches / len(cases) if cases else 0.0,
        "negation_exact_match_rate": negation_exact_matches / len(cases) if cases else 0.0,
        "context_recall": context_hits / context_total if context_total else 0.0,
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
