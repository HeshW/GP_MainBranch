"""Validate curated targeted interactive diagnosis cases.

The goal is to maintain a high-quality in-scope benchmark/training set that
matches the project's clarification workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_FIELDS = ("id", "raw_text", "expected_conditions", "follow_up_answers")
OPTIONAL_FIELDS = (
    "language",
    "difficulty",
    "ambiguity_group",
    "discriminative_symptoms",
    "negated_symptoms",
    "notes",
)
ALLOWED_LANGUAGES = {"en", "ar", "mixed"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate targeted interactive cases JSON.")
    parser.add_argument("--cases", type=Path, required=True, help="Path to JSON list of cases.")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Cases file must be a JSON list.")
    if not all(isinstance(item, dict) for item in payload):
        raise TypeError("Each case must be a JSON object.")
    return payload


def validate_case(case: dict[str, Any], seen_ids: set[str], index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"case[{index}]"

    for field in REQUIRED_FIELDS:
        if field not in case:
            errors.append(f"{prefix}: missing required field '{field}'")

    case_id = str(case.get("id", "")).strip()
    if not case_id:
        errors.append(f"{prefix}: 'id' must be a non-empty string")
    elif case_id in seen_ids:
        errors.append(f"{prefix}: duplicate id '{case_id}'")
    else:
        seen_ids.add(case_id)

    raw_text = str(case.get("raw_text", "")).strip()
    if not raw_text:
        errors.append(f"{prefix}: 'raw_text' must be a non-empty string")

    expected = case.get("expected_conditions")
    if not isinstance(expected, list) or not expected or not all(str(item).strip() for item in expected):
        errors.append(f"{prefix}: 'expected_conditions' must be a non-empty list of strings")

    answers = case.get("follow_up_answers")
    if not isinstance(answers, list) or not answers or not all(str(item).strip() for item in answers):
        errors.append(f"{prefix}: 'follow_up_answers' must be a non-empty list of strings")

    language = str(case.get("language", "")).strip()
    if language and language not in ALLOWED_LANGUAGES:
        errors.append(f"{prefix}: unsupported language '{language}'")

    difficulty = str(case.get("difficulty", "")).strip()
    if difficulty and difficulty not in ALLOWED_DIFFICULTIES:
        errors.append(f"{prefix}: unsupported difficulty '{difficulty}'")

    for field in ("discriminative_symptoms", "negated_symptoms"):
        value = case.get(field)
        if value is not None and (not isinstance(value, list) or not all(str(item).strip() for item in value)):
            errors.append(f"{prefix}: '{field}' must be a list of strings when provided")

    unknown_fields = sorted(set(case.keys()) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown_fields:
        errors.append(f"{prefix}: unknown fields present: {', '.join(unknown_fields)}")

    return errors


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    seen_ids: set[str] = set()
    errors: list[str] = []
    for index, case in enumerate(cases, start=1):
        errors.extend(validate_case(case, seen_ids, index))

    summary = {
        "cases_file": str(args.cases),
        "num_cases": len(cases),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
