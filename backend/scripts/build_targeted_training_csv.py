"""Convert targeted interactive diagnosis cases into classifier-ready CSV rows.

Typical usage:

    python build_targeted_training_csv.py --include-follow-up
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

DEFAULT_BASE_DIR = Path.cwd()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build classifier CSVs from targeted interactive cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "targeted_cases_v1.json",
            DEFAULT_BASE_DIR / "data" / "evaluation" / "targeted_cases_v1.json",
        ),
        help="Input targeted cases JSON. Searches ./targeted_cases_v1.json then ./data/evaluation/targeted_cases_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "targeted_training",
        help="Output directory for split CSVs. Default: ./targeted_training",
    )
    parser.add_argument("--include-follow-up", action="store_true", help="Include follow-up-enriched rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Cases file must be a JSON list.")
    return payload


def build_rows(cases: list[dict[str, Any]], include_follow_up: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        pathology = str((case.get("expected_conditions") or [""])[0]).strip()
        raw_text = str(case.get("raw_text", "")).strip()
        language = str(case.get("language", "")).strip() or "en"
        difficulty = str(case.get("difficulty", "")).strip() or "hard"
        ambiguity_group = str(case.get("ambiguity_group", "")).strip()
        if not case_id or not pathology or not raw_text:
            continue

        rows.append(
            {
                "patient_id": f"{case_id}_first_turn",
                "pathology": pathology,
                "combined_text": raw_text,
                "symptoms_text": raw_text,
                "language": language,
                "difficulty": difficulty,
                "ambiguity_group": ambiguity_group,
                "case_source": "targeted_first_turn",
            }
        )

        if include_follow_up:
            answers = [str(item).strip() for item in case.get("follow_up_answers", []) if str(item).strip()]
            if answers:
                rows.append(
                    {
                        "patient_id": f"{case_id}_clarified",
                        "pathology": pathology,
                        "combined_text": raw_text + "\n\nFollow-up clarification: " + " ".join(answers),
                        "symptoms_text": raw_text,
                        "language": language,
                        "difficulty": difficulty,
                        "ambiguity_group": ambiguity_group,
                        "case_source": "targeted_clarified",
                    }
                )
    return rows


def split_rows(
    rows: list[dict[str, str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    items = rows[:]
    random.Random(seed).shuffle(items)
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "pathology",
        "combined_text",
        "symptoms_text",
        "language",
        "difficulty",
        "ambiguity_group",
        "case_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    rows = build_rows(cases, include_follow_up=args.include_follow_up)
    train_rows, val_rows, test_rows = split_rows(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    write_csv(args.out_dir / "train_targeted.csv", train_rows)
    write_csv(args.out_dir / "validate_targeted.csv", val_rows)
    write_csv(args.out_dir / "test_targeted.csv", test_rows)

    print(
        json.dumps(
            {
                "cases": len(cases),
                "rows": len(rows),
                "train_rows": len(train_rows),
                "validate_rows": len(val_rows),
                "test_rows": len(test_rows),
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
