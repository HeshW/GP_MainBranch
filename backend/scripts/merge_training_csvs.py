"""Merge base classifier CSVs with targeted hard-case CSVs.

Typical usage:

    python merge_training_csvs.py \
        --base-csv processed_ddxplus/train_natural.csv \
        --targeted-csv targeted_training/train_targeted.csv \
        --output-csv targeted_training/train_merged.csv
"""

from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Merge base and targeted classifier CSVs.")
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "processed_ddxplus" / "train_natural.csv",
            DEFAULT_BASE_DIR / "data" / "processed_ddxplus" / "train_natural.csv",
        ),
        help="Base CSV path. Searches ./processed_ddxplus/... then ./data/processed_ddxplus/...",
    )
    parser.add_argument(
        "--targeted-csv",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "targeted_training" / "train_targeted.csv",
            DEFAULT_BASE_DIR / "data" / "targeted_training" / "train_targeted.csv",
        ),
        help="Targeted CSV path. Searches ./targeted_training/... then ./data/targeted_training/...",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_BASE_DIR / "targeted_training" / "train_merged.csv",
        help="Merged output CSV path. Default: ./targeted_training/train_merged.csv",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = ((row.get("combined_text") or "").strip(), (row.get("pathology") or "").strip())
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    merged = dedupe_rows(read_rows(args.base_csv) + read_rows(args.targeted_csv))
    write_rows(args.output_csv, merged)
    print(f"Wrote {len(merged)} merged rows to {args.output_csv}")


if __name__ == "__main__":
    main()
