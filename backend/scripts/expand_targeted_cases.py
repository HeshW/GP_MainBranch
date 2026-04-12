"""Expand targeted interactive diagnosis cases into a larger benchmark/training draft.

This script is designed to accelerate Phase 5 by generating schema-compatible
variants from the existing curated targeted set.

Typical usage:

    python backend/scripts/expand_targeted_cases.py \
        --input-cases data/evaluation/targeted_cases_v1.json \
        --output-cases data/evaluation/targeted_cases_v1_expanded_128.json \
        --target-count 128
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()

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
    parser = argparse.ArgumentParser(description="Expand targeted interactive diagnosis cases.")
    parser.add_argument(
        "--input-cases",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_ROOT / "data" / "evaluation" / "targeted_cases_v1.json",
            DEFAULT_ROOT / "targeted_cases_v1.json",
        ),
        help="Input targeted cases JSON list.",
    )
    parser.add_argument(
        "--output-cases",
        type=Path,
        default=DEFAULT_ROOT / "data" / "evaluation" / "targeted_cases_v1_expanded_128.json",
        help="Output JSON path for expanded cases.",
    )
    parser.add_argument("--target-count", type=int, default=128, help="Target number of total cases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mixed-language-rate",
        type=float,
        default=0.25,
        help="Probability of generating mixed-language variants.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input cases not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise TypeError("Input cases file must be a JSON list of objects.")
    return payload


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return cleaned.strip("_") or "case"


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def surface_variant(symptom: str, rng: random.Random) -> str:
    key = str(symptom or "").strip().lower()
    variants: dict[str, list[str]] = {
        "shortness of breath": ["shortness of breath", "breathlessness", "dyspnea"],
        "chest pain": ["chest pain", "chest discomfort", "chest pressure"],
        "pleuritic chest pain": ["pleuritic chest pain", "pain worse with deep breaths"],
        "productive cough": ["productive cough", "cough with sputum"],
        "dry cough": ["dry cough", "non-productive cough"],
        "wheezing": ["wheezing", "whistling breathing"],
        "palpitations": ["palpitations", "rapid heartbeats"],
        "irregular heartbeat": ["irregular heartbeat", "irregular pulse"],
        "weight loss": ["weight loss", "unintentional weight loss"],
        "fatigue": ["fatigue", "tiredness"],
        "fever": ["fever", "high temperature"],
        "hoarseness": ["hoarseness", "voice hoarseness"],
        "sore throat": ["sore throat", "throat pain"],
        "reflux": ["reflux", "acid reflux"],
    }
    options = variants.get(key)
    if not options:
        return symptom
    return rng.choice(options)


def join_en(items: list[str]) -> str:
    if not items:
        return "symptoms"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def build_raw_text(
    base: dict[str, Any],
    *,
    language: str,
    variant_index: int,
    rng: random.Random,
) -> str:
    discriminative = [
        surface_variant(str(item).strip(), rng)
        for item in (base.get("discriminative_symptoms") or [])
        if str(item).strip()
    ]
    negated = [str(item).strip() for item in (base.get("negated_symptoms") or []) if str(item).strip()]

    if not discriminative:
        source_text = str(base.get("raw_text", "")).strip()
        parts = [p.strip() for p in re.split(r"[,.]", source_text) if p.strip()]
        discriminative = parts[:3] if parts else ["general symptoms"]

    rng.shuffle(discriminative)
    focus = discriminative[: min(3, len(discriminative))]
    progression_options = [
        "Symptoms have become more noticeable over the last 24-48 hours",
        "The pattern has been persistent and clinically concerning",
        "The episode started clearly and then kept progressing",
    ]
    context = rng.choice(progression_options)

    if language == "mixed":
        symptom_block = " و ".join(focus)
        neg_block = negated[0] if negated else "high fever"
        return (
            f"عندي {symptom_block}، والموضوع بقاله فترة قصيرة لكنه بيزيد. "
            f"{context}. "
            f"There is no {neg_block}."
        )

    neg_sentence = f"There is no {negated[0]}." if negated else "No obvious unrelated symptoms were noted."
    opening_options = [
        "I have",
        "I have been experiencing",
        "I am dealing with",
    ]
    opening = rng.choice(opening_options)
    lead = join_en(focus)
    return f"{opening} {lead}. {context}. {neg_sentence}"


def build_follow_up_answers(
    base: dict[str, Any],
    *,
    language: str,
    rng: random.Random,
) -> list[str]:
    discriminative = [str(item).strip() for item in (base.get("discriminative_symptoms") or []) if str(item).strip()]
    negated = [str(item).strip() for item in (base.get("negated_symptoms") or []) if str(item).strip()]
    if not discriminative:
        discriminative = ["the main symptoms", "the progression"]

    top = surface_variant(discriminative[0], rng)
    second = surface_variant(discriminative[1], rng) if len(discriminative) > 1 else "the timeline"
    third = negated[0] if negated else "features from unrelated conditions"

    if language == "mixed":
        return [
            f"الأعراض الأساسية هي {top} و {second}.",
            "The clinical pattern is focused and not random.",
            f"There is no {third}.",
        ]

    return [
        f"The key findings are {top} and {second}.",
        "The evolution pattern is internally consistent for this diagnosis.",
        f"There is no {third}.",
    ]


def next_case_id(base_id: str, variant_no: int) -> str:
    return f"{slugify(base_id)}_v{variant_no:02d}"


def build_variant_case(
    base: dict[str, Any],
    *,
    variant_no: int,
    mixed_language_rate: float,
    rng: random.Random,
) -> dict[str, Any]:
    base_language = str(base.get("language", "")).strip() or "en"
    language = "mixed" if rng.random() < mixed_language_rate else ("mixed" if base_language == "mixed" else "en")

    expected = [str(item).strip() for item in (base.get("expected_conditions") or []) if str(item).strip()]
    if not expected:
        expected = ["Unknown condition"]

    return {
        "id": next_case_id(str(base.get("id", "targeted_case")), variant_no),
        "language": language,
        "difficulty": str(base.get("difficulty", "hard") or "hard"),
        "ambiguity_group": str(base.get("ambiguity_group", "")).strip(),
        "raw_text": build_raw_text(base, language=language, variant_index=variant_no, rng=rng),
        "expected_conditions": [expected[0]],
        "follow_up_answers": build_follow_up_answers(base, language=language, rng=rng),
        "discriminative_symptoms": unique_keep_order(
            [str(item).strip() for item in (base.get("discriminative_symptoms") or []) if str(item).strip()]
        ),
        "negated_symptoms": unique_keep_order(
            [str(item).strip() for item in (base.get("negated_symptoms") or []) if str(item).strip()]
        ),
        "notes": f"Auto-generated variant {variant_no} from {base.get('id', 'unknown_base')}",
    }


def expand_cases(
    base_cases: list[dict[str, Any]],
    *,
    target_count: int,
    mixed_language_rate: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        raise ValueError("target_count must be > 0")
    if not base_cases:
        raise ValueError("No base cases provided.")

    expanded = list(base_cases)
    existing_ids = {str(item.get("id", "")).strip() for item in expanded}
    variants_per_base = max(1, (target_count - len(base_cases) + len(base_cases) - 1) // len(base_cases))

    for base in base_cases:
        for i in range(variants_per_base):
            variant_no = i + 2
            candidate = build_variant_case(
                base,
                variant_no=variant_no,
                mixed_language_rate=mixed_language_rate,
                rng=rng,
            )
            case_id = str(candidate.get("id", "")).strip()
            if not case_id or case_id in existing_ids:
                continue
            existing_ids.add(case_id)
            expanded.append(candidate)

    # Keep deterministic output and exact target size.
    expanded = sorted(expanded, key=lambda item: str(item.get("id", "")))
    return expanded[:target_count]


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for case in cases:
        group = str(case.get("ambiguity_group", "")).strip() or "unknown"
        by_group[group] = by_group.get(group, 0) + 1
        language = str(case.get("language", "")).strip() or "unknown"
        by_language[language] = by_language.get(language, 0) + 1
    return {
        "num_cases": len(cases),
        "language_distribution": dict(sorted(by_language.items())),
        "ambiguity_group_distribution": dict(sorted(by_group.items())),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    base_cases = load_cases(args.input_cases)
    expanded = expand_cases(
        base_cases,
        target_count=args.target_count,
        mixed_language_rate=max(0.0, min(1.0, float(args.mixed_language_rate))),
        rng=rng,
    )

    args.output_cases.parent.mkdir(parents=True, exist_ok=True)
    with args.output_cases.open("w", encoding="utf-8") as handle:
        json.dump(expanded, handle, indent=2, ensure_ascii=False)

    report = {
        "input_cases": str(args.input_cases),
        "output_cases": str(args.output_cases),
        "target_count": args.target_count,
        "seed": args.seed,
        "mixed_language_rate": args.mixed_language_rate,
        "summary": summarize(expanded),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
