from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .patterns import LAB_LABEL_PATTERNS, LAB_PATTERNS, LEADING_VALUE_PATTERN
from .types import LabEntry, OCRResult

CONFIDENCE_WARNING = 0.6
CONFIDENCE_CRITICAL = 0.4


def normalise_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def build_entry(canonical: str, match: "re.Match[str]", warnings: List[str]) -> Optional[LabEntry]:
    raw_value = match.group(1)
    raw_unit = match.group(2)
    source_match = match.group(0)
    try:
        value = float(raw_value.replace(",", "."))
    except ValueError:
        warnings.append(f"Could not parse numeric value for '{canonical}': '{raw_value}'")
        return None
    return {
        "value": value,
        "unit": raw_unit if raw_unit else None,
        "source_match": source_match,
    }


def extract_from_text_block(
    text_block: str,
    labs: Dict[str, LabEntry],
    warnings: List[str],
    *,
    skip_existing: bool = True,
) -> None:
    for canonical, pattern in LAB_PATTERNS.items():
        if skip_existing and canonical in labs:
            continue
        matches = list(pattern.finditer(text_block))
        if not matches:
            continue
        if len(matches) > 1:
            warnings.append(
                f"Multiple matches ({len(matches)}) for '{canonical}'; keeping last match."
            )
        entry = build_entry(canonical, matches[-1], warnings)
        if entry is not None:
            labs[canonical] = entry


def cross_line_fallback(lines: List[str], labs: Dict[str, LabEntry], warnings: List[str]) -> None:
    missing = set(LAB_PATTERNS) - set(labs)
    if not missing:
        return

    for index, raw_line in enumerate(lines):
        if not missing:
            break
        norm_line = normalise_text(raw_line)
        if not norm_line:
            continue
        next_index = index + 1
        while next_index < len(lines) and not normalise_text(lines[next_index]):
            next_index += 1
        if next_index >= len(lines):
            continue

        next_norm = normalise_text(lines[next_index])
        value_match = LEADING_VALUE_PATTERN.match(next_norm)
        if value_match is None:
            continue

        for canonical in list(missing):
            if LAB_LABEL_PATTERNS[canonical].search(norm_line) is None:
                continue
            if LAB_PATTERNS[canonical].search(norm_line) is not None:
                continue
            raw_value = value_match.group(1)
            raw_unit = value_match.group(2)
            try:
                value = float(raw_value.replace(",", "."))
            except ValueError:
                continue
            warnings.append(
                f"Cross-line fallback for '{canonical}': label on line {index + 1}, value on line {next_index + 1}."
            )
            labs[canonical] = {
                "value": value,
                "unit": raw_unit if raw_unit else None,
                "source_match": f"{norm_line} -> {next_norm}",
            }
            missing.discard(canonical)


def parse_labs(text: str) -> tuple[Dict[str, LabEntry], List[str]]:
    labs: Dict[str, LabEntry] = {}
    warnings: List[str] = []
    normalised = normalise_text(text)
    extract_from_text_block(normalised, labs, warnings, skip_existing=False)
    if len(labs) < len(LAB_PATTERNS):
        for line in text.splitlines():
            norm_line = normalise_text(line)
            if norm_line:
                extract_from_text_block(norm_line, labs, warnings, skip_existing=True)
    cross_line_fallback(text.splitlines(), labs, warnings)
    return labs, warnings


def attach_confidences(labs: Dict[str, LabEntry], raw_ocr: list[dict], warnings: List[str]) -> None:
    all_confidences = [item["confidence"] for item in raw_ocr if isinstance(item.get("confidence"), (int, float))]
    for canonical, entry in labs.items():
        source = (entry.get("source_match") or "").lower()
        tokens = set(re.findall(r"\w{2,}", source))
        matched = []
        for item in raw_ocr:
            text = (item.get("text") or "").lower()
            if text and any(token in text for token in tokens):
                confidence = item.get("confidence")
                if isinstance(confidence, (int, float)):
                    matched.append(confidence)
        if matched:
            avg = float(sum(matched)) / len(matched)
        elif all_confidences:
            avg = float(sum(all_confidences)) / len(all_confidences)
        else:
            avg = None
        entry["confidence"] = round(avg, 3) if avg is not None else None
        if avg is not None and avg < CONFIDENCE_WARNING:
            warnings.append(f"Low OCR confidence for '{canonical}': {avg:.2f}")
        if avg is not None and avg < CONFIDENCE_CRITICAL:
            warnings.append(f"Critical OCR confidence for '{canonical}': {avg:.2f}")


def extract_labs_from_text(text: str) -> OCRResult:
    labs, warnings = parse_labs(text)
    return {
        "labs": labs,
        "raw_text": normalise_text(text),
        "warnings": warnings,
    }
