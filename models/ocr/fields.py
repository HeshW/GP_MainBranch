"""Header field and section extraction utilities for OCREngine.

Provides tolerant regexes to extract common header fields (title, patient
name, sex/age, date, vcode, path_code) and split narrative sections such as
Clinical, Gross, Microscopic, Diagnosis, and Notes.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple


HEADER_PATTERNS = {
    "title": re.compile(r"^\s*([A-Za-z ].{3,200})\s*$", re.IGNORECASE),
    "patient_name": re.compile(r"Patient(?:'s)?\s*Name[:\s]*([^\n\r]+)", re.IGNORECASE),
    "sex_age": re.compile(r"Sex\s*/\s*Age[:\s]*([^\n\r]+)", re.IGNORECASE),
    "date": re.compile(r"Date[:\s]*([0-9]{1,2}[\/-][0-9]{1,2}[\/-][0-9]{2,4})", re.IGNORECASE),
    "vcode": re.compile(r"VCode[:\s]*([A-Za-z0-9\-_]+)", re.IGNORECASE),
    "path_code": re.compile(r"Path\.?Code[:\s]*([^\n\r]+)", re.IGNORECASE),
}

SECTION_HEADING = re.compile(r"^(Clinical|Gross|Microscopic|Diagnosis|Notes)\s*:\s*$", re.IGNORECASE | re.MULTILINE)


def parse_header_fields(text: str) -> Dict[str, str]:
    """Extract header fields from OCR text.

    Returns a mapping of detected field names to their raw string values.
    Fields not found are omitted.
    """
    fields: Dict[str, str] = {}

    # Try specific patterns first
    for key, pat in HEADER_PATTERNS.items():
        m = pat.search(text)
        if m:
            fields[key] = m.group(1).strip()

    # Heuristic title: first non-empty line if no explicit patient_name found
    if "patient_name" not in fields:
        for line in text.splitlines():
            ln = line.strip()
            if ln:
                # Skip obvious headings
                if len(ln) > 3 and not ln.lower().startswith("clinical"):
                    fields.setdefault("title", ln)
                break

    return fields


def split_sections(text: str) -> List[Dict[str, str]]:
    """Split the text into labeled sections using known headings.

    Returns list of dicts: {"label": str, "text": str}
    """
    lines = text.splitlines()
    sections: List[Dict[str, str]] = []
    current_label = None
    current_lines: List[str] = []

    def flush():
        if current_label or current_lines:
            label = current_label if current_label else "body"
            sections.append({"label": label, "text": "\n".join(current_lines).strip()})

    i = 0
    while i < len(lines):
        ln = lines[i]
        m = SECTION_HEADING.match(ln)
        if m:
            # flush previous
            if current_label or current_lines:
                flush()
            current_label = m.group(1).capitalize()
            current_lines = []
            i += 1
            continue

        current_lines.append(ln)
        i += 1

    # final flush
    if current_label or current_lines:
        flush()

    return sections


def extract_fields_and_sections(text: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Convenience wrapper returning (fields, sections).

    Normalises input then extracts header fields and sections.
    """
    if not text:
        return {}, []
    fields = parse_header_fields(text)
    sections = split_sections(text)
    return fields, sections
