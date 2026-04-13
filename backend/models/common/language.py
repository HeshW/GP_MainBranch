from __future__ import annotations

import re
from typing import Any

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def contains_arabic(text: Any) -> bool:
    return bool(_ARABIC_RE.search(str(text or "")))


def normalize_language(value: str | None, default: str = "en") -> str:
    language = str(value or "").strip().lower()
    if language.startswith("ar"):
        return "ar"
    if language.startswith("en"):
        return "en"
    return "ar" if str(default).strip().lower().startswith("ar") else "en"


def detect_preferred_language(*values: Any, default: str = "en") -> str:
    def _walk(item: Any) -> bool:
        if item is None:
            return False
        if isinstance(item, dict):
            return any(_walk(value) for value in item.values())
        if isinstance(item, (list, tuple, set)):
            return any(_walk(value) for value in item)
        return contains_arabic(item)

    if any(_walk(value) for value in values):
        return "ar"
    return normalize_language(default, default=default)
