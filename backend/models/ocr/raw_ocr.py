from __future__ import annotations

from typing import Any, List


def collect_text(paddle_result: Any) -> str:
    lines: List[str] = []
    if not paddle_result:
        return ""
    for page in paddle_result:
        if not page:
            continue
        for line in page:
            try:
                text, _confidence = line[1]
                lines.append(text)
            except (IndexError, TypeError, ValueError):
                continue
    return "\n".join(lines)


def collect_raw_ocr(paddle_result: Any) -> list[dict]:
    items: list[dict] = []
    if not paddle_result:
        return items
    for page in paddle_result:
        if not page:
            continue
        for line in page:
            try:
                bbox = line[0]
                text, conf = line[1]
                items.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": float(conf) if conf is not None else None,
                })
            except (IndexError, TypeError, ValueError):
                continue
    return items
