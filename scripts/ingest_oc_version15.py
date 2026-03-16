"""Ingest GPProject OC version text file and emit a synonyms JSON.

This version calculates repository root robustly so it can run from
`models/ocr/scripts/` after being moved into the package.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


MAPPING_LINE = re.compile(r"^\s*([A-Za-z0-9%µμ\-\s\(\)/]+?)\s*[:\-]\s*(.+)$")
SPLIT_ALIASES = re.compile(r"[,;\|/]")


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / 'requirements.txt').exists() or (p / '.git').exists() or (p / 'README.md').exists():
            return p
    return start.parents[-1]


def ingest(source: Path, out: Path) -> int:
    if not source.exists():
        print(f"Source file not found: {source}")
        return 2

    aliases: dict[str, str] = {}
    lines = source.read_text(encoding='utf-8', errors='ignore').splitlines()

    for ln in lines:
        m = MAPPING_LINE.match(ln)
        if not m:
            continue
        left = m.group(1).strip()
        right = m.group(2).strip()
        canonical = left.lower()
        parts = [p.strip().lower() for p in SPLIT_ALIASES.split(right) if p.strip()]
        for alias in parts:
            if not alias:
                continue
            if alias == canonical:
                continue
            aliases[alias] = canonical

    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "aliases": aliases,
        "meta": {
            "source": str(source),
            "generated": datetime.utcnow().isoformat() + 'Z',
            "count": len(aliases),
        },
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    report_dir = Path('logs')
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / 'ingest_v15_report.txt'
    report_lines = [f"Wrote {out} ({len(aliases)} aliases)", f"Source: {source}"]
    report_file.write_text('\n'.join(report_lines), encoding='utf-8')

    print(report_lines[0])
    return 0


if __name__ == '__main__':
    repo_root = find_repo_root(Path(__file__).resolve())
    p = argparse.ArgumentParser()
    p.add_argument('--source', '-s', type=Path, default=repo_root / 'GPProject_OC_Version15.txt')
    p.add_argument('--out', '-o', type=Path, default=repo_root / 'models' / 'ocr' / 'synonyms_v15.json')
    args = p.parse_args()
    raise SystemExit(ingest(args.source, args.out))
