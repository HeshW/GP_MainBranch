#!/usr/bin/env python3
"""Compare installed packages in the active environment against a requirements file.

This script is moved into models/ocr/scripts; it discovers the repo root so the
default requirements path resolves correctly when run from the package.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / 'requirements.txt').exists() or (p / '.git').exists() or (p / 'README.md').exists():
            return p
    return start.parents[-1]


repo_root = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(repo_root))


@dataclass
class RequirementSpec:
    name: str
    raw: str
    op: Optional[str] = None
    version: Optional[str] = None


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(==|!=|>=|<=|>|<)?\s*([^\s;#]+)?")
_INCLUDE_RE = re.compile(r"^\s*(?:-r|--requirement)\s+(.+?)\s*$")


def _strip_inline_comment(line: str) -> str:
    """Drop trailing comments while keeping URL fragments intact."""
    return line.split(" #", 1)[0].strip()


def _parse_requirements_file(path: Path, specs: Dict[str, RequirementSpec], visited: Set[Path]) -> None:
    path = path.resolve()
    if path in visited:
        return
    visited.add(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = _strip_inline_comment(line.strip())
            if not stripped or stripped.startswith("#"):
                continue

            include_match = _INCLUDE_RE.match(stripped)
            if include_match:
                include_target = include_match.group(1).strip()
                include_path = (path.parent / include_target).resolve()
                _parse_requirements_file(include_path, specs, visited)
                continue

            # Ignore pip options that are not package requirement specs.
            if stripped.startswith("-"):
                continue

            m = _REQ_RE.match(stripped)
            if not m:
                continue
            name, op, version = m.groups()
            key = name.lower().replace("_", "-")
            specs[key] = RequirementSpec(name=name, raw=stripped, op=op, version=version)


def parse_requirements(path: str) -> Dict[str, RequirementSpec]:
    specs: Dict[str, RequirementSpec] = {}
    _parse_requirements_file(Path(path), specs, set())
    return specs


def get_installed() -> Dict[str, str]:
    cmd = [sys.executable, "-m", "pip", "list", "--format", "json"]
    out = subprocess.check_output(cmd, text=True)
    rows = json.loads(out)
    installed: Dict[str, str] = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        version = str(row.get("version", "")).strip()
        if not name:
            continue
        installed[name.lower().replace("_", "-")] = version
    return installed


def cmp_version(a: str, b: str) -> Optional[int]:
    try:
        from packaging.version import Version  # type: ignore

        va = Version(a)
        vb = Version(b)
        return (va > vb) - (va < vb)
    except Exception:
        pass

    def to_nums(v: str) -> Tuple[int, ...]:
        parts = re.findall(r"\d+", v)
        return tuple(int(p) for p in parts)

    try:
        na, nb = to_nums(a), to_nums(b)
        return (na > nb) - (na < nb)
    except Exception:
        return None


def satisfies(installed: str, op: Optional[str], expected: Optional[str]) -> Optional[bool]:
    if op is None or expected is None:
        return True
    c = cmp_version(installed, expected)
    if c is None:
        return None
    if op == "==":
        return c == 0
    if op == "!=":
        return c != 0
    if op == ">=":
        return c >= 0
    if op == "<=":
        return c <= 0
    if op == ">":
        return c > 0
    if op == "<":
        return c < 0
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check env package mismatch against requirements")
    parser.add_argument("--requirements", default=str(repo_root / "requirements.txt"), help="Path to requirements file")
    args = parser.parse_args()

    try:
        specs = parse_requirements(args.requirements)
    except FileNotFoundError:
        print(f"ERROR: requirements file not found: {args.requirements}")
        return 2

    installed = get_installed()

    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    print(f"Requirements file: {args.requirements}")
    print()

    missing: List[str] = []
    mismatched: List[str] = []
    unknown_cmp: List[str] = []
    matched_count = 0

    for key, spec in sorted(specs.items()):
        found = installed.get(key)
        if found is None:
            missing.append(f"- {spec.name}: missing (required: {spec.raw})")
            continue

        ok = satisfies(found, spec.op, spec.version)
        if ok is True:
            matched_count += 1
            continue
        if ok is False:
            mismatched.append(f"- {spec.name}: installed {found}, required {spec.raw}")
        else:
            unknown_cmp.append(f"- {spec.name}: installed {found}, unable to compare with {spec.raw}")

    print(f"Matched:   {matched_count}")
    print(f"Missing:   {len(missing)}")
    print(f"Mismatch:  {len(mismatched)}")
    print(f"Unknown:   {len(unknown_cmp)}")

    if missing:
        print("\nMissing packages:")
        print("\n".join(missing))

    if mismatched:
        print("\nVersion mismatches:")
        print("\n".join(mismatched))

    if unknown_cmp:
        print("\nComparison warnings:")
        print("\n".join(unknown_cmp))

    return 1 if (missing or mismatched) else 0


if __name__ == "__main__":
    raise SystemExit(main())
