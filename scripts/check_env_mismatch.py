#!/usr/bin/env python3
"""Compare installed packages in the active environment against a requirements file.

Usage:
    python scripts/check_env_mismatch.py --requirements requirements.txt
    python scripts/check_env_mismatch.py --requirements requirements-runtime.txt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RequirementSpec:
    name: str
    raw: str
    op: Optional[str] = None
    version: Optional[str] = None


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(==|!=|>=|<=|>|<)?\s*([^\s;#]+)?")


def parse_requirements(path: str) -> Dict[str, RequirementSpec]:
    specs: Dict[str, RequirementSpec] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _REQ_RE.match(stripped)
            if not m:
                continue
            name, op, version = m.groups()
            key = name.lower().replace("_", "-")
            specs[key] = RequirementSpec(name=name, raw=stripped, op=op, version=version)
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
    """Best-effort version comparison.

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b, None if comparison failed.
    """
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
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements file")
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
