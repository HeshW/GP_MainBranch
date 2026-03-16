#!/usr/bin/env python3
"""Run the test suite and print a concise exit code line for CI-friendly parsing."""
import sys
import pytest


def main() -> int:
    rc = pytest.main(["-q"])
    print(f"PYTEST_EXIT_CODE:{rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
