from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_checker_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_env_mismatch.py"
    spec = importlib.util.spec_from_file_location("check_env_mismatch", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_requirements_resolves_include_directives(tmp_path):
    checker = _load_checker_module()

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    (tmp_path / "requirements.txt").write_text(
        "-r nested/runtime.txt\n--requirement nested/api.txt\n",
        encoding="utf-8",
    )
    (nested_dir / "runtime.txt").write_text("numpy==1.23.5\n", encoding="utf-8")
    (nested_dir / "api.txt").write_text("fastapi>=0.110\n", encoding="utf-8")

    specs = checker.parse_requirements(str(tmp_path / "requirements.txt"))

    assert "numpy" in specs
    assert specs["numpy"].op == "=="
    assert specs["numpy"].version == "1.23.5"
    assert "fastapi" in specs
    assert specs["fastapi"].op == ">="


def test_parse_requirements_handles_nested_relative_includes(tmp_path):
    checker = _load_checker_module()

    level1 = tmp_path / "level1"
    level2 = level1 / "level2"
    level2.mkdir(parents=True)

    (tmp_path / "requirements.txt").write_text("-r level1/base.txt\n", encoding="utf-8")
    (level1 / "base.txt").write_text("-r level2/extra.txt\n", encoding="utf-8")
    (level2 / "extra.txt").write_text("pydantic-settings==2.13.1\n", encoding="utf-8")

    specs = checker.parse_requirements(str(tmp_path / "requirements.txt"))

    assert "pydantic-settings" in specs
    assert specs["pydantic-settings"].version == "2.13.1"


def test_parse_requirements_ignores_non_package_pip_options(tmp_path):
    checker = _load_checker_module()

    (tmp_path / "requirements.txt").write_text(
        "--index-url https://example.com/simple\n"
        "--extra-index-url https://example2.com/simple\n"
        "--find-links https://example.com/wheels\n"
        "-c constraints.txt\n"
        "pytest==9.0.3\n",
        encoding="utf-8",
    )

    specs = checker.parse_requirements(str(tmp_path / "requirements.txt"))

    assert list(specs.keys()) == ["pytest"]
