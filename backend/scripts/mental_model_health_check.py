"""Health check and asset inventory for the separate mental-health LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mental_health.mental_llm import (
    DISCLAIMER,
    apply_mental_health_guardrails,
    generate_mental_support_reply,
    inspect_mental_model_assets,
    preload_model,
)

DEFAULT_OUTPUT_DIR = Path("data/evaluation/mental_model_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check mental-health LoRA adapter deployment readiness.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--load-model", action="store_true", help="Actually load base model and LoRA adapter.")
    parser.add_argument("--generate", action="store_true", help="Generate one smoke reply after loading is attempted.")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any], *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None)
        handle.write("\n")


def write_asset_report(output_dir: Path, assets: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "mental_model_assets_report.json", assets, pretty=True)
    lines = [
        "# Mental Health Model Assets Report",
        "",
        f"- Status: `{assets['status']}`",
        f"- Configured model dir: `{assets['configured_model_dir']}`",
        f"- Resolved model dir: `{assets['resolved_model_dir']}`",
        f"- Base model: `{assets.get('base_model_name_or_path')}`",
        f"- PEFT type: `{assets.get('peft_type')}`",
        f"- LoRA adapter: `{assets['is_lora_adapter']}`",
        f"- Merged model: `{assets['is_merged_model']}`",
        f"- Adapter config exists: `{assets['adapter_config_exists']}`",
        f"- Adapter weights exist: `{assets['adapter_weights_exists']}`",
        f"- Tokenizer files exist: `{assets['tokenizer_files_exist']}`",
        f"- Missing files: `{', '.join(assets['missing_files']) if assets['missing_files'] else 'none'}`",
    ]
    if assets.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in assets["warnings"])
    lines.extend(
        [
            "",
            "This adapter is deployed separately from the medical diagnosis RAG/classifier/rules pipeline.",
            "",
        ]
    )
    (output_dir / "mental_model_assets_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    assets = inspect_mental_model_assets(args.model_dir)
    crisis = apply_mental_health_guardrails("I want to kill myself", "en")
    payload: dict[str, Any] = {
        "status": "ok" if assets["status"] == "ok" and crisis["safety_status"] == "crisis" else "blocked",
        "disclaimer": DISCLAIMER,
        "assets": assets,
        "guardrails": {
            "crisis_without_model_load": crisis["safety_status"] == "crisis" and bool(crisis["reply"]),
            "crisis_reply": crisis["reply"],
        },
        "model_loaded": False,
        "load_error": None,
        "generation": None,
    }

    if args.load_model or args.generate:
        load_status = preload_model()
        payload["model_loaded"] = bool(load_status.get("model_loaded"))
        payload["load_error"] = load_status.get("error")
        if not payload["model_loaded"]:
            payload["status"] = "degraded"

    if args.generate:
        result = generate_mental_support_reply("I feel overwhelmed and anxious", language="en")
        payload["generation"] = {
            "reply": result.get("reply"),
            "safety_status": result.get("safety_status"),
            "detected_language": result.get("detected_language"),
            "model_loaded": result.get("model_loaded"),
            "latency_ms": result.get("latency_ms"),
            "error": result.get("error"),
        }

    return payload


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    write_asset_report(args.output_dir, payload["assets"])
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    if payload["status"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

