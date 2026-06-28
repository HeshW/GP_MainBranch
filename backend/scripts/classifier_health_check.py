"""Health check for the active ClinicalBERT classifier bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from models.diagnosis.rag import FineTunedDiagnosisClassifier


REQUIRED_FILES = (
    "config.json",
    "label_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the configured classifier without printing secrets.")
    parser.add_argument("--classifier-dir", type=Path, default=None)
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    classifier_dir = args.classifier_dir or Path(settings.finetuned_model_dir or "")
    label_map = read_json(classifier_dir / "label_map.json")
    label_to_id = label_map.get("label_to_id") or label_map.get("label2id") or {}
    missing = [name for name in REQUIRED_FILES if not (classifier_dir / name).exists()]
    has_weights = any((classifier_dir / name).exists() for name in WEIGHT_FILES)
    if not has_weights:
        missing.append("model weights (model.safetensors or pytorch_model.bin)")

    payload: dict[str, Any] = {
        "status": "ok" if classifier_dir.is_dir() and not missing else "blocked",
        "enabled": bool(settings.use_finetuned_classifier),
        "classifier_dir": str(classifier_dir),
        "classifier_dir_exists": classifier_dir.is_dir(),
        "missing_files": missing,
        "label_count": len(label_to_id),
        "max_length": settings.classifier_max_length,
        "translate_arabic": bool(settings.classifier_translate_arabic),
        "model_loaded": False,
        "smoke_prediction": None,
    }

    if args.skip_model_load or missing:
        return payload

    try:
        classifier = FineTunedDiagnosisClassifier(
            model_dir=classifier_dir,
            max_length=settings.classifier_max_length,
        )
        prediction = classifier.predict(
            "Patient with exertional chest pressure radiating to the left arm that improves with rest."
        )
        payload["model_loaded"] = True
        payload["smoke_prediction"] = {
            "predicted_label": prediction.get("predicted_label"),
            "confidence": prediction.get("confidence"),
            "top_predictions": prediction.get("top_predictions"),
        }
        payload["status"] = "ok"
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
