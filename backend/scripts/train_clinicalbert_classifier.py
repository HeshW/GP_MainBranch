"""Fine-tune ClinicalBERT for diagnosis classification.

Expected dataset
----------------
CSV files with at least:

- one text column, default: ``combined_text``
- one label column, default: ``pathology``

Typical usage:

    python train_clinicalbert_classifier.py \
        --train-csv targeted_training/train_merged.csv \
        --val-csv targeted_training/validate_merged.csv \
        --test-csv targeted_training/test_merged.csv

Outputs
-------
- best model weights + tokenizer
- label mapping
- training history
- confusion matrix
- classification report
- per-case predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_BASE_DIR = Path.cwd()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ClinicalBERT on diagnosis labels.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "targeted_training" / "train_merged.csv",
            DEFAULT_BASE_DIR / "data" / "targeted_training" / "train_merged.csv",
        ),
        help="Training CSV path. Searches ./targeted_training/... then ./data/targeted_training/...",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "targeted_training" / "validate_merged.csv",
            DEFAULT_BASE_DIR / "data" / "targeted_training" / "validate_merged.csv",
        ),
        help="Validation CSV path. Searches ./targeted_training/... then ./data/targeted_training/...",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "targeted_training" / "test_merged.csv",
            DEFAULT_BASE_DIR / "data" / "targeted_training" / "test_merged.csv",
        ),
        help="Test CSV path. Searches ./targeted_training/... then ./data/targeted_training/...",
    )
    parser.add_argument(
        "--model-name-or-dir",
        default=resolve_existing_path(
            DEFAULT_BASE_DIR / "clinicalbert_classifier_natural",
            DEFAULT_BASE_DIR / "backend" / "artifacts" / "clinicalbert_classifier_natural",
        ),
        help="Base model name or existing fine-tuned model directory. Searches ./clinicalbert_classifier_natural then ./backend/artifacts/clinicalbert_classifier_natural",
    )
    parser.add_argument("--text-column", default="combined_text", help="Text column name")
    parser.add_argument("--label-column", default="pathology", help="Label column name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "clinicalbert_classifier_targeted",
        help="Directory for model and metrics outputs. Default: ./clinicalbert_classifier_targeted",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick smoke training")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for quick validation")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for quick test evaluation")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    return parser.parse_args()


def read_csv_rows(path: Path, text_column: str, label_column: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"Missing text column '{text_column}' in {path}")
        if label_column not in reader.fieldnames:
            raise ValueError(f"Missing label column '{label_column}' in {path}")

        for row in reader:
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not text or not label:
                continue
            rows.append({"text": text, "label": label})
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def maybe_limit_rows(rows: list[dict[str, str]], max_samples: Optional[int], seed: int) -> list[dict[str, str]]:
    if not max_samples or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    sampled = rows[:]
    rng.shuffle(sampled)
    return sampled[:max_samples]


class DiagnosisDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.label_to_id[row["label"]], dtype=torch.long)
        item["raw_label"] = row["label"]
        item["raw_text"] = row["text"]
        return item


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("input_ids", "attention_mask", "labels")
    collated = {key: torch.stack([item[key] for item in batch]) for key in keys}
    if "token_type_ids" in batch[0]:
        collated["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])
    collated["raw_label"] = [item["raw_label"] for item in batch]
    collated["raw_text"] = [item["raw_text"] for item in batch]
    return collated


def build_label_mapping(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({row["label"] for row in rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def build_confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[int]) -> list[list[int]]:
    idx_map = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[idx_map[true_label]][idx_map[pred_label]] += 1
    return matrix


def build_classification_report(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
    id_to_label: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    supports = Counter(y_true)
    rows: list[dict[str, Any]] = []

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = supports.get(label, 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                "label": id_to_label[label],
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    total = len(y_true) if y_true else 1
    class_count = len(labels) if labels else 1
    averages = {
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
        "weighted_precision": weighted_precision / total,
        "weighted_recall": weighted_recall / total,
        "weighted_f1": weighted_f1 / total,
    }
    return rows, averages


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, matrix: list[list[int]], id_to_label: dict[int, str], labels: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *[id_to_label[label] for label in labels]])
        for label, row in zip(labels, matrix):
            writer.writerow([id_to_label[label], *row])


def move_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def evaluate_model(
    model: Any,
    dataloader: DataLoader,
    device: str,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions_rows: list[dict[str, str]] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                token_type_ids=batch.get("token_type_ids"),
            )
            losses.append(float(outputs.loss.item()))
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            true_ids = batch["labels"].detach().cpu().tolist()
            pred_ids = preds.detach().cpu().tolist()
            y_true.extend(true_ids)
            y_pred.extend(pred_ids)

            for text, true_id, pred_id in zip(batch["raw_text"], true_ids, pred_ids):
                predictions_rows.append(
                    {
                        "text": text,
                        "true_label": id_to_label[true_id],
                        "predicted_label": id_to_label[pred_id],
                        "correct": str(true_id == pred_id),
                    }
                )

    labels = sorted(set(y_true) | set(y_pred))
    confusion = build_confusion_matrix(y_true, y_pred, labels)
    report_rows, averages = build_classification_report(y_true, y_pred, labels, id_to_label)
    return {
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion": confusion,
        "labels": labels,
        "report_rows": report_rows,
        "averages": averages,
        "prediction_rows": predictions_rows,
    }


def train_one_epoch(model: Any, dataloader: DataLoader, optimizer: Any, scheduler: Any, device: str) -> float:
    model.train()
    losses: list[float] = []

    for batch in dataloader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    return sum(losses) / len(losses) if losses else 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = maybe_limit_rows(
        read_csv_rows(args.train_csv, args.text_column, args.label_column),
        args.max_train_samples,
        args.seed,
    )
    val_rows = maybe_limit_rows(
        read_csv_rows(args.val_csv, args.text_column, args.label_column) if args.val_csv else [],
        args.max_val_samples,
        args.seed + 1,
    )
    test_rows = maybe_limit_rows(
        read_csv_rows(args.test_csv, args.text_column, args.label_column) if args.test_csv else [],
        args.max_test_samples,
        args.seed + 2,
    )

    label_to_id, id_to_label = build_label_mapping(train_rows)

    for dataset_name, rows in (("validation", val_rows), ("test", test_rows)):
        unseen = sorted({row["label"] for row in rows if row["label"] not in label_to_id})
        if unseen:
            raise ValueError(f"{dataset_name} split contains unseen labels not in train split: {unseen}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_dir,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(args.device)

    train_dataset = DiagnosisDataset(train_rows, tokenizer, label_to_id, args.max_length)
    val_dataset = DiagnosisDataset(val_rows, tokenizer, label_to_id, args.max_length) if val_rows else None
    test_dataset = DiagnosisDataset(test_rows, tokenizer, label_to_id, args.max_length) if test_rows else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        if val_dataset else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        if test_dataset else None
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, math.ceil(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, args.device)
        epoch_record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, args.device, id_to_label)
            epoch_record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["averages"]["macro_f1"],
                }
            )
            current_metric = val_metrics["averages"]["macro_f1"]
        else:
            current_metric = -train_loss

        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with (args.output_dir / "label_map.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "label_to_id": label_to_id,
                        "id_to_label": {str(k): v for k, v in id_to_label.items()},
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

    summary: dict[str, Any] = {
        "model_name": args.model_name_or_dir,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "num_labels": len(label_to_id),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
    }

    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    if test_loader is not None:
        best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(args.device)
        test_metrics = evaluate_model(best_model, test_loader, args.device, id_to_label)
        summary.update(
            {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_precision": test_metrics["averages"]["macro_precision"],
                "test_macro_recall": test_metrics["averages"]["macro_recall"],
                "test_macro_f1": test_metrics["averages"]["macro_f1"],
                "test_weighted_precision": test_metrics["averages"]["weighted_precision"],
                "test_weighted_recall": test_metrics["averages"]["weighted_recall"],
                "test_weighted_f1": test_metrics["averages"]["weighted_f1"],
            }
        )
        write_confusion_csv(
            args.output_dir / "test_confusion_matrix.csv",
            test_metrics["confusion"],
            id_to_label,
            test_metrics["labels"],
        )
        write_csv(
            args.output_dir / "test_classification_report.csv",
            test_metrics["report_rows"],
            ["label", "precision", "recall", "f1_score", "support"],
        )
        write_csv(
            args.output_dir / "test_predictions.csv",
            test_metrics["prediction_rows"],
            ["text", "true_label", "predicted_label", "correct"],
        )

    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
