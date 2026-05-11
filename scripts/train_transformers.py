from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from matcha_sentiment.config import (
    ARTIFACT_DIR,
    DATA_PATH,
    DEFAULT_TRANSFORMER_MODELS,
    ID2LABEL,
    LABEL2ID,
    MODEL_DIR,
)
from matcha_sentiment.data import load_binary_dataset
from matcha_sentiment.metrics import binary_metrics, report_dict
from matcha_sentiment.plots import plot_confusion, plot_roc_curve, plot_training_history


ROOT = Path(__file__).resolve().parents[1]


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", model_id)


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax(logits)[:, 1]
        preds = logits.argmax(axis=1)
        return binary_metrics(labels, preds, probs)

    return compute_metrics


def split_data(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        df,
        test_size=0.10,
        stratify=df["label"],
        random_state=random_state,
    )
    train, valid = train_test_split(
        train_valid,
        test_size=0.111111,
        stratify=train_valid["label"],
        random_state=random_state,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def save_predictions(path: Path, frame: pd.DataFrame, preds: np.ndarray, scores: np.ndarray) -> None:
    out = frame[["text", "label", "label_name"]].copy()
    out["prediction"] = preds
    out["prediction_name"] = [ID2LABEL[int(v)] for v in preds]
    out["score"] = scores
    out.to_csv(path, index=False)


def train_one_model(
    *,
    model_id: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
    fig_dir: Path,
    device_name: str,
) -> dict:
    slug = slugify_model_id(model_id)
    model_out = out_dir / slug
    model_out.mkdir(parents=True, exist_ok=True)
    set_seed(args.random_state)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    train_ds = SentimentDataset(
        train_df["text"].tolist(),
        train_df["label"].astype(int).tolist(),
        tokenizer,
        args.max_length,
    )
    valid_ds = SentimentDataset(
        valid_df["text"].tolist(),
        valid_df["label"].astype(int).tolist(),
        tokenizer,
        args.max_length,
    )
    test_ds = SentimentDataset(
        test_df["text"].tolist(),
        test_df["label"].astype(int).tolist(),
        tokenizer,
        args.max_length,
    )

    use_cuda = torch.cuda.is_available() and not args.cpu
    training_kwargs = {
        "output_dir": str(model_out / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "fp16": bool(use_cuda and args.fp16),
        "report_to": "none",
        "seed": args.random_state,
        "dataloader_num_workers": 0,
    }
    training_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_params:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(),
    )
    trainer.train()
    eval_metrics = trainer.evaluate(valid_ds)
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    probs = softmax(logits)[:, 1]
    preds = logits.argmax(axis=1)
    y_test = test_df["label"].to_numpy(dtype=int)
    test_metrics = binary_metrics(y_test, preds, probs)

    trainer.save_model(model_out / "model")
    tokenizer.save_pretrained(model_out / "model")
    log_history = trainer.state.log_history

    metrics = {
        "model_id": model_id,
        "slug": slug,
        "device": device_name,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "test_metrics": test_metrics,
    }
    (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_out / "trainer_log_history.json").write_text(
        json.dumps(log_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_predictions(model_out / "test_predictions.csv", test_df, preds, probs)
    (model_out / "classification_report.json").write_text(
        json.dumps(report_dict(y_test, preds), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_training_history(
        log_history,
        fig_dir / f"training_loss_{slug}.png",
        title=f"Training loss: {model_id}",
    )
    plot_confusion(
        y_test,
        preds,
        fig_dir / f"confusion_matrix_{slug}.png",
        title=f"Confusion matrix: {model_id}",
    )
    plot_roc_curve(
        y_test,
        probs,
        fig_dir / f"roc_auc_{slug}.png",
        title=f"ROC AUC: {model_id}",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune 5 Indonesian transformer sentiment models.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Prepared binary CSV.")
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR / "transformers"), help="Transformer artifacts.")
    parser.add_argument("--fig-dir", default=str(ARTIFACT_DIR / "figures"), help="Shared figure directory.")
    parser.add_argument("--models-dir", default=str(MODEL_DIR / "transformers"), help="Model output directory.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_TRANSFORMER_MODELS, help="Hugging Face model IDs.")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is not available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable mixed precision.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop if one model fails.")
    parser.set_defaults(fp16=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_gpu and (args.cpu or not torch.cuda.is_available()):
        raise SystemExit("CUDA GPU is required but not available inside this environment.")

    if torch.cuda.is_available() and not args.cpu:
        torch.set_float32_matmul_precision("high")
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"
    print(f"Training device: {device_name}")

    df = load_binary_dataset(args.data)
    train_df, valid_df, test_df = split_data(df, args.random_state)
    out_dir = Path(args.out_dir)
    model_root = Path(args.models_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    failures: list[dict] = []
    for model_id in args.models:
        print(f"\n=== Fine-tuning {model_id} ===")
        try:
            metrics = train_one_model(
                model_id=model_id,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                args=args,
                out_dir=model_root,
                fig_dir=fig_dir,
                device_name=device_name,
            )
            summaries.append(metrics)
        except Exception as exc:
            failure = {"model_id": model_id, "error": repr(exc)}
            failures.append(failure)
            print(f"FAILED {model_id}: {exc!r}")
            if args.stop_on_error:
                raise

    summary_path = out_dir / "results.json"
    summary_path.write_text(
        json.dumps({"results": summaries, "failures": failures}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not summaries:
        raise SystemExit(f"No transformer model finished. Details saved to {summary_path}")

    rows = []
    for item in summaries:
        row = {"model_id": item["model_id"], "slug": item["slug"]}
        row.update({f"test_{k}": v for k, v in item["test_metrics"].items()})
        rows.append(row)
    results_df = pd.DataFrame(rows).sort_values(["test_f1", "test_roc_auc", "test_accuracy"], ascending=False)
    results_df.to_csv(out_dir / "results.csv", index=False)
    best = results_df.iloc[0].to_dict()
    best_src = model_root / best["slug"] / "model"
    best_dst = MODEL_DIR / "best_transformer"
    if best_dst.exists():
        shutil.rmtree(best_dst)
    shutil.copytree(best_src, best_dst)

    best_slug = best["slug"]
    for src_name, dst_name in [
        (f"training_loss_{best_slug}.png", "transformer_best_training_loss.png"),
        (f"confusion_matrix_{best_slug}.png", "transformer_best_confusion_matrix.png"),
        (f"roc_auc_{best_slug}.png", "transformer_best_roc_auc.png"),
    ]:
        src = fig_dir / src_name
        if src.exists():
            shutil.copyfile(src, fig_dir / dst_name)

    (best_dst / "matcha_training_metadata.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nBest transformer:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"Saved best model to {best_dst}")


if __name__ == "__main__":
    main()
