from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from .config import ID2LABEL, STOPWORDS
from .text import tokenize


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion(y_true, y_pred, path: str | Path, *, title: str) -> Path:
    path = ensure_parent(path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.8, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID2LABEL[0], ID2LABEL[1]],
        yticklabels=[ID2LABEL[0], ID2LABEL[1]],
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_roc_curve(y_true, y_score, path: str | Path, *, title: str) -> Path | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    path = ensure_parent(path)
    plt.figure(figsize=(5.8, 4.8))
    RocCurveDisplay.from_predictions(y_true, y_score, name="Positif")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_training_history(log_history: list[dict], path: str | Path, *, title: str) -> Path | None:
    train = [
        (item.get("epoch", item.get("step")), item["loss"])
        for item in log_history
        if "loss" in item and item.get("loss") is not None
    ]
    evals = [
        (item.get("epoch", item.get("step")), item["eval_loss"])
        for item in log_history
        if "eval_loss" in item and item.get("eval_loss") is not None
    ]
    if not train and not evals:
        return None

    path = ensure_parent(path)
    plt.figure(figsize=(7.4, 4.6))
    if train:
        xs, ys = zip(*train)
        plt.plot(xs, ys, marker="o", label="train loss")
    if evals:
        xs, ys = zip(*evals)
        plt.plot(xs, ys, marker="o", label="eval loss")
    plt.xlabel("Epoch/step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_top_words(top_words: pd.DataFrame, path: str | Path, *, title: str) -> Path | None:
    if top_words.empty:
        return None
    path = ensure_parent(path)
    frame = top_words.copy()
    frame["signed_weight"] = np.where(
        frame["label_name"].eq("Positif"),
        frame["weight"].abs(),
        -frame["weight"].abs(),
    )
    frame = pd.concat(
        [
            frame[frame["label_name"].eq("Negatif")].sort_values("signed_weight").head(20),
            frame[frame["label_name"].eq("Positif")].sort_values("signed_weight", ascending=False).head(20),
        ]
    )
    plt.figure(figsize=(8, 8))
    colors = frame["label_name"].map({"Negatif": "#c44536", "Positif": "#2f8f46"})
    plt.barh(frame["term"], frame["signed_weight"], color=colors)
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.xlabel("Bobot terhadap kelas")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_wordcloud(texts: list[str], path: str | Path, *, title: str | None = None) -> Path | None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None

    tokens: list[str] = []
    for text in texts:
        tokens.extend(tokenize(text, remove_stopwords=True, min_len=3))
    tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return None

    path = ensure_parent(path)
    wordcloud = WordCloud(
        width=1100,
        height=650,
        background_color="white",
        colormap="viridis",
        max_words=120,
        collocations=True,
        stopwords=STOPWORDS,
        random_state=42,
    ).generate(" ".join(tokens))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path
