from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LABEL2ID
from .text import compact_for_key, normalize_label, normalize_text


BAD_TEXT_VALUES = {"", "x", "-", ".", "n/a", "na", "none", "null"}


def load_binary_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label", "label_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["text"] = df["text"].map(normalize_text)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label_name"].map(normalize_label)
    df = df[df["label_name"].isin(LABEL2ID)]
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def prepare_binary_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sheet_name: str = "Data",
    dedupe: bool = True,
) -> tuple[pd.DataFrame, dict]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw = pd.read_excel(input_path, sheet_name=sheet_name)
    raw.columns = [normalize_text(c) for c in raw.columns]

    rows: list[dict] = []
    seen: set[str] = set()
    summary = {
        "input_file": input_path.name,
        "output_file": str(output_path).replace("\\", "/"),
        "original_rows": int(len(raw)),
        "dropped_netral": 0,
        "dropped_other_label": 0,
        "dropped_bad_text": 0,
        "dropped_duplicates": 0,
    }

    for _, row in raw.iterrows():
        label_name = normalize_label(row.get("sentimen"))
        if label_name == "Netral":
            summary["dropped_netral"] += 1
            continue
        if label_name not in LABEL2ID:
            summary["dropped_other_label"] += 1
            continue

        source_column = "perbaikan"
        text = normalize_text(row.get("perbaikan"))
        if not text:
            source_column = "textTranslated"
            text = normalize_text(row.get("textTranslated"))
        if not text:
            source_column = "text"
            text = normalize_text(row.get("text"))

        if text.lower() in BAD_TEXT_VALUES:
            summary["dropped_bad_text"] += 1
            continue

        key = compact_for_key(text)
        if dedupe and key in seen:
            summary["dropped_duplicates"] += 1
            continue
        seen.add(key)

        rows.append(
            {
                "text": text,
                "label": LABEL2ID[label_name],
                "label_name": label_name,
                "kategori": normalize_text(row.get("kategori")),
                "stars": normalize_text(row.get("stars")),
                "source_column": source_column,
            }
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    summary["kept_rows"] = int(len(df))
    summary["labels"] = df["label_name"].value_counts().to_dict()
    return df, summary
