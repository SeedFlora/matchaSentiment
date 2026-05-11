from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from matcha_sentiment.data import prepare_binary_dataset


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare binary Indonesian sentiment dataset.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "Matchaya Gandaria City + IKUYO (done; need recheck).xlsx"),
        help="Input Excel file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "matcha_sentiment_binary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--sheet", default="Data", help="Excel sheet name.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate cleaned texts. Default removes duplicates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, summary = prepare_binary_dataset(
        args.input,
        args.output,
        sheet_name=args.sheet,
        dedupe=not args.keep_duplicates,
    )
    summary_path = Path(args.output).with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
