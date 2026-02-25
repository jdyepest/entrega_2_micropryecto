import argparse

import pandas as pd


DEFAULT_IN = "data_lake/datasets/task2_contributions_silver.parquet"
DEFAULT_OUT = "data_lake/datasets/task2_contributions_silver_binary.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", default=DEFAULT_IN, help="Input Task2 silver parquet")
    ap.add_argument("--out_parquet", default=DEFAULT_OUT, help="Output parquet (binary target only)")
    ap.add_argument(
        "--keep_rhetorical_label",
        action="store_true",
        help="Keep rhetorical_label column as feature (default: drop it so dataset is purely binary).",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    required = {"fragment_id", "text", "is_contribution"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input parquet: {sorted(missing)}")

    out = df.copy()

    # Force strict binary
    out["is_contribution"] = out["is_contribution"].astype(bool)

    # Drop non-binary label-ish fields by default
    if not args.keep_rhetorical_label and "rhetorical_label" in out.columns:
        out = out.drop(columns=["rhetorical_label"])

    # Keep a stable/clean column order when possible
    preferred = [
        "fragment_id",
        "source_chunk_id",
        "doc_id",
        "source_path",
        "heading",
        "n_words",
        "text",
        "is_contribution",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    out.to_parquet(args.out_parquet, index=False)

    counts = out["is_contribution"].value_counts(dropna=False).to_dict()
    print("Saved:", args.out_parquet)
    print("Rows:", len(out))
    print("Binary counts (is_contribution):", counts)
    print("Columns:", out.columns.tolist())


if __name__ == "__main__":
    main()

