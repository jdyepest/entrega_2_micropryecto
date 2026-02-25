import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver_parquet", required=True)
    ap.add_argument("--annotated_csv", required=True)
    ap.add_argument("--gold_parquet", required=True)
    args = ap.parse_args()

    # Load data
    silver = pd.read_parquet(args.silver_parquet)
    ann = pd.read_csv(args.annotated_csv)

    # --- Validate ---
    expected_cols = ['chunk_id', 'doc_id', 'source_path', 'label', 'heading', 'n_words', 'text']
    missing = [c for c in expected_cols if c not in silver.columns]
    if missing:
        raise ValueError(f"Silver parquet missing columns: {missing}")

    if "chunk_id" not in ann.columns or "gold_label" not in ann.columns:
        raise ValueError("Annotated CSV must contain: chunk_id, gold_label")

    # Normalize ids
    silver["chunk_id"] = silver["chunk_id"].astype(str)
    ann["chunk_id"] = ann["chunk_id"].astype(str)

    # Normalize labels (8-class setup)
    ann["gold_label"] = (
        ann["gold_label"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"RES": "RESU"})
    )

    # Keep only annotated rows
    gold_ids = set(ann["chunk_id"])
    gold = silver[silver["chunk_id"].isin(gold_ids)].copy()

    # Replace label with gold label
    gold_label_map = dict(zip(ann["chunk_id"], ann["gold_label"]))
    gold["label"] = gold["chunk_id"].map(gold_label_map)

    # Enforce identical column order
    gold = gold[expected_cols]

    # Save
    gold.to_parquet(args.gold_parquet, index=False)

    print("Gold parquet created")
    print("Rows:", len(gold))
    print("Columns:", gold.columns.tolist())

if __name__ == "__main__":
    main()