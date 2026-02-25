import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--silver_parquet",
        default="data_lake/datasets/task2_contributions_silver.parquet",
        help="Parquet silver de Task2 (salida de build_task2.py)",
    )
    ap.add_argument(
        "--out_csv",
        default="data_lake/datasets/task2_gold_to_annotate.csv",
        help="CSV para anotación manual",
    )
    ap.add_argument("--pos_n", type=int, default=500, help="Cantidad de positivos a anotar")
    ap.add_argument("--neg_n", type=int, default=500, help="Cantidad de negativos a anotar")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.silver_parquet)
    required = {"fragment_id", "doc_id", "source_path", "rhetorical_label", "is_contribution", "n_words", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Silver parquet missing required columns: {sorted(missing)}")

    df = df.copy()
    df["fragment_id"] = df["fragment_id"].astype(str)
    df["rhetorical_label"] = df["rhetorical_label"].astype(str).str.upper().str.strip().replace({"RES": "RESU"})
    df["is_contribution"] = df["is_contribution"].astype(bool)

    pos = df[df["is_contribution"]].sample(frac=1, random_state=args.seed).head(args.pos_n)
    neg = df[~df["is_contribution"]].sample(frac=1, random_state=args.seed + 1).head(args.neg_n)

    gold = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Campos gold para anotación
    gold["gold_is_contribution"] = ""
    gold["gold_rhetorical_label"] = ""  # confirmación/ajuste de etiqueta retórica
    gold["annotator_notes"] = ""

    out = gold[
        [
            "fragment_id",
            "doc_id",
            "source_path",
            "rhetorical_label",
            "is_contribution",
            "n_words",
            "text",
            "gold_is_contribution",
            "gold_rhetorical_label",
            "annotator_notes",
        ]
    ].rename(
        columns={
            "rhetorical_label": "silver_rhetorical_label",
            "is_contribution": "silver_is_contribution",
        }
    )

    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print("Rows:", len(out))
    print("Silver contribution counts:\n", out["silver_is_contribution"].value_counts())
    print("Silver rhetorical label counts:\n", out["silver_rhetorical_label"].value_counts())


if __name__ == "__main__":
    main()
