import pandas as pd

SILVER_PATH = "data_lake/datasets/task1_retórica.parquet"
OUT_PATH = "data_lake/datasets/task1_gold_to_annotate.csv"
GOLD_PER_LABEL = 100  # 50 if you want faster

df = pd.read_parquet(SILVER_PATH)
print("Available columns:", df.columns.tolist())

# Sample balanced per label
gold = (
    df.groupby("label")
      .apply(lambda x: x.sample(min(len(x), GOLD_PER_LABEL), random_state=42))
      .reset_index()  # <-- bring ALL index levels back as columns
)

# If reset_index created extra columns like 'level_1', drop them
for c in ["level_0", "level_1", "index"]:
    if c in gold.columns:
        gold = gold.drop(columns=[c])

# IMPORTANT: after reset_index, you might end up with TWO 'label' columns.
# Keep the original 'label' if it exists; otherwise rename the grouped one.
label_cols = [c for c in gold.columns if c == "label"]
if len(label_cols) == 0:
    # sometimes the grouped label becomes a column named 'label' already; if not, find it:
    # (rare) could be named something like 'label_x' etc., but unlikely here.
    raise KeyError("Could not recover 'label' as a column after sampling.")

# Shuffle
gold = gold.sample(frac=1, random_state=42).reset_index(drop=True)

# Add annotation fields
gold["gold_label"] = ""
gold["annotator_notes"] = ""

# Build annotator view
gold_out = gold[["chunk_id", "label", "text", "gold_label", "annotator_notes"]].rename(
    columns={"label": "silver_label"}
)

gold_out.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print("Rows:", len(gold_out))
print("Silver label counts:\n", gold_out["silver_label"].value_counts())
