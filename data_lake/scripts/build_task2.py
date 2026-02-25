from __future__ import annotations

import argparse
import re
import uuid
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


REQUIRED_SILVER_COLS = ["chunk_id", "doc_id", "source_path", "label", "heading", "text"]


@dataclass(frozen=True)
class BuildConfig:
    silver_parquet: Path
    out_parquet: Path
    n_pos: int
    n_neg: int
    neg_min_words: int
    neg_max_words: int
    seed: int


def compile_contribution_patterns() -> list[re.Pattern]:
    """
    Patrones para detectar formulaciones explícitas de contribución científica.
    Este conjunto es deliberadamente conservador (silver).
    """
    patterns = [
        r"\b(en\s+este\s+trabajo\s+)?(presentamos|proponemos|introducimos|planteamos)\b",
        r"\b(nuestra|nuestro)\s+(propuesta|aporte|contribuci[oó]n)\b",
        r"\b(este\s+trabajo|este\s+art[ií]culo)\s+(propone|presenta|introduce)\b",
        r"\b(aportamos|contribuimos)\b",
        r"\b(a\s+diferencia\s+de)\b",
        r"\b(primer(a)?\s+vez|por\s+primera\s+vez)\b",
        r"\b(nuevo|nueva)\s+(m[eé]todo|enfoque|algoritmo|sistema|marco|modelo|recurso|corpus|dataset)\b",
        r"\b(ponemos\s+a\s+disposici[oó]n|liberamos|publicamos)\b",
    ]
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def looks_like_contribution(text: str, regexes: list[re.Pattern]) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(rx.search(t) for rx in regexes)


def validate_silver_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SILVER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Silver parquet missing columns: {missing}. Required: {REQUIRED_SILVER_COLS}")


def sample_diverse(df: pd.DataFrame, n: int, seed: int, key_col: str = "doc_id") -> pd.DataFrame:
    """
    Muestreo simple con diversidad por documento: prioriza 1 fragmento por doc,
    y luego rellena con el resto si n excede #docs.
    """
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()

    shuffled = df.sample(frac=1, random_state=seed)
    first_per_doc = shuffled.drop_duplicates(subset=[key_col], keep="first")
    picked = first_per_doc.head(n)
    if len(picked) >= n:
        return picked

    remaining = shuffled[~shuffled.index.isin(picked.index)]
    extra = remaining.head(n - len(picked))
    return pd.concat([picked, extra], axis=0)


def make_negative_window(text: str, min_words: int, max_words: int, seed: int, max_tries: int = 5) -> tuple[str, int] | None:
    """
    Crea una ventana (sub-fragmento) de longitud en [min_words, max_words] desde un texto largo.
    Devuelve (window_text, n_words) o None si no es posible.
    """
    words = (text or "").split()
    if len(words) < min_words:
        return None

    rng = random.Random(seed ^ len(words))
    for _ in range(max_tries):
        target_len = rng.randint(min_words, min(max_words, len(words)))
        if target_len < min_words:
            continue

        start = rng.randint(0, max(0, len(words) - target_len))
        window_words = words[start : start + target_len]
        window_text = " ".join(window_words).strip()
        if window_text:
            return window_text, len(window_words)

    return None


def build_task2(cfg: BuildConfig) -> None:
    cfg.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.silver_parquet)
    validate_silver_df(df)

    df = df.copy()
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["source_path"] = df["source_path"].astype(str)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["heading"] = df["heading"].astype(str)
    df["text"] = df["text"].astype(str)
    if "n_words" not in df.columns or df["n_words"].isna().any():
        df["n_words"] = df["text"].str.split().str.len()

    regexes = compile_contribution_patterns()
    df["is_contribution_match"] = df["text"].apply(lambda t: looks_like_contribution(t, regexes))

    # -------------------------
    # Positivos (contribución)
    # -------------------------
    pos_strict = df[df["is_contribution_match"]].copy()
    pos_strict["pos_reason"] = "pattern"

    pos_picked = sample_diverse(pos_strict, cfg.n_pos, cfg.seed)

    # Si no alcanza, completar desde label CONTR (aunque no haya match)
    if len(pos_picked) < cfg.n_pos:
        remaining_n = cfg.n_pos - len(pos_picked)
        already = set(pos_picked["chunk_id"].tolist())
        pos_relaxed = df[(df["label"] == "CONTR") & (~df["chunk_id"].isin(already))].copy()
        pos_relaxed["pos_reason"] = "label_CONTR"
        pos_fill = sample_diverse(pos_relaxed, remaining_n, cfg.seed + 1)
        pos_picked = pd.concat([pos_picked, pos_fill], axis=0)

    pos_picked = pos_picked.head(cfg.n_pos).copy()

    # -------------------------
    # Negativos (no contribución)
    # -------------------------
    pos_ids = set(pos_picked["chunk_id"].tolist())
    neg_candidates = df[~df["chunk_id"].isin(pos_ids)].copy()
    neg_candidates = neg_candidates[~neg_candidates["is_contribution_match"]].copy()
    neg_candidates = neg_candidates[neg_candidates["n_words"] >= cfg.neg_min_words].copy()

    neg_candidates = neg_candidates.sample(frac=1, random_state=cfg.seed + 7)

    neg_rows = []
    neg_seen = 0
    for _, r in tqdm(neg_candidates.iterrows(), total=len(neg_candidates), desc="Selecting negatives"):
        if neg_seen >= cfg.n_neg:
            break

        raw_text = str(r["text"])
        n_words = int(r["n_words"])

        if cfg.neg_min_words <= n_words <= cfg.neg_max_words:
            window_text = raw_text
            window_n = n_words
        else:
            window = make_negative_window(
                raw_text,
                min_words=cfg.neg_min_words,
                max_words=min(cfg.neg_max_words, n_words),
                seed=cfg.seed + neg_seen,
            )
            if window is None:
                continue
            window_text, window_n = window

        # Re-chequear por seguridad que el window no tenga patrones de contribución
        if looks_like_contribution(window_text, regexes):
            continue

        neg_rows.append(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "source_path": r["source_path"],
                "label": r["label"],
                "heading": r["heading"],
                "text": window_text,
                "n_words": window_n,
            }
        )
        neg_seen += 1

    neg_picked = pd.DataFrame(neg_rows)
    if len(neg_picked) < cfg.n_neg:
        raise RuntimeError(
            f"No hay suficientes negativos. Se pidieron {cfg.n_neg} y se obtuvieron {len(neg_picked)}. "
            "Ajusta --neg-min-words/--neg-max-words o expande el silver."
        )

    # -------------------------
    # Dataset final (Task2)
    # -------------------------
    def to_rows(frame: pd.DataFrame, is_pos: bool) -> list[dict]:
        rows: list[dict] = []
        for _, r in frame.iterrows():
            rows.append(
                {
                    "fragment_id": str(uuid.uuid4()),
                    "source_chunk_id": str(r["chunk_id"]),
                    "doc_id": str(r["doc_id"]),
                    "source_path": str(r["source_path"]),
                    "heading": str(r["heading"]),
                    "rhetorical_label": str(r["label"]),
                    "is_contribution": bool(is_pos),
                    "n_words": int(r["n_words"]),
                    "text": str(r["text"]),
                }
            )
        return rows

    out_rows = []
    out_rows.extend(to_rows(pos_picked, True))
    out_rows.extend(to_rows(neg_picked, False))

    out_df = pd.DataFrame(out_rows).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    out_df.to_parquet(cfg.out_parquet, index=False)

    print("Saved:", cfg.out_parquet)
    print("Positives:", int(out_df["is_contribution"].sum()))
    print("Negatives:", int((~out_df["is_contribution"]).sum()))
    print("Total:", len(out_df))
    print("Positive reasons:", dict(pos_picked.get("pos_reason", pd.Series(dtype=str)).value_counts()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--silver-parquet",
        default="data_lake/datasets/task1_retórica.parquet",
        help="Silver Task1 parquet con columnas: chunk_id,doc_id,source_path,label,heading,n_words,text",
    )
    ap.add_argument(
        "--out",
        default="data_lake/datasets/task2_contributions_silver.parquet",
        help="Output parquet para Task2 (contribuciones)",
    )
    ap.add_argument("--n-pos", type=int, default=1000, help="Cantidad de fragmentos positivos (con contribución)")
    ap.add_argument("--n-neg", type=int, default=1000, help="Cantidad de fragmentos negativos (sin contribución)")
    ap.add_argument("--neg-min-words", type=int, default=250)
    ap.add_argument("--neg-max-words", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = BuildConfig(
        silver_parquet=Path(args.silver_parquet),
        out_parquet=Path(args.out),
        n_pos=int(args.n_pos),
        n_neg=int(args.n_neg),
        neg_min_words=int(args.neg_min_words),
        neg_max_words=int(args.neg_max_words),
        seed=int(args.seed),
    )
    build_task2(cfg)


if __name__ == "__main__":
    main()
