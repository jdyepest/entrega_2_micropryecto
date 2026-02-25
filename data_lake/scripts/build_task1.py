from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ----------------------------
# 1) Heading patterns -> labels
# ----------------------------
# You can expand/adjust these based on your corpus.
HEADING_TO_LABEL = [

    # Introduction
    (re.compile(r"^\s*(introducci[oó]n|introduction)\s*[:\-]?\s*$", re.IGNORECASE), "INTRO"),

    # Background / Related work / State of the art
    (re.compile(r"^\s*(antecedentes|marco te[oó]rico|estado del arte|trabajos relacionados|related work)\s*[:\-]?\s*$", re.IGNORECASE), "BACK"),

    # Methodology
    (re.compile(r"^\s*(metodolog[ií]a|materiales y m[eé]todos|m[eé]todos|methodology|methods)\s*[:\-]?\s*$", re.IGNORECASE), "METH"),

    # Results
    (re.compile(r"^\s*(resultados|results)\s*[:\-]?\s*$", re.IGNORECASE), "RESU"),

    # Discussion
    (re.compile(r"^\s*(discusi[oó]n|discussion)\s*[:\-]?\s*$", re.IGNORECASE), "DISC"),

    # Contributions (sometimes explicit)
    (re.compile(r"^\s*(contribuci[oó]n(es)?|aport(es)?|contribution(s)?)\s*[:\-]?\s*$", re.IGNORECASE), "CONTR"),

    # Limitations
    (re.compile(r"^\s*(limitaci[oó]n(es)?|amenazas a la validez|threats to validity|limitations)\s*[:\-]?\s*$", re.IGNORECASE), "LIM"),

    # Conclusions
    (re.compile(r"^\s*(conclusiones|conclusion(es)?|concluding remarks)\s*[:\-]?\s*$", re.IGNORECASE), "CONC"),
]

# Some corpora have numbered headings: "1. Introducción", "2 Metodología", etc.
NUMBERED_HEADING = re.compile(r"^\s*(\d+(\.\d+)*)\s*[)\.\-]?\s*(.+?)\s*$")

# ----------------------------
# 2) Chunking config
# ----------------------------
MIN_WORDS = 250
MAX_WORDS = 1000
TARGET_WORDS = 600  # used for splitting long sections more evenly

def normalize_line(line: str) -> str:
    # Keep it simple: trim and collapse spaces
    return re.sub(r"\s+", " ", line).strip()

def heading_label(line: str) -> tuple[str | None, str | None]:
    """
    Returns (label, heading_text) if line looks like a heading, else (None, None).
    """
    raw = line.strip()
    if not raw:
        return None, None

    # Try numbered heading: "1. Introducción"
    m = NUMBERED_HEADING.match(raw)
    candidate = m.group(3) if m else raw

    # Headings are usually short
    if len(candidate) > 150:
        return None, None

    for rx, label in HEADING_TO_LABEL:
        if rx.match(candidate):
            return label, candidate

    return None, None

def split_into_sections(text: str) -> list[dict]:
    """
    Split full doc text into labeled sections using headings.
    Returns list of {label, heading, section_text}.
    Unlabeled text before first heading is ignored (you can change that).
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current = None  # dict with label/heading/lines

    for line in lines:
        norm = normalize_line(line)
        label, heading = heading_label(norm)

        if label:
            # Close previous section
            if current and current["lines"]:
                current["section_text"] = "\n".join(current["lines"]).strip()
                sections.append(current)

            # Start new section
            current = {"label": label, "heading": heading, "lines": []}
            continue

        # regular content line
        if current is not None:
            if norm:  # skip empty lines (optional)
                current["lines"].append(norm)

    # Close last
    if current and current["lines"]:
        current["section_text"] = "\n".join(current["lines"]).strip()
        sections.append(current)

    # Filter super-short sections (often noise)
    cleaned = []
    for s in sections:
        if len(s["section_text"].split()) >= 50:
            cleaned.append(s)
    return cleaned

def chunk_words(words: list[str], min_w: int, max_w: int) -> list[list[str]]:
    """
    Chunk a list of words into chunks within [min_w, max_w] where possible.
    """
    chunks: list[list[str]] = []
    i = 0
    n = len(words)

    while i < n:
        remaining = n - i
        if remaining <= max_w:
            # last chunk
            if remaining >= min_w:
                chunks.append(words[i:])
            # else: too short tail -> drop it (or merge backward; keep simple for baseline)
            break

        # take a target chunk
        take = min(max_w, max(min_w, TARGET_WORDS))
        chunks.append(words[i:i + take])
        i += take

    return chunks

def iter_parquet_files(root: Path) -> Iterator[Path]:
    # supports either single folder of parts or nested chunk folders
    for p in sorted(root.rglob("*.parquet")):
        yield p

def build_task1(
    clean_parquet_root: str,
    out_path: str,
    per_label_cap: int = 5000,
    min_quality_score: int = 0,
    require_sections: bool = False,
    max_docs: int | None = None,
):
    """
    Build Task1 dataset from clean parquet shards.

    Filters you can toggle:
    - min_quality_score: only keep docs with quality_score >= X
    - require_sections: only keep docs where has_sections=True (if present)
    - per_label_cap: cap examples per label to balance-ish

    Output: a single parquet with chunks and labels.
    """
    root = Path(clean_parquet_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {lbl: 0 for _, lbl in HEADING_TO_LABEL}
    rows = []

    docs_seen = 0
    for fp in tqdm(list(iter_parquet_files(root)), desc="Reading clean_parquet"):
        table = pq.read_table(fp)
        df = table.to_pandas()

        for _, r in df.iterrows():
            if max_docs is not None and docs_seen >= max_docs:
                break
            docs_seen += 1

            # Optional filters (useful later when scaling)
            if "quality_score" in df.columns and int(r.get("quality_score", 0)) < min_quality_score:
                continue
            if require_sections and "has_sections" in df.columns and not bool(r.get("has_sections", False)):
                continue

            text = r["text"]
            doc_id = r.get("doc_id")
            path = r.get("path")

            sections = split_into_sections(text)
            if not sections:
                continue

            for s in sections:
                label = s["label"]

                # Skip label if already enough examples (balance cap)
                if label in counts and counts[label] >= per_label_cap:
                    continue

                words = s["section_text"].split()
                chunks = chunk_words(words, MIN_WORDS, MAX_WORDS)

                for ch_words in chunks:
                    if label in counts and counts[label] >= per_label_cap:
                        break

                    chunk_text = " ".join(ch_words).strip()
                    if not chunk_text:
                        continue

                    rows.append({
                        "chunk_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "source_path": path,
                        "label": label,
                        "heading": s["heading"],
                        "n_words": len(ch_words),
                        "text": chunk_text,
                    })
                    if label in counts:
                        counts[label] += 1

        if max_docs is not None and docs_seen >= max_docs:
            break

        # Early stop if all labels reached cap
        if all(counts.get(lbl, 0) >= per_label_cap for _, lbl in HEADING_TO_LABEL):
            break

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print("Saved:", out_path)
    print("Counts per label:", counts)
    print("Total rows:", len(out_df))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", required=True, help="Root folder with clean parquet shards")
    ap.add_argument("--out", required=True, help="Output parquet path for task1 dataset")
    ap.add_argument("--per-label-cap", type=int, default=5000)
    ap.add_argument("--min-quality-score", type=int, default=0)
    ap.add_argument("--require-sections", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    build_task1(
        clean_parquet_root=args.clean_root,
        out_path=args.out,
        per_label_cap=args.per_label_cap,
        min_quality_score=args.min_quality_score,
        require_sections=args.require_sections,
        max_docs=args.max_docs,
    )