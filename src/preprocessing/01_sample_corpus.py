#!/usr/bin/env python3
"""Sample a fixed number of .txt files from datos/core using reservoir sampling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import time
from pathlib import Path


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_txt_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".txt"):
                yield Path(dirpath) / name


def reservoir_sample(paths, k: int, seed: int):
    rnd = random.Random(seed)
    sample = []
    for i, p in enumerate(paths, start=1):
        if i <= k:
            sample.append(p)
        else:
            j = rnd.randint(1, i)
            if j <= k:
                sample[j - 1] = p
    return sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datos/core", help="Root folder with .txt files")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="artifacts/samples/sample_100.csv")
    args = parser.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sample = reservoir_sample(iter_txt_files(root), args.sample_size, args.seed)

    rows = []
    for p in sample:
        rel = p.relative_to(root)
        stat = p.stat()
        rows.append({
            "doc_id": rel.as_posix(),
            "path": p.as_posix(),
            "size_bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
            "sha1": sha1_file(p),
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "path", "size_bytes", "mtime", "sha1"])
        w.writeheader()
        w.writerows(rows)

    elapsed = time.time() - t0
    print(f"Sampled {len(rows)} files in {elapsed:.1f}s -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
