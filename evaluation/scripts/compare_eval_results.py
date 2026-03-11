#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara resultados JSON de evals.")
    parser.add_argument("inputs", nargs="*", help="Archivos JSON de resultados.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    if args.inputs:
        files = [Path(p) for p in args.inputs]
    else:
        repo_root = Path(__file__).resolve().parents[2]
        files = sorted((repo_root / "artifacts" / "eval_results").glob("*.json"))

    print(f"[Compare] Archivos encontrados: {len(files)}")
    rows = []
    for path in files:
        payload = _load_payload(path)
        meta = payload.get("meta") or {}
        model_label = meta.get("model_label") or path.stem
        for res in payload.get("results") or []:
            row = {"model": model_label, **res}
            rows.append(row)

    if not rows:
        print("No hay resultados para comparar.")
        return

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        cols = ["model", "task", "accuracy", "precision", "recall", "f1", "total_time_s", "time_per_doc_s"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols])
        if args.output_json:
            Path(args.output_json).write_text(df.to_json(orient="records", indent=2, force_ascii=False))
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
    except Exception:
        print(rows)


if __name__ == "__main__":
    main()
