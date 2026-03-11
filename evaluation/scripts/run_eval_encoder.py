#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import eval_gold as eg


def _default_output(repo_root: Path, encoder_variant: str) -> Path:
    name = f"encoder_{encoder_variant}.json"
    return repo_root / "artifacts" / "eval_results" / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa Golden Set con encoder (Task1+Task2).")
    parser.add_argument("--encoder-variant", default="roberta", choices=["roberta", "scibert"])
    parser.add_argument("--batch-task1", type=int, default=200)
    parser.add_argument("--batch-task2", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--mlflow-experiment", default="")
    args = parser.parse_args()

    if args.mlflow_experiment:
        os.environ["MLFLOW_EXPERIMENT"] = args.mlflow_experiment

    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.output) if args.output else _default_output(repo_root, args.encoder_variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[Encoder] Iniciando evaluación")
    print(f"[Encoder] encoder_variant={args.encoder_variant}")
    print(f"[Encoder] batch_task1={args.batch_task1} batch_task2={args.batch_task2} limit={args.limit}")
    print(f"[Encoder] MLFLOW_TRACKING_URI={os.environ.get('MLFLOW_TRACKING_URI')}")

    res_t1 = eg.evaluate_task1(
        "encoder",
        encoder_variant=args.encoder_variant,
        batch_size=args.batch_task1,
        limit=args.limit,
    )
    res_t2 = eg.evaluate_task2(
        "encoder",
        encoder_variant=args.encoder_variant,
        batch_size=args.batch_task2,
        limit=args.limit,
    )

    payload = {
        "meta": {
            "model_label": f"encoder-{args.encoder_variant}",
            "run_at": datetime.now(timezone.utc).isoformat(),
            "params": {
                "encoder_variant": args.encoder_variant,
                "batch_task1": args.batch_task1,
                "batch_task2": args.batch_task2,
                "limit": args.limit,
            },
        },
        "results": [res_t1.__dict__, res_t2.__dict__],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] Guardado: {out_path}")

    try:
        import pandas as pd

        df = pd.DataFrame([res_t1.__dict__, res_t2.__dict__])
        print(df[["task", "accuracy", "precision", "recall", "f1", "total_time_s", "time_per_doc_s"]])
    except Exception:
        pass


if __name__ == "__main__":
    main()
