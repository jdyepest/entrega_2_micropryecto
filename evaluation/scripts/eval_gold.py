#!/usr/bin/env python3
"""Evaluate models on golden sets for Task1 and Task2 with MLflow logging."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # Optional dependency; continue if dotenv is not available.
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "app" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.segmentation import analyze_segments  # noqa: E402
from services.contributions import analyze_contributions  # noqa: E402


TASK1_PATH = REPO_ROOT / "data_lake" / "datasets" / "task1_gold_labeled.csv"
TASK2_PATH = REPO_ROOT / "data_lake" / "datasets" / "task2_gold_labeled.csv"


@dataclass
class EvalResult:
    task: str
    model: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    total_time_s: float
    time_per_doc_s: float


def _sim_config_for_model(model: str, task: str) -> dict[str, float]:
    """
    Simulaciones por modelo (medias y desviaciones estándar).
    Métricas: alrededor de la media indicada por el usuario.
    Tiempo: segundos por documento.
    """
    m = (model or "").strip().lower()
    if m == "encoder":
        return {"metric_mean": 0.36, "metric_std": 0.04, "time_mean": 20.0, "time_std": 3.0}
    if m == "llm":
        llm_mean = 0.43 if task == "task1" else 0.65
        return {"metric_mean": llm_mean, "metric_std": 0.05, "time_mean": 11.0, "time_std": 2.0}
    # api
    return {"metric_mean": 0.95, "metric_std": 0.02, "time_mean": 5.0, "time_std": 1.0}


def _sim_enabled() -> bool:
    flag = (os.environ.get("SIM_EVAL") or "").strip().lower()
    legacy = (os.environ.get("MOCK_EVAL") or "").strip().lower()
    return flag in {"1", "true", "yes", "y"} or legacy in {"1", "true", "yes", "y"}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _maybe_sim_eval(task: str, model: str, n: int) -> EvalResult | None:
    if not _sim_enabled():
        return None
    cfg = _sim_config_for_model(model, task)
    seed_env = os.environ.get("SIM_EVAL_SEED")
    seed = int(seed_env) if seed_env and seed_env.isdigit() else None
    rng = random.Random(seed)

    def metric() -> float:
        return _clamp01(rng.gauss(cfg["metric_mean"], cfg["metric_std"]))

    accuracy = metric()
    precision = metric()
    recall = metric()
    f1 = metric()

    time_per_doc = max(0.1, rng.gauss(cfg["time_mean"], cfg["time_std"]))
    total_time = float(round(time_per_doc * max(n, 1), 4))

    return EvalResult(
        task=task,
        model=model,
        n=n,
        accuracy=float(round(accuracy, 4)),
        precision=float(round(precision, 4)),
        recall=float(round(recall, 4)),
        f1=float(round(f1, 4)),
        total_time_s=total_time,
        time_per_doc_s=float(round(time_per_doc, 4)),
    )


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    # Collapse newlines to keep one paragraph per row.
    return re.sub(r"\s+", " ", str(text)).strip()


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    # Handle NaN from pandas
    try:
        import math

        if isinstance(label, float) and math.isnan(label):
            return ""
    except Exception:
        pass
    lbl = str(label).strip().upper()
    return "RES" if lbl == "RESU" else lbl


def _batch_iter(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _llm_provider() -> str:
    return (os.environ.get("LOCAL_LLM_PROVIDER") or "ollama").strip().lower()


def _llm_max_prompt_chars(provider: str) -> int | None:
    raw = (os.environ.get("LLM_MAX_PROMPT_CHARS") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if provider == "openrouter":
        # 80k tokens aprox; usamos un tope conservador por caracteres.
        return 240_000
    return None


def _build_batches(
    rows: list[dict],
    batch_size: int,
    max_chars: int | None,
) -> list[list[dict]]:
    if not rows:
        return []
    if max_chars is None:
        return list(_batch_iter(rows, batch_size))

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0

    for row in rows:
        text = str(row.get("text") or "")
        # Estimación simple de tamaño por fila
        row_len = len(text) + 50

        if current and (len(current) >= batch_size or current_len + row_len > max_chars):
            batches.append(current)
            current = []
            current_len = 0

        current.append(row)
        current_len += row_len

    if current:
        batches.append(current)

    return batches


def _log_mlflow(params: dict, metrics: dict, tags: dict | None = None) -> None:
    try:
        import mlflow
    except Exception as e:
        print(f"[MLflow] No disponible (pip install mlflow boto3). Detalle: {e}")
        return

    exp_name = os.environ.get("MLFLOW_EXPERIMENT") or "golden-set-eval"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as e:  # noqa: BLE001
            print(f"[MLflow] No se pudo setear MLFLOW_TRACKING_URI={tracking_uri}. Detalle: {e}")
    mlflow.set_experiment(exp_name)
    print(f"[MLflow] Tracking URI: {tracking_uri or 'default'} | Experiment: {exp_name}")

    with mlflow.start_run(run_name=params.get("run_name")):
        for k, v in params.items():
            if k == "run_name":
                continue
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if tags:
            mlflow.set_tags(tags)


def evaluate_task1(
    model: str,
    encoder_variant: str = "roberta",
    batch_size: int = 200,
    limit: int | None = None,
    use_mlflow: bool = True,
) -> EvalResult:
    df = pd.read_csv(TASK1_PATH)
    if limit:
        df = df.head(limit)

    rows = [
        {
            "text": _clean_text(t),
            "gold_label": _normalize_label(lbl),
        }
        for t, lbl in zip(df["text"], df["gold_label"], strict=False)
    ]

    y_true: list[str] = []
    y_pred: list[str] = []

    maybe = _maybe_sim_eval("task1", model, n=len(rows))
    if maybe:
        if use_mlflow:
            _log_mlflow(
                params={
                    "run_name": f"task1-{model}",
                    "task": "task1",
                    "model": model,
                    "encoder_variant": encoder_variant,
                    "batch_size": batch_size,
                    "n_samples": len(rows),
                    "simulated": True,
                },
                metrics={
                    "accuracy": maybe.accuracy,
                    "precision": maybe.precision,
                    "recall": maybe.recall,
                    "f1": maybe.f1,
                    "total_time_s": maybe.total_time_s,
                    "time_per_doc_s": maybe.time_per_doc_s,
                },
                tags={"dataset": TASK1_PATH.name},
            )
        return maybe

    provider = _llm_provider() if model == "llm" else "n/a"
    if model == "llm" and provider == "ollama":
        batch_size = 1  # Ollama sin batch
    max_chars = _llm_max_prompt_chars(provider) if model == "llm" else None
    batches = _build_batches(rows, batch_size, max_chars)

    started = time.perf_counter()
    log_every = int(os.environ.get("EVAL_LOG_EVERY") or "10")
    total_batches = len(batches)
    print(
        f"[Task1] model={model} encoder={encoder_variant} rows={len(rows)} "
        f"batch_size={batch_size} batches={total_batches} provider={provider} max_chars={max_chars}"
    )
    for b_idx, batch in enumerate(batches, start=1):
        text_blob = "\n\n".join([r["text"] for r in batch])
        result = analyze_segments(text_blob, model, encoder_variant=encoder_variant)
        segments = result.get("segments") or []

        if len(segments) < len(batch):
            # Skip malformed batch
            continue

        for i, item in enumerate(batch):
            pred = segments[i].get("label")
            y_true.append(item["gold_label"])
            y_pred.append(_normalize_label(pred))
        if log_every > 0 and (b_idx == 1 or b_idx % log_every == 0 or b_idx == total_batches):
            print(f"[Task1] progreso {b_idx}/{total_batches} batches | y_true={len(y_true)}")

    elapsed = time.perf_counter() - started

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    result = EvalResult(
        task="task1",
        model=model,
        n=len(y_true),
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        total_time_s=float(round(elapsed, 4)),
        time_per_doc_s=float(round(elapsed / max(len(y_true), 1), 6)),
    )

    if use_mlflow:
        _log_mlflow(
            params={
                "run_name": f"task1-{model}",
                "task": "task1",
                "model": model,
                "encoder_variant": encoder_variant,
                "batch_size": batch_size,
                "n_samples": len(y_true),
            },
            metrics={
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "total_time_s": result.total_time_s,
                "time_per_doc_s": result.time_per_doc_s,
            },
            tags={"dataset": TASK1_PATH.name},
        )

    return result


def evaluate_task2(
    model: str,
    encoder_variant: str = "roberta",
    batch_size: int = 100,
    limit: int | None = None,
    use_mlflow: bool = True,
) -> EvalResult:
    df = pd.read_csv(TASK2_PATH)
    if limit:
        df = df.head(limit)

    rows = [
        {
            "text": _clean_text(t),
            "gold_is_contribution": _safe_bool(lbl),
            "gold_rhet_label": _normalize_label(lbl_r),
        }
        for t, lbl, lbl_r in zip(
            df["text"], df["gold_is_contribution"], df["gold_rhetorical_label"], strict=False
        )
    ]

    y_true: list[bool] = []
    y_pred: list[bool] = []

    maybe = _maybe_sim_eval("task2", model, n=len(rows))
    if maybe:
        if use_mlflow:
            _log_mlflow(
                params={
                    "run_name": f"task2-{model}",
                    "task": "task2",
                    "model": model,
                    "encoder_variant": encoder_variant,
                    "batch_size": batch_size,
                    "n_samples": len(rows),
                    "simulated": True,
                },
                metrics={
                    "accuracy": maybe.accuracy,
                    "precision": maybe.precision,
                    "recall": maybe.recall,
                    "f1": maybe.f1,
                    "total_time_s": maybe.total_time_s,
                    "time_per_doc_s": maybe.time_per_doc_s,
                },
                tags={"dataset": TASK2_PATH.name},
            )
        return maybe

    provider = _llm_provider() if model == "llm" else "n/a"
    if model == "llm" and provider == "ollama":
        batch_size = 1  # Ollama sin batch
    max_chars = _llm_max_prompt_chars(provider) if model == "llm" else None
    batches = _build_batches(rows, batch_size, max_chars)

    started = time.perf_counter()
    log_every = int(os.environ.get("EVAL_LOG_EVERY") or "10")
    total_batches = len(batches)
    print(
        f"[Task2] model={model} encoder={encoder_variant} rows={len(rows)} "
        f"batch_size={batch_size} batches={total_batches} provider={provider} max_chars={max_chars}"
    )
    for b_idx, batch in enumerate(batches, start=1):
        segments = [
            {
                "paragraph_index": i,
                "text": r["text"],
                "label": r["gold_rhet_label"],
            }
            for i, r in enumerate(batch)
        ]
        result = analyze_contributions(segments, model, encoder_variant=encoder_variant)
        frags = result.get("fragments") or []

        if len(frags) < len(batch):
            continue

        for i, item in enumerate(batch):
            pred = bool(frags[i].get("is_contribution"))
            y_true.append(item["gold_is_contribution"])
            y_pred.append(pred)
        if log_every > 0 and (b_idx == 1 or b_idx % log_every == 0 or b_idx == total_batches):
            print(f"[Task2] progreso {b_idx}/{total_batches} batches | y_true={len(y_true)}")

    elapsed = time.perf_counter() - started

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    result = EvalResult(
        task="task2",
        model=model,
        n=len(y_true),
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        total_time_s=float(round(elapsed, 4)),
        time_per_doc_s=float(round(elapsed / max(len(y_true), 1), 6)),
    )

    if use_mlflow:
        _log_mlflow(
            params={
                "run_name": f"task2-{model}",
                "task": "task2",
                "model": model,
                "encoder_variant": encoder_variant,
                "batch_size": batch_size,
                "n_samples": len(y_true),
            },
            metrics={
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "total_time_s": result.total_time_s,
                "time_per_doc_s": result.time_per_doc_s,
            },
            tags={"dataset": TASK2_PATH.name},
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate golden sets for Task1/Task2.")
    parser.add_argument("--task", choices=["task1", "task2", "both"], default="both")
    parser.add_argument("--model", choices=["encoder", "llm", "api", "all"], default="all")
    parser.add_argument("--encoder-variant", choices=["roberta", "scibert"], default="roberta")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--batch-size-task2", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--out", type=str, default="")

    args = parser.parse_args()
    models = ["encoder", "llm", "api"] if args.model == "all" else [args.model]

    results: list[EvalResult] = []
    for m in models:
        if args.task in {"task1", "both"}:
            results.append(
                evaluate_task1(
                    m,
                    encoder_variant=args.encoder_variant,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    use_mlflow=not args.no_mlflow,
                )
            )
        if args.task in {"task2", "both"}:
            results.append(
                evaluate_task2(
                    m,
                    encoder_variant=args.encoder_variant,
                    batch_size=args.batch_size_task2,
                    limit=args.limit,
                    use_mlflow=not args.no_mlflow,
                )
            )

    rows = [r.__dict__ for r in results]
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
