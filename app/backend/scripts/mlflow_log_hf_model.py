import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load .env from current working dir (and parents) if present
    load_dotenv()
except Exception:
    # Optional dependency; script still works if env vars are already exported.
    pass


def main() -> None:
    """
    Sube una carpeta de modelo HuggingFace (config/tokenizer/pesos) a MLflow como artefacto.

    Ejemplo (con server MLflow configurado con artifact store en S3):
      python3 app/backend/scripts/mlflow_log_hf_model.py \
        --model_dir src/models/scibert_task1 \
        --artifact_path hf_model \
        --run_name scibert_task1

    Imprime un URI que puedes usar luego en el backend:
      TASK1_ENCODER_MLFLOW_MODEL_URI=runs:/<run_id>/hf_model
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directorio con config.json + model.safetensors + tokenizer.json")
    ap.add_argument("--artifact_path", default="hf_model", help="Ruta dentro del run donde se guardará")
    ap.add_argument("--run_name", default="hf_model", help="Nombre del run")
    ap.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "scitext-models"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    required = ["config.json", "tokenizer.json"]
    for r in required:
        if not (model_dir / r).exists():
            raise ValueError(f"Falta {r} en {model_dir}")
    if not (model_dir / "model.safetensors").exists() and not (model_dir / "pytorch_model.bin").exists():
        raise ValueError("Falta archivo de pesos: model.safetensors o pytorch_model.bin")

    try:
        import mlflow
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Instala dependencias: pip install mlflow boto3") from e

    tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if args.experiment:
        mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        # Helpful debug: shows which bucket/root MLflow will use.
        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Experiment:", args.experiment)
        print("Artifact URI:", run.info.artifact_uri)
        mlflow.log_artifacts(str(model_dir), artifact_path=args.artifact_path)
        run_id = run.info.run_id

    print("Logged artifacts from:", str(model_dir))
    print("Run ID:", run_id)
    print("Use this URI:")
    print(f"runs:/{run_id}/{args.artifact_path}")


if __name__ == "__main__":
    main()
