import json
import os
import sys


def main() -> None:
    """
    Smoke test: /api/analyze con model=encoder y task=contributions (Task2 encoder).

    Requiere:
      - TASK2_ENCODER_<VARIANT>_MLFLOW_MODEL_URI (o *_MODEL_PATH) configurado
      - deps: torch + transformers + safetensors
    """
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, backend_dir)

    from main import app  # noqa: E402

    sample_text = (
        "En este trabajo presentamos SciText-ES, un sistema de análisis automático de documentos científicos en español.\n\n"
        "Nuestra contribución principal es proponer un pipeline de anotación y liberar un dataset.\n\n"
        "Según la literatura, existen enfoques previos similares.\n\n"
        "En conclusión, el método es útil."
    )

    variant = (os.environ.get("ENCODER_VARIANT") or "roberta").strip().lower()
    payload = {"text": sample_text, "model": "encoder", "tasks": ["segmentation", "contributions"], "encoder_variant": variant}

    with app.test_client() as client:
        resp = client.post("/api/analyze", json=payload)
        print("status:", resp.status_code)
        data = resp.get_json()
        if resp.status_code != 200:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return

        cont = (data.get("contributions") or {}).get("fragments") or []
        print("fragments:", len(cont))
        if cont:
            print("first:", json.dumps(cont[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

