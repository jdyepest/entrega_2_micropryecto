import json
import os
import sys


def main() -> None:
    """
    Smoke test: ejecuta el endpoint /api/analyze con model=encoder y task=segmentation.

    Ejecutar desde app/backend:
      python3 scripts/smoke_test_encoder_api.py
    """
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, backend_dir)

    from main import app  # noqa: E402

    sample_text = (
        "En este trabajo presentamos SciText-ES, un sistema de análisis automático de documentos científicos en español.\n\n"
        "En la literatura se han propuesto múltiples enfoques basados en transformadores para tareas de PLN.\n\n"
        "Nuestra metodología utiliza un modelo RoBERTa fine-tuned y evaluamos su rendimiento con F1.\n\n"
        "Los resultados muestran una mejora de 3 puntos porcentuales en F1 respecto a baselines.\n\n"
        "En conclusión, el sistema es útil y planteamos trabajo futuro."
    )

    payload = {"text": sample_text, "model": "encoder", "tasks": ["segmentation"]}

    with app.test_client() as client:
        resp = client.post("/api/analyze", json=payload)
        print("status:", resp.status_code)
        data = resp.get_json()
        print("keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        seg = (data or {}).get("segmentation") or {}
        segments = seg.get("segments") or []
        print("segments:", len(segments))
        if segments:
            print("first:", json.dumps(segments[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

