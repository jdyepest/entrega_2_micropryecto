"""
Tarea 1 – Segmentación retórica.

Función pública: analyze_segments(text, model)
→ Devuelve lista de segmentos con label y confianza.

Para reemplazar el mock por un modelo real, edita únicamente
_call_real_model() y pon la lógica de inferencia ahí.
"""

import re
import time
import random

from services.models import MODELS

# ---------------------------------------------------------------------------
# Keywords por categoría (usadas en el mock para simular decisión)
# ---------------------------------------------------------------------------
_KEYWORDS = {
    "INTRO": [
        "introducción", "este trabajo", "en este artículo", "el presente",
        "motivación", "este estudio", "objetivo", "propósito", "presente trabajo",
    ],
    "BACK": [
        "antecedentes", "trabajos previos", "revisión de literatura", "estado del arte",
        "investigaciones anteriores", "han propuesto", "fue propuesto", "en la literatura",
        "según", "de acuerdo con", "autores",
    ],
    "METH": [
        "metodología", "método", "metodológico", "procedimiento", "experimento",
        "implementación", "arquitectura", "entrenamiento", "corpus", "dataset",
        "conjunto de datos", "evaluamos", "utilizamos", "se utilizó", "fine-tuning",
        "hiperparámetros", "configuración",
    ],
    "RES": [
        "resultado", "tabla", "figura", "obtuvo", "obtuvimos", "rendimiento",
        "accuracy", "f1", "precisión", "recall", "métricas", "muestra",
        "se observa", "se puede ver", "en la tabla", "en la figura",
    ],
    "DISC": [
        "discusión", "análisis", "interpretamos", "esto sugiere", "esto indica",
        "podemos inferir", "se debe a", "explicar", "probable", "parece",
        "comparado con", "en comparación",
    ],
    "CONTR": [
        "contribución", "aporte", "propuesta", "novedad", "innovación",
        "original", "nueva", "nuevo", "presentamos", "proponemos",
        "a diferencia de", "primer trabajo",
    ],
    "LIM": [
        "limitación", "limitaciones", "trabajo futuro", "futuras investigaciones",
        "no se consideró", "fuera del alcance", "restricción", "sesgo",
    ],
    "CONC": [
        "conclusión", "conclusiones", "en conclusión", "en resumen",
        "resumiendo", "concluimos", "hemos mostrado", "hemos demostrado",
        "finalmente", "en definitiva",
    ],
}

# Orden posicional esperado (para párrafos sin keywords claros)
_POSITIONAL_ORDER = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]


def _split_paragraphs(text: str) -> list[str]:
    """Divide el texto en párrafos no vacíos."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if len(p.strip()) > 20]


def _keyword_score(paragraph: str) -> dict[str, float]:
    """Calcula score por categoría según keywords."""
    lower = paragraph.lower()
    scores = {}
    for label, kws in _KEYWORDS.items():
        count = sum(1 for kw in kws if kw in lower)
        scores[label] = count
    return scores


def _mock_label_paragraph(
    paragraph: str,
    index: int,
    total: int,
    model: str,
    rng: random.Random,
) -> tuple[str, float]:
    """
    Asigna un label y confianza simulados a un párrafo.

    Estrategia:
    1. Calcular scores por keywords.
    2. Si hay un ganador claro, usarlo con confianza alta.
    3. Si hay empate, usar posición en el texto como desempate.
    4. Añadir variación según modelo.
    """
    scores = _keyword_score(paragraph)
    max_score = max(scores.values())

    # Fuerza el primer y último párrafo a INTRO/CONC si no tienen keywords fuertes
    if index == 0 and max_score < 2:
        label = "INTRO"
        base_conf = 0.82
    elif index == total - 1 and max_score < 2:
        label = "CONC"
        base_conf = 0.80
    elif max_score > 0:
        # Tomar el label con mayor score
        label = max(scores, key=lambda k: scores[k])
        # Confianza proporcional al score
        base_conf = 0.72 + min(max_score * 0.05, 0.22)
    else:
        # Sin keywords: usar posición relativa
        pos_ratio = index / max(total - 1, 1)
        order_idx = int(pos_ratio * (len(_POSITIONAL_ORDER) - 1))
        label = _POSITIONAL_ORDER[order_idx]
        base_conf = 0.62 + rng.uniform(0, 0.10)

    # Ajuste por modelo (encoder es más rápido pero varía más)
    if model == "encoder":
        noise = rng.uniform(-0.06, 0.06)
    elif model == "llm":
        noise = rng.uniform(-0.04, 0.05)
    else:  # api
        noise = rng.uniform(-0.02, 0.04)

    confidence = round(min(max(base_conf + noise, 0.50), 0.99), 2)
    return label, confidence


def _mock_analyze(text: str, model: str) -> dict:
    """
    Mock completo de segmentación retórica.
    Usa el texto real del usuario para producir resultados coherentes.
    """
    model_config = MODELS[model]
    rng = random.Random(hash(text[:100]) ^ hash(model))  # determinista por texto+modelo

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    total = len(paragraphs)
    segments = []
    for i, para in enumerate(paragraphs):
        label, confidence = _mock_label_paragraph(para, i, total, model, rng)
        segments.append({
            "paragraph_index": i,
            "text": para,
            "label": label,
            "confidence": confidence,
        })

    word_count = sum(len(p.split()) for p in paragraphs)
    avg_conf = round(sum(s["confidence"] for s in segments) / len(segments), 3)
    elapsed = round(model_config["simulated_delay_s"] + rng.uniform(0.1, 0.4), 2)

    return {
        "segments": segments,
        "stats": {
            "total_paragraphs": total,
            "total_words": word_count,
            "avg_confidence": avg_conf,
            "time_seconds": elapsed,
        },
    }


# ---------------------------------------------------------------------------
# Punto de entrada público — REEMPLAZA ESTE BLOQUE CON EL MODELO REAL
# ---------------------------------------------------------------------------

def analyze_segments(text: str, model: str) -> dict:
    """
    Analiza el texto y devuelve segmentos retóricos con labels y confianzas.

    Args:
        text:  Texto del artículo científico.
        model: Identificador del modelo ("encoder", "llm", "api").

    Returns:
        Dict con "segments" y "stats".

    TODO: Reemplazar _mock_analyze() por _call_real_model() cuando las APIs estén disponibles.
    """
    # Simular latencia de red/inferencia
    time.sleep(MODELS[model]["simulated_delay_s"])
    return _mock_analyze(text, model)


# ---------------------------------------------------------------------------
# Stub para el modelo real (implementar cuando esté disponible)
# ---------------------------------------------------------------------------

def _call_real_model(text: str, model: str) -> dict:
    """
    STUB — Implementar con llamada real al modelo.

    Encoder:
        from transformers import pipeline
        classifier = pipeline("text-classification", model="dccuchile/bert-base-spanish-wwm-cased")
        ...

    LLM Open-Weight:
        import requests
        response = requests.post(OLLAMA_ENDPOINT, json={"model": "llama3", "prompt": prompt})
        ...

    API Comercial:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(model="gpt-4o", messages=[...])
        ...
    """
    raise NotImplementedError("Real model not yet configured.")
