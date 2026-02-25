"""
Tarea 2 – Extracción de contribuciones científicas.

Función pública: analyze_contributions(segments, model)
→ Recibe los segmentos de la Tarea 1 y devuelve fragmentos anotados.

Para reemplazar el mock por un modelo real, edita únicamente
_call_real_model() y pon la lógica de inferencia ahí.
"""

import re
import time
import random
import json

from services.models import MODELS
from services.local_llm import ollama_chat_json

# ---------------------------------------------------------------------------
# Mapeo de categorías retóricas a tipos de contribución
# ---------------------------------------------------------------------------
_LABEL_TO_TYPE = {
    "METH":  "Metodológica",
    "RES":   "Empírica",
    "CONTR": "Recurso",
    "DISC":  "Conceptual",
    "INTRO": "Conceptual",
    "BACK":  "Conceptual",
    "LIM":   "Conceptual",
    "CONC":  "Metodológica",
}

# Labels más propensos a tener contribuciones
_HIGH_CONTRIBUTION_LABELS = {"CONTR", "METH", "RES"}
_MED_CONTRIBUTION_LABELS = {"DISC", "CONC", "INTRO"}

# Frases de contribución por tipo (para el highlight mock)
_HIGHLIGHT_PATTERNS = {
    "Metodológica": [
        r"propone?mos\s+(?:un|una|el|la)\s+\w+(?:\s+\w+){1,5}",
        r"nuevo\s+(?:método|enfoque|algoritmo|sistema|marco|modelo)\s+\w+(?:\s+\w+){0,4}",
        r"implementa(?:mos|ción)\s+\w+(?:\s+\w+){1,4}",
        r"utilizamos\s+(?:un|una)\s+\w+(?:\s+\w+){1,5}",
    ],
    "Empírica": [
        r"obtuv(?:imos|o)\s+(?:un|una)\s+\w+(?:\s+\w+){1,4}",
        r"f1\s*(?:score|=|de)\s*[\d.,]+",
        r"precisión\s+de\s+[\d.,]+\s*%",
        r"supera(?:mos|ndo)?\s+(?:el|la|los)\s+\w+(?:\s+\w+){1,4}",
        r"mejora(?:mos|ndo)?\s+(?:en|el|la)\s+\w+(?:\s+\w+){1,4}",
    ],
    "Recurso": [
        r"corpus\s+\w+(?:\s+\w+){0,4}",
        r"dataset\s+\w+(?:\s+\w+){0,4}",
        r"recurso\s+(?:léxico|lingüístico|anotado)\s+\w+(?:\s+\w+){0,3}",
        r"publicamos\s+\w+(?:\s+\w+){1,4}",
        r"ponemos\s+a\s+disposición\s+\w+(?:\s+\w+){1,4}",
    ],
    "Conceptual": [
        r"definimos\s+(?:el|la|un|una)\s+\w+(?:\s+\w+){1,4}",
        r"proponemos\s+(?:un|una)\s+(?:marco|taxonomía|definición)\s+\w+(?:\s+\w+){0,4}",
        r"concepto\s+de\s+\w+(?:\s+\w+){1,4}",
        r"demostram(?:os|amos)\s+que\s+\w+(?:\s+\w+){1,4}",
    ],
}


def _find_highlight(text: str, contribution_type: str, rng: random.Random) -> str:
    """Extrae una frase clave del texto para resaltar como contribución."""
    patterns = _HIGHLIGHT_PATTERNS.get(contribution_type, [])
    lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            start, end = match.span()
            return text[start:end].strip()

    # Fallback: devolver las primeras palabras de una oración con contribución
    sentences = re.split(r"[.;]", text)
    for sent in sentences:
        if len(sent.split()) >= 6:
            words = sent.strip().split()
            return " ".join(words[:min(10, len(words))])

    return text[:80].strip()


def _mock_analyze(segments: list[dict], model: str) -> dict:
    """
    Mock de extracción de contribuciones.
    Usa los segmentos reales de la Tarea 1.
    """
    model_config = MODELS[model]
    rng = random.Random(hash(str(segments)[:80]) ^ hash(model))

    fragments = []
    for seg in segments:
        label = seg["label"]
        text = seg["text"]

        # Probabilidad de que este fragmento sea contribución según su label y modelo
        if label in _HIGH_CONTRIBUTION_LABELS:
            base_prob = 0.80
        elif label in _MED_CONTRIBUTION_LABELS:
            base_prob = 0.40
        else:
            base_prob = 0.20

        # Ajuste por modelo
        if model == "encoder":
            prob_noise = rng.uniform(-0.10, 0.05)
        elif model == "llm":
            prob_noise = rng.uniform(-0.05, 0.08)
        else:  # api
            prob_noise = rng.uniform(0.0, 0.10)

        is_contribution = rng.random() < (base_prob + prob_noise)

        # Tipo de contribución (basado en label)
        contribution_type = _LABEL_TO_TYPE.get(label, "Conceptual")

        # Confianza
        if is_contribution:
            base_conf = 0.78 + rng.uniform(0, 0.18)
            if model == "api":
                base_conf += 0.04
            elif model == "encoder":
                base_conf -= 0.03
        else:
            base_conf = 0.30 + rng.uniform(0, 0.30)

        confidence = round(min(max(base_conf, 0.10), 0.99), 2)

        highlight = ""
        if is_contribution:
            highlight = _find_highlight(text, contribution_type, rng)

        fragments.append({
            "paragraph_index": seg["paragraph_index"],
            "text": text,
            "is_contribution": is_contribution,
            "contribution_type": contribution_type if is_contribution else None,
            "confidence": confidence,
            "highlight": highlight,
            "source_label": label,
        })

    positives = [f for f in fragments if f["is_contribution"]]
    avg_conf_pos = (
        round(sum(f["confidence"] for f in positives) / len(positives), 3)
        if positives else 0.0
    )

    return {
        "fragments": fragments,
        "stats": {
            "total_fragments": len(fragments),
            "positive": len(positives),
            "negative": len(fragments) - len(positives),
            "avg_confidence_positive": avg_conf_pos,
        },
    }


# ---------------------------------------------------------------------------
# Punto de entrada público — REEMPLAZA ESTE BLOQUE CON EL MODELO REAL
# ---------------------------------------------------------------------------

def analyze_contributions(segments: list[dict], model: str) -> dict:
    """
    Analiza los segmentos e identifica contribuciones científicas.

    Args:
        segments: Lista de segmentos con label (salida de analyze_segments).
        model:    Identificador del modelo ("encoder", "llm", "api").

    Returns:
        Dict con "fragments" y "stats".

    TODO: Reemplazar _mock_analyze() por _call_real_model() cuando las APIs estén disponibles.
    """
    if model == "llm":
        return _call_local_llm(segments)

    time.sleep(MODELS[model]["simulated_delay_s"] * 0.5)
    return _mock_analyze(segments, model)


# ---------------------------------------------------------------------------
# Stub para el modelo real
# ---------------------------------------------------------------------------

def _call_real_model(segments: list[dict], model: str) -> dict:
    """
    STUB — Implementar con llamada real al modelo.

    El formato de retorno debe ser idéntico al de _mock_analyze().
    """
    raise NotImplementedError("Real model not yet configured.")


def _call_local_llm(segments: list[dict]) -> dict:
    """
    LLM open-weight local (p.ej. Ollama) para decidir si cada segmento expresa
    explícitamente una contribución científica (binario).
    """
    items = [{"paragraph_index": s["paragraph_index"], "text": s["text"], "label": s["label"]} for s in segments]
    prompt = (
        "Decide si cada fragmento (párrafo) expresa EXPLÍCITAMENTE una contribución científica.\n"
        "Marca positivo SOLO si hay formulación explícita de aporte/novelty (p.ej. 'proponemos', 'presentamos', 'nuestra contribución', 'ponemos a disposición').\n"
        "Devuelve SOLO JSON (sin Markdown) como ARREGLO, un item por entrada con:\n"
        '  - "paragraph_index": int\n'
        '  - "is_contribution": boolean\n'
        '  - "confidence": number 0..1\n\n'
        "Entrada (JSON):\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )

    started = time.perf_counter()
    parsed = ollama_chat_json(prompt)
    elapsed = round(time.perf_counter() - started, 2)

    if not isinstance(parsed, list):
        raise TypeError("Salida del modelo inválida: se esperaba un arreglo JSON.")

    by_idx = {}
    for it in parsed:
        if isinstance(it, dict) and isinstance(it.get("paragraph_index"), int):
            by_idx[int(it["paragraph_index"])] = it

    fragments = []
    for seg in segments:
        idx = int(seg["paragraph_index"])
        out = by_idx.get(idx, {})
        is_contribution = bool(out.get("is_contribution", False))
        try:
            confidence = float(out.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        contribution_type = _LABEL_TO_TYPE.get(seg["label"], "Conceptual")
        highlight = _find_highlight(seg["text"], contribution_type, random.Random(idx))
        if not is_contribution:
            contribution_type = None
            highlight = ""

        fragments.append(
            {
                "paragraph_index": idx,
                "text": seg["text"],
                "is_contribution": is_contribution,
                "contribution_type": contribution_type,
                "confidence": confidence,
                "highlight": highlight,
                "source_label": seg["label"],
            }
        )

    positives = [f for f in fragments if f["is_contribution"]]
    avg_conf_pos = (
        round(sum(f["confidence"] for f in positives) / len(positives), 3)
        if positives else 0.0
    )

    return {
        "fragments": fragments,
        "stats": {
            "total_fragments": len(fragments),
            "positive": len(positives),
            "negative": len(fragments) - len(positives),
            "avg_confidence_positive": avg_conf_pos,
            "time_seconds": elapsed,
        },
    }
