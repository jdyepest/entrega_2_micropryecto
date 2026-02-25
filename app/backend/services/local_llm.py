import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from services.errors import UpstreamServiceError


def get_local_llm_model_name() -> str:
    """
    Selección de modelo local (open-weight) con flag para prod:
      - LOCAL_LLM_MODEL: override absoluto (si existe, se usa tal cual)
      - LOCAL_LLM_VARIANT: "small" | "large" (default: small)
      - LOCAL_LLM_SMALL_MODEL: default llama3.2:3b
      - LOCAL_LLM_LARGE_MODEL: default llama3.1:8b
    """
    override = (os.environ.get("LOCAL_LLM_MODEL") or "").strip()
    if override:
        return override

    variant = (os.environ.get("LOCAL_LLM_VARIANT") or "small").strip().lower()
    # NOTE: Ollama tags can vary; use simple defaults and allow override via env vars.
    small = (os.environ.get("LOCAL_LLM_SMALL_MODEL") or "llama3.2:3b").strip()
    large = (os.environ.get("LOCAL_LLM_LARGE_MODEL") or "llama3.1:8b").strip()

    return large if variant == "large" else small


def get_ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")


def parse_json_loose(text: str) -> Any:
    def _raw_decode_first(s: str) -> Any:
        dec = json.JSONDecoder()
        s2 = s.lstrip()
        obj, _end = dec.raw_decode(s2)
        return obj

    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output")

    try:
        return _raw_decode_first(t)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return _raw_decode_first(fenced.group(1))

    # Try decoding from the first JSON-looking bracket to ignore trailing text ("extra data").
    start_any = min([p for p in [t.find("["), t.find("{")] if p != -1], default=-1)
    if start_any != -1:
        try:
            return _raw_decode_first(t[start_any:])
        except json.JSONDecodeError:
            pass

    # Last attempt: scan for any '{' or '[' and try raw_decode from there.
    for m in re.finditer(r"[\[{]", t):
        try:
            return _raw_decode_first(t[m.start() :])
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON from model output")


def ollama_chat_json(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout_s: float | None = None,
) -> Any:
    """
    Llama a Ollama /api/chat con format=json y stream=false.
    Devuelve el JSON parseado del contenido del mensaje.
    """
    model_name = model or get_local_llm_model_name()
    base_url = get_ollama_base_url()
    temp = float(os.environ.get("LOCAL_LLM_TEMPERATURE") or "0.2") if temperature is None else float(temperature)
    tout = float(os.environ.get("LOCAL_LLM_TIMEOUT_S") or "60") if timeout_s is None else float(timeout_s)

    url = f"{base_url}/api/chat"
    payload = {
        "model": model_name,
        "stream": False,
        "format": "json",
        "options": {"temperature": temp},
        "messages": [{"role": "user", "content": prompt}],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=tout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            print(f"Ollama raw response (truncado): {raw[:800]}")
            outer = json.loads(raw)
            print(f"Ollama parsed response: {outer}")
    except urllib.error.HTTPError as e:
        
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise UpstreamServiceError("Ollama", f"HTTP {e.code}: {detail}", status_code=502) from e
    except urllib.error.URLError as e:
        raise UpstreamServiceError(
            "Ollama",
            f"No se pudo conectar a {base_url} (¿ollama está corriendo?). Detalle: {e}",
            status_code=503,
        ) from e

    content = ((outer.get("message") or {}).get("content") or "").strip()
    if not content:
        raise UpstreamServiceError("Ollama", f"Respuesta inválida (sin message.content): {outer}", status_code=502)

    return parse_json_loose(content)
