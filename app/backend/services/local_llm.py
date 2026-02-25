import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def get_local_llm_model_name() -> str:
    """
    Selección de modelo local (open-weight) con flag para prod:
      - LOCAL_LLM_MODEL: override absoluto (si existe, se usa tal cual)
      - LOCAL_LLM_VARIANT: "small" | "large" (default: small)
      - LOCAL_LLM_SMALL_MODEL: default llama3.2:3b-instruct
      - LOCAL_LLM_LARGE_MODEL: default llama3.1:8b-instruct
    """
    override = (os.environ.get("LOCAL_LLM_MODEL") or "").strip()
    if override:
        return override

    variant = (os.environ.get("LOCAL_LLM_VARIANT") or "small").strip().lower()
    small = (os.environ.get("LOCAL_LLM_SMALL_MODEL") or "llama3.2:3b-instruct").strip()
    large = (os.environ.get("LOCAL_LLM_LARGE_MODEL") or "llama3.1:8b-instruct").strip()

    return large if variant == "large" else small


def get_ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")


def parse_json_loose(text: str) -> Any:
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start_list = t.find("[")
    end_list = t.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        return json.loads(t[start_list : end_list + 1])

    start_obj = t.find("{")
    end_obj = t.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return json.loads(t[start_obj : end_obj + 1])

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
            outer = json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Error de red llamando a Ollama: {e}") from e

    content = ((outer.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Ollama response missing message.content: {outer}")

    return parse_json_loose(content)

