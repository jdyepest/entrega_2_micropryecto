# Manual de Instalación — SciText-ES (Local)

**Versión:** 1.0  \
**Audiencia:** Equipo técnico que instala la app localmente (con y sin Docker).  \
**Alcance:** Instalación de backend + frontend + proxy LLM + MLflow (opcional).

---

## 1. Requisitos

### Opción A: Sin Docker
- Python 3.10+ (recomendado 3.11)
- `pip`
- (Opcional) Ollama local si usarás LLM local
- (Opcional) MLflow si usarás modelos vía MLflow

### Opción B: Con Docker
- Docker Desktop (o Docker Engine + Compose)
- AWS CLI (si el build requiere `dvc pull` desde S3)

---

## 2. Variables de entorno (explicación)

Estas variables se cargan desde `.env`:

### Backend
- `PORT`: puerto del backend (default `5000`)
- `DEBUG`: `1` para debug, `0` para producción
- `LOG_LEVEL`: `DEBUG`/`INFO`/`WARNING`
- `LOG_FILE`: ruta para logs del backend

### LLM (OpenRouter proxy)
- `OLLAMA_BASE_URL`: endpoint del proxy compatible con Ollama
  - Local (sin Docker): `http://127.0.0.1:11434`
  - Docker Compose: `http://ollama:11434`
  - ECS (mismo task): `http://127.0.0.1:11434`
- `OPENROUTER_API_KEY`: clave de OpenRouter (obligatoria si usas OpenRouter)
- `OPENROUTER_MODEL`: ID de modelo (ej: `meta-llama/llama-3.2-3b-instruct`)
- `OPENROUTER_BASE_URL`: `https://openrouter.ai/api/v1`
- `OPENROUTER_TIMEOUT_S`: timeout en segundos
- `OPENROUTER_PROVIDER_ONLY`: opcional (forzar proveedor)
- `OPENROUTER_QUANTIZATIONS`: opcional (ej: `int8`)
- `OPENROUTER_ALLOW_FALLBACKS`: `1` para permitir fallback

### MLflow (opcional)
- `MLFLOW_TRACKING_URI`: URL del servidor MLflow
  - Local sin Docker: `http://127.0.0.1:5006`
  - Docker Compose: `http://mlflow:5006`
  - ECS (mismo task): `http://127.0.0.1:5006`

### Modelos encoder (Task1/Task2)
- `TASK1_ENCODER_ROBERTA_MLFLOW_MODEL_URI`
- `TASK1_ENCODER_SCIBERT_MLFLOW_MODEL_URI`
- `TASK2_ENCODER_ROBERTA_MLFLOW_MODEL_URI`
- `TASK2_ENCODER_SCIBERT_MLFLOW_MODEL_URI`

> Si quieres evitar MLflow, usa URIs directos a S3 (ej. `s3://.../hf_model`).

---

## 3. Instalación SIN Docker

### 3.1 Backend
```bash
cd app/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r ../../requirments.txt
python main.py
```

Backend en: `http://localhost:5000`

### 3.2 Proxy LLM (OpenRouter)
```bash
cd app/ollama
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx python-dotenv
uvicorn app:app --host 0.0.0.0 --port 11434
```

### 3.3 MLflow (opcional)
```bash
pip install mlflow boto3
mlflow server \
  --backend-store-uri sqlite:///./mlflow.db \
  --default-artifact-root s3://bucket-artifacts-models-2026-03/mlflow \
  --host 0.0.0.0 --port 5006
```

---

## 4. Instalación CON Docker (local)

### 4.1 Preparar `.env`
Asegura estas variables:

```env
MLFLOW_TRACKING_URI=http://mlflow:5006
OLLAMA_BASE_URL=http://ollama:11434
OPENROUTER_API_KEY=sk-or-xxxx
OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct
```

### 4.2 Levantar servicios
```bash
docker compose up --build
```

Backend en: `http://localhost:5000`

---

## 5. Datos con DVC (si aplica)

Si necesitas los datos versionados:

```bash
dvc pull
```

> Requiere credenciales de AWS con acceso al bucket.

---

## 6. Verificación rápida

- Abre: `http://localhost:5000`
- Pega un texto de prueba y ejecuta análisis
- Si hay errores, revisa los logs:

```bash
docker compose logs -f backend
```

---

## 7. Solución de problemas

- **Timeouts del LLM**: aumenta `OPENROUTER_TIMEOUT_S`
- **401 OpenRouter**: revisa `OPENROUTER_API_KEY`
- **MLflow no responde**: revisa `MLFLOW_TRACKING_URI`
- **Rate limit**: espera unos minutos entre ejecuciones

---

**Fin del manual.**
