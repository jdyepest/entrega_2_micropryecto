# Microproyecto MAIA — Análisis de documentos científicos en español

**Tema 1 – 2026** · Grupo FLAG-TICsW · Universidad de los Andes

Este repositorio corresponde a un microproyecto MAIA cuyo objetivo es desarrollar una solución computacional para el análisis automático de documentos científicos en español, abordando (i) la segmentación y clasificación retórica y (ii) la extracción de contribuciones científicas.

## Propósito del repositorio

Este proyecto implementa:

- Preparación y curaduría de un corpus científico en español.
- Modelos para segmentación y clasificación retórica.
- Modelos para detección de contribuciones científicas.
- Evaluación comparativa entre modelos entrenados y modelos de lenguaje.
- Aplicación web interactiva para visualización de resultados.
- Scripts reproducibles para experimentación y análisis de resultados.

El enfoque es académico y experimental, orientado a entregar una solución funcional y evaluable.

## Tareas

| Tarea | Descripción | Labels / Tipos |
|-------|-------------|----------------|
| **Tarea 1** | Segmentación retórica | INTRO, BACK, METH, RES, DISC, CONTR, LIM, CONC |
| **Tarea 2** | Detección de contribuciones (binaria) | is_contribution = true/false |

## Estructura del proyecto

```
.
├── app/                           # 🖥️ Aplicación web (frontend + backend)
│   ├── backend/
│   │   ├── main.py                # Entry point Flask
│   │   ├── routes/
│   │   │   ├── analysis.py        # POST /api/analyze
│   │   │   └── comparison.py      # GET /api/compare/<id>
│   │   ├── services/
│   │   │   ├── segmentation.py    # Tarea 1 (mock → real)
│   │   │   ├── contributions.py   # Tarea 2 (mock → real)
│   │   │   └── models.py          # Configuración de modelos
│   │   ├── mock_data/             # JSONs de referencia
│   │   └── requirements.txt
│   └── frontend/
│       ├── index.html             # Vista 1: Entrada de texto
│       ├── segmentation.html      # Vista 2: Segmentación retórica
│       ├── contributions.html     # Vista 3: Contribuciones
│       ├── comparison.html        # Vista 4: Comparación de modelos
│       ├── css/styles.css
│       └── js/
│           ├── app.js
│           ├── api.js
│           └── charts.js
│
├── datos/                         # 📊 Datos del proyecto
│   └── core/                      # Corpus científico crudo (CORE)
│
├── src/                           # 🔧 Código fuente principal
│   ├── preprocessing/             # Limpieza, normalización y segmentación
│   ├── task1_rhetorical/          # Segmentación y clasificación retórica
│   ├── task2_contributions/       # Extracción de contribuciones científicas
│   ├── models/                    # Definición y carga de modelos
│   └── utils/                     # Funciones auxiliares comunes
│
├── experiments/                   # 🧪 Experimentos y configuraciones
│   ├── task1/                     # Experimentos de clasificación retórica
│   └── task2/                     # Experimentos de extracción de contribuciones
│
├── evaluation/                    # 📈 Evaluación y análisis de resultados
│   ├── metrics/                   # Métricas cuantitativas
│   └── error_analysis/            # Análisis cualitativo de errores
│
├── notebooks/                     # 📓 Análisis exploratorio y pruebas
├── artifacts/                     # Artefactos generados
├── configs/                       # Configuraciones de modelos y experimentos
├── data_lake/scripts/             # Scripts de data lake
│
├── .dvc/                          # Configuración DVC
├── .dvcignore
├── .gitignore
├── datos.dvc                      # Tracking DVC del corpus
├── Propuesta_Proyecto_PLN_FLAG.pdf
└── README.md
```

## Descripción de componentes

| Carpeta | Descripción |
|---------|-------------|
| `app/` | Aplicación web con backend Flask y frontend vanilla. Interfaz para analizar textos y comparar modelos. |
| `datos/` | Corpus de documentos científicos en español (CORE API). |
| `src/` | Lógica central: preprocesamiento, clasificación retórica, detección de contribuciones. |
| `experiments/` | Scripts para ejecutar experimentos controlados y comparables. |
| `evaluation/` | Cálculo de métricas, matrices de confusión y análisis de errores. |
| `notebooks/` | Exploración de datos, pruebas de modelos y análisis intermedios. |

## Alcance

Este repositorio cubre todo el flujo de solución:

- Desde documentos científicos crudos → hasta resultados evaluados y comparables
- Incluye una aplicación web para visualización interactiva de ambas tareas

---

## Cómo lanzar el proyecto

### 1. Configurar variables de entorno (recomendado: `.env`)

El backend carga automáticamente un archivo `.env` (si existe) al arrancar.

Variables típicas:

```bash
# Flask
PORT=5000
DEBUG=1
LOG_LEVEL=INFO  # DEBUG para más detalle

# Gemini (model="api")
GEMINI_API_KEY="..."
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_TIMEOUT_S=60
GEMINI_STRUCTURED_OUTPUT=1

# Ollama (model="llm")
OLLAMA_BASE_URL="http://localhost:11434"
LOCAL_LLM_VARIANT=small   # small (~3B) | large (8B)
LOCAL_LLM_MODEL=""        # opcional: override del tag

# Cache local (evita re-descargas de modelos)
MODEL_CACHE_DIR=artifacts/model_cache
```

### 2. Configurar AWS (para descargar modelos desde S3)

```bash
aws configure
```

Verificar que quedaron activas:

```bash
aws sts get-caller-identity
```

### 3. Descargar los datos con DVC

Desde la raíz del repositorio:

```bash
dvc pull
```

### 4. Ejecutar la aplicación web

```bash
cd app/backend

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el servidor
python main.py
# → Servidor en http://localhost:5000
```

Abre http://localhost:5000 en tu navegador.

---

## Aplicación web — SciText-ES

### Vistas

1. **Entrada** (`/`) — Pega el texto, selecciona modelo y tareas
2. **Segmentación** (`/segmentation.html`) — Párrafos clasificados con colores y confianza
3. **Contribuciones** (`/contributions.html`) — Fragmentos con aportes identificados
4. **Comparación** (`/comparison.html`) — Métricas F1/Precisión/Recall/Latencia de los 3 modelos

### API

**`POST /api/analyze`**
```json
{
  "text": "Texto del artículo…",
  "model": "encoder | llm | api",
  "tasks": ["segmentation", "contributions"],
  "encoder_variant": "roberta | scibert"
}
```
Devuelve segmentos etiquetados y fragmentos con contribuciones.

**`GET /api/compare/<analysis_id>`**

Devuelve métricas comparativas de los 3 modelos para el texto analizado.

### Swagger / OpenAPI

La app expone documentación interactiva:

- UI: `http://localhost:5000/apidocs/`
- Spec JSON: `http://localhost:5000/apispec_1.json`

Para exportar el spec:
```bash
curl -s "http://localhost:5000/apispec_1.json" > apispec_1.json
```

### Variables de entorno

```bash
PORT=5000       # Puerto del servidor (por defecto: 5000)
DEBUG=1         # Modo debug de Flask (por defecto: 1)
```

### Modelos encoder (Task1 + Task2) vía MLflow (S3)

Para evitar commitear archivos grandes (por ejemplo `model.safetensors`) en GitHub, puedes subir los modelos a MLflow y referenciarlos por URI `runs:/...`.

**Cache local (evita re-descargas):**
```bash
MODEL_CACHE_DIR=artifacts/model_cache
```

**Task1 (segmentación retórica):**
```bash
TASK1_ENCODER_ROBERTA_MLFLOW_MODEL_URI="runs:/<run_id>/hf_model"
TASK1_ENCODER_SCIBERT_MLFLOW_MODEL_URI="runs:/<run_id>/hf_model"
```

**Task2 (contribuciones binario):**
```bash
TASK2_ENCODER_ROBERTA_MLFLOW_MODEL_URI="runs:/<run_id>/hf_model"
TASK2_ENCODER_SCIBERT_MLFLOW_MODEL_URI="runs:/<run_id>/hf_model"
TASK2_ENCODER_THRESHOLD=0.5
```

**Nota (S3 directo, sin servidor MLflow):**

También puedes apuntar directamente a un prefix en S3, por ejemplo:
```bash
TASK2_ENCODER_ROBERTA_MLFLOW_MODEL_URI="s3://<bucket>/<prefix>/hf_model"
```
El backend soporta descarga desde S3 sin listar el bucket (útil si tu IAM tiene `Deny` para `s3:ListBucket` pero permite `s3:GetObject`).

**Scripts útiles:**
- Subir una carpeta HF a MLflow: `app/backend/scripts/mlflow_log_hf_model.py`

### Dependencias de la app

```
flask==3.0.3
flask-cors==4.0.1
flasgger==0.9.7.1
python-dotenv==1.0.1
```

Frontend: HTML + CSS + JavaScript vanilla (sin frameworks, sin build step).

---

## Despliegue en EC2 (Ubuntu) — guía rápida

1) Instalar dependencias del sistema:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

2) Clonar repo y crear venv:
```bash
git clone <tu_repo>
cd <tu_repo>/app/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Instalar dependencias de inferencia (si usas `model="encoder"`):
```bash
pip install torch transformers safetensors
```

4) Configurar `.env` en la raíz del repo (o exportar env vars). Para S3, idealmente usa un IAM Role en la instancia.

5) Ejecutar (dev):
```bash
python main.py
```

Para producción, usa un WSGI server:
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 main:app
```

## Integrantes

- Álvaro Andrés Ruiz Flórez
- José David Yepes Tumay
- Andrés Julián González Barrera
