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
| **Tarea 2** | Extracción de contribuciones | Metodológica, Empírica, Recurso, Conceptual |

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

### 1. Configurar AWS (credenciales)

```bash
aws configure
```

Verificar que quedaron activas:

```bash
aws sts get-caller-identity
```

### 2. Descargar los datos con DVC

Desde la raíz del repositorio:

```bash
dvc pull
```

### 3. Ejecutar la aplicación web

```bash
# Desde la raíz del repositorio
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
  "tasks": ["segmentation", "contributions"]
}
```
Devuelve segmentos etiquetados y fragmentos con contribuciones.

**`GET /api/compare/<analysis_id>`**

Devuelve métricas comparativas de los 3 modelos para el texto analizado.

### Cómo reemplazar los mocks por modelos reales

Los servicios están diseñados para facilitar la transición:

**Tarea 1 — `app/backend/services/segmentation.py`**
```python
def analyze_segments(text: str, model: str) -> dict:
    # Reemplaza _mock_analyze() por _call_real_model()
    return _call_real_model(text, model)  # ← descomentar cuando esté listo
```

**Tarea 2 — `app/backend/services/contributions.py`**
```python
def analyze_contributions(segments: list[dict], model: str) -> dict:
    # Mismo patrón
    return _call_real_model(segments, model)
```

Los stubs tienen comentarios con ejemplos de integración para Hugging Face (encoder), Ollama (LLM open-weight) y OpenAI SDK (API comercial).

### Variables de entorno

```bash
PORT=5000       # Puerto del servidor (por defecto: 5000)
DEBUG=1         # Modo debug de Flask (por defecto: 1)
```

### Dependencias de la app

```
flask==3.0.3
flask-cors==4.0.1
```

Frontend: HTML + CSS + JavaScript vanilla (sin frameworks, sin build step).

---

## Integrantes

- Álvaro Andrés Ruiz Flórez
- José David Yepes Tumay
- Andrés Julián González Barrera
