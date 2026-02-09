# Micropoyecto MAIA  
## Análisis de documentos científicos en español (Tema 1 – 2026)

Este repositorio corresponde a un **micropoyecto MAIA** cuyo objetivo es **desarrollar una solución computacional** al problema definido en el proyecto:

> **Análisis automático de documentos científicos en español**, abordando  
> (i) la **segmentación y clasificación retórica** y  
> (ii) la **extracción de contribuciones científicas**.

El repositorio está organizado para **desarrollar, evaluar y documentar la solución**, de acuerdo con lo definido en el documento del proyecto.

---

## Propósito del repositorio

Este proyecto implementa:

- Preparación y curaduría de un corpus científico en español.
- Modelos para **segmentación y clasificación retórica**.
- Modelos para **detección de contribuciones científicas**.
- Evaluación comparativa entre modelos entrenados y modelos de lenguaje.
- Scripts reproducibles para experimentación y análisis de resultados.

El enfoque es **académico y experimental**, orientado a entregar una solución funcional y evaluable.

---

## Estructura del proyecto

La siguiente estructura organiza el desarrollo completo de la solución.  
Las carpetas se crearán conforme avance el trabajo.

```text
.
├── datos/                     # Datos del proyecto
│   └── core/                  # Corpus científico crudo (CORE)
│
├── src/                       # Código fuente principal
│   ├── preprocessing/         # Limpieza, normalización y segmentación
│   ├── task1_rhetorical/      # Segmentación y clasificación retórica
│   ├── task2_contributions/   # Extracción de contribuciones científicas
│   ├── models/                # Definición y carga de modelos
│   └── utils/                 # Funciones auxiliares comunes
│
├── experiments/               # Experimentos y configuraciones
│   ├── task1/                 # Experimentos de clasificación retórica
│   └── task2/                 # Experimentos de extracción de contribuciones
│
├── evaluation/                # Evaluación y análisis de resultados
│   ├── metrics/               # Métricas cuantitativas
│   └── error_analysis/        # Análisis cualitativo de errores
│
├── notebooks/                 # Análisis exploratorio y pruebas
│
├── configs/                   # Configuraciones de modelos y experimentos
│
├── README.md                  # Documentación principal
└── .gitignore
```

---

## Descripción de componentes clave

### `datos/`
Contiene el corpus de documentos científicos en español utilizado como entrada del sistema.

### `src/`
Implementa la lógica central de la solución:
- Preprocesamiento del texto.
- Modelos de clasificación retórica.
- Modelos de detección de contribuciones.

### `experiments/`
Scripts para ejecutar experimentos controlados y comparables.

### `evaluation/`
Cálculo de métricas, matrices de confusión y análisis de errores.

### `notebooks/`
Exploración de datos, pruebas de modelos y análisis intermedios.

---

## Alcance del proyecto

Este repositorio cubre **todo el flujo de solución** definido en el proyecto:

- Desde documentos científicos crudos
- Hasta resultados evaluados y comparables



---

## Integrantes

- **Álavara**
- **José**
- **Julián**

---
## Cómo lanzar el proyecto

### 1) Configurar AWS (credenciales)

Configura las credenciales de AWS suministradas:

```bash
aws configure
```

Verifica que quedaron activas correctamente:

```bash
aws sts get-caller-identity
```

### 2) Descargar los datos con DVC

Desde la raíz del repositorio:

```bash
dvc pull
```