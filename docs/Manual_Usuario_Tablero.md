# Manual de Usuario — Tablero SciText-ES

**Versión:** 1.0  \
**Audiencia:** Usuarios finales que consumen el tablero ya desplegado (sin instalación local).  \
**Alcance:** Uso del tablero para análisis de documentos científicos en español.

---

## 1. Descripción general

SciText-ES es una aplicación web para analizar textos científicos en español. Permite:

- **Tarea 1:** Segmentación retórica (INTRO, BACK, METH, RES, DISC, CONTR, LIM, CONC).
- **Tarea 2:** Detección de contribuciones científicas (binario: sí/no).
- **Comparación:** Métricas y trade-offs entre tres enfoques de modelos.

El tablero está organizado en 4 vistas principales:

1. **Entrada** — Pegar texto y seleccionar modelo/tareas.
2. **Segmentación** — Resultados por párrafo con etiquetas y confianza.
3. **Contribuciones** — Fragmentos detectados como aportes.
4. **Comparación** — Métricas y tiempo/costo estimado.

---

## 2. Acceso al tablero

Abre en tu navegador la URL suministrada por el equipo. Ejemplo:

```
http://<IP_PUBLICA>:5000
```

Recomendación: usar **Chrome** o **Firefox** actualizados.

**Imagen 1 — Pantalla de acceso**

![Pantalla de acceso](../docs/images/01_acceso.png)

---

## 3. Vista de Entrada (inicio)

### 3.1 Pegar texto

1. Copia el contenido del artículo científico (mínimo recomendado: 250 caracteres).
2. Pega el texto en el campo principal.
3. El sistema separa párrafos cuando hay líneas en blanco.

### 3.2 Selección de modelo

- **Encoder (BETO/RoBERTa)**
  - Rápido y bajo costo.
  - Ideal para pruebas rápidas.

- **Llama 3.3 70B (OpenRouter)**
  - Alta calidad, pero sujeto a latencia y límites de API.
  - Recomendado si necesitas mejor desempeño.

- **API Comercial**
  - Máxima calidad (si está habilitada).
  - Puede requerir clave/API configurada.

### 3.3 Selección de tareas

- **Tarea 1 — Segmentación retórica** (recomendado)
- **Tarea 2 — Extracción de contribuciones** (recomendado)

Puedes ejecutar una o ambas.

### 3.4 Ejecutar análisis

Haz clic en **Analizar documento**. Aparecerá un indicador de carga.

**Imagen 2 — Vista de entrada**

![Vista de entrada](../docs/images/02_entrada.png)

---

## 4. Vista de Segmentación

En esta vista verás:

- Lista de párrafos con su **etiqueta retórica**.
- **Confianza** por párrafo (barra y porcentaje).
- Resumen con:
  - Total de párrafos
  - Total de palabras
  - Confianza promedio
  - Tiempo de análisis

### Interpretación rápida

- **INTRO**: contexto y objetivo del trabajo.
- **BACK**: antecedentes y trabajos previos.
- **METH**: metodología o procedimiento.
- **RES**: resultados y métricas.
- **DISC**: discusión e interpretación.
- **CONTR**: contribución explícita.
- **LIM**: limitaciones y trabajo futuro.
- **CONC**: conclusiones.

**Imagen 3 — Segmentación retórica**

![Segmentación retórica](../docs/images/03_segmentacion.png)

---

## 5. Vista de Contribuciones

Muestra fragmentos del texto donde el sistema detecta aportes científicos.

Para cada fragmento:

- **Is contribution:** verdadero/falso
- **Confianza** del clasificador
- **Highlight**: frase clave destacada (si aplica)

### Consejos

- Si no aparecen contribuciones, prueba con un texto más largo o más explícito.
- La sección de **Metodología** y **Resultados** suele tener más contribuciones.

**Imagen 4 — Contribuciones**

![Contribuciones](../docs/images/04_contribuciones.png)

---

## 6. Vista de Comparación

Resume métricas entre los modelos disponibles:

- **F1, Precisión, Recall** por tarea.
- **Latencia** estimada.
- **Costo por documento** (estimación comparativa).
- **Trade-offs**: pros y contras de cada modelo.

**Imagen 5 — Comparación de modelos**

![Comparación de modelos](../docs/images/05_comparacion.png)

---

## 7. Exportar reporte

En la vista de comparación, puedes usar **Exportar reporte** para guardar un JSON con:

- Resultados de segmentación
- Contribuciones detectadas
- Métricas comparativas

---

## 8. Errores frecuentes y soluciones

### 8.1 “Backend no disponible”
- El servidor no está accesible.
- Verifica conexión o consulta al administrador.

### 8.2 “Timeout llamando al backend”
- El modelo tardó mucho en responder.
- Reintenta o cambia a un modelo más rápido (Encoder).

### 8.3 “Rate limited” (OpenRouter)
- El modelo gratuito está temporalmente limitado.
- Espera y reintenta o cambia de modelo.

---

## 9. Buenas prácticas

- Usa textos con **varios párrafos** y contenido académico real.
- Evita textos demasiado cortos o sin estructura.
- Si necesitas rapidez, usa **Encoder**.
- Para calidad, usa **Llama 3.3 70B** (cuando haya disponibilidad).

---

## 10. Glosario

- **Retórica**: estructura funcional de un texto (introducción, metodología, etc.).
- **F1**: métrica balanceada entre precisión y recall.
- **Recall**: proporción de verdaderos positivos detectados.
- **Precisión**: proporción de detecciones correctas.
- **Latencia**: tiempo de respuesta del modelo.

---

## 11. Soporte

Si tienes problemas, contacta al equipo responsable del despliegue y comparte:

- URL que estabas usando
- Texto de error completo
- Modelo seleccionado
- Fecha y hora del incidente

---

**Fin del manual.**
