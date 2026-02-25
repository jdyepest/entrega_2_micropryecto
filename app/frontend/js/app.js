/**
 * app.js — Lógica principal y navegación
 *
 * Estado de la aplicación almacenado en sessionStorage:
 *   scitext_analysis  → resultado completo del POST /api/analyze
 */

const STATE_KEY = "scitext_analysis";

/* ------------------------------------------------------------------ */
/* Helpers de estado                                                    */
/* ------------------------------------------------------------------ */

function saveAnalysis(data) {
  sessionStorage.setItem(STATE_KEY, JSON.stringify(data));
}

function loadAnalysis() {
  const raw = sessionStorage.getItem(STATE_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function clearAnalysis() {
  sessionStorage.removeItem(STATE_KEY);
}

/* ------------------------------------------------------------------ */
/* Helpers de UI                                                        */
/* ------------------------------------------------------------------ */

function showLoading(msg = "Analizando documento…") {
  const el = document.getElementById("loading-overlay");
  if (!el) return;
  const txt = el.querySelector(".loading-text");
  if (txt) txt.textContent = msg;
  el.classList.add("visible");
}

function hideLoading() {
  const el = document.getElementById("loading-overlay");
  if (el) el.classList.remove("visible");
}

function showError(msg, containerId = "error-container") {
  const el = document.getElementById(containerId);
  if (!el) { console.error(msg); return; }
  el.innerHTML = `<div class="error-banner">⚠️ ${msg}</div>`;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearError(containerId = "error-container") {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = "";
}

/* ------------------------------------------------------------------ */
/* Helpers de render                                                    */
/* ------------------------------------------------------------------ */

const LABEL_COLORS = {
  INTRO: "#1565C0", BACK: "#6A1B9A", METH: "#00838F",
  RES:   "#2E7D32", DISC: "#E65100", CONTR: "#C62828",
  LIM:   "#4E342E", CONC: "#283593",
};

const LABEL_NAMES = {
  INTRO: "Introducción", BACK: "Antecedentes", METH: "Metodología",
  RES:   "Resultados",   DISC: "Discusión",   CONTR: "Contribución",
  LIM:   "Limitaciones", CONC: "Conclusión",
};

function confidenceClass(val) {
  if (val >= 0.80) return "conf-high";
  if (val >= 0.60) return "conf-med";
  return "conf-low";
}

function confPct(val) {
  return Math.round(val * 100) + "%";
}

function renderConfBar(confidence, width = 180) {
  const cls = confidenceClass(confidence);
  const pct = confPct(confidence);
  return `
    <div class="confidence-bar" style="max-width:${width}px">
      <div class="confidence-bar-track">
        <div class="confidence-bar-fill ${cls}" style="width:${pct}"></div>
      </div>
      <span class="confidence-value">${pct}</span>
    </div>`;
}

function labelBadge(label) {
  return `<span class="badge badge-label badge-${label.toLowerCase()}">${label}</span>`;
}

function modelName(model) {
  const names = { encoder: "Encoder (BETO/RoBERTa)", llm: "LLM Open-Weight", api: "API Comercial" };
  return names[model] || model;
}

/* ------------------------------------------------------------------ */
/* Vista 1 — Entrada                                                    */
/* ------------------------------------------------------------------ */

function initInputView() {
  const textarea   = document.getElementById("text-input");
  const counter    = document.getElementById("char-counter");
  const form       = document.getElementById("analyze-form");
  const modelCards = document.querySelectorAll(".model-card");

  if (!textarea || !form) return;

  // Contador de caracteres
  textarea.addEventListener("input", () => {
    const len = textarea.value.length;
    counter.textContent = `${len} caracteres`;
    counter.className = "char-counter" +
      (len < 250 ? " warning" : " ok");
  });

  // Selección de tarjeta de modelo
  modelCards.forEach(card => {
    card.addEventListener("click", () => {
      modelCards.forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      card.querySelector("input[type='radio']").checked = true;
    });
  });

  // Envío del formulario
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const text = textarea.value.trim();
    if (text.length < 50) {
      showError("El texto debe tener al menos 50 caracteres.");
      return;
    }

    const model = form.querySelector("input[name='model']:checked")?.value || "encoder";
    const tasks = [];
    if (form.querySelector("#task-seg")?.checked)   tasks.push("segmentation");
    if (form.querySelector("#task-cont")?.checked)  tasks.push("contributions");

    if (tasks.length === 0) {
      showError("Selecciona al menos una tarea.");
      return;
    }

    showLoading("Analizando documento…");
    try {
      const result = await apiAnalyze(text, model, tasks);
      saveAnalysis(result);
      window.location.href = "segmentation.html";
    } catch (err) {
      hideLoading();
      showError(err.message);
    }
  });
}

/* ------------------------------------------------------------------ */
/* Vista 2 — Segmentación retórica                                      */
/* ------------------------------------------------------------------ */

function initSegmentationView() {
  const analysis = loadAnalysis();

  if (!analysis || !analysis.segmentation) {
    document.getElementById("segments-container").innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <h3>Sin datos de segmentación</h3>
        <p><a href="index.html">Realiza un nuevo análisis</a></p>
      </div>`;
    return;
  }

  const { segments, stats } = analysis.segmentation;

  // Contar por categoría
  const counts = {};
  segments.forEach(s => { counts[s.label] = (counts[s.label] || 0) + 1; });

  // Sidebar: leyenda
  const legendEl = document.getElementById("legend-list");
  if (legendEl) {
    legendEl.innerHTML = Object.entries(LABEL_COLORS).map(([lbl, color]) => `
      <li class="legend-item">
        <div class="legend-dot-label">
          <span class="legend-dot" style="background:${color}"></span>
          <span>${LABEL_NAMES[lbl]}</span>
        </div>
        <span class="legend-count">${counts[lbl] || 0}</span>
      </li>`).join("");
  }

  // Sidebar: estadísticas
  const statsEl = document.getElementById("sidebar-stats");
  if (statsEl) {
    statsEl.innerHTML = `
      <div class="stat-row"><span class="stat-label">Párrafos</span><span class="stat-value">${stats.total_paragraphs}</span></div>
      <div class="stat-row"><span class="stat-label">Palabras</span><span class="stat-value">${stats.total_words.toLocaleString("es")}</span></div>
      <div class="stat-row"><span class="stat-label">Confianza media</span><span class="stat-value">${confPct(stats.avg_confidence)}</span></div>
      <div class="stat-row"><span class="stat-label">Tiempo</span><span class="stat-value">${stats.time_seconds}s</span></div>
    `;
  }

  // Sidebar: badge del modelo
  const modelEl = document.getElementById("model-badge");
  if (modelEl) {
    modelEl.innerHTML = `
      <span class="text-secondary text-small">Modelo usado</span>
      <strong>${modelName(analysis.model)}</strong>`;
  }

  // Segmentos
  const container = document.getElementById("segments-container");
  container.innerHTML = segments.map((seg, i) => `
    <div class="segment-card seg-${seg.label}">
      <div class="segment-header">
        <div class="segment-meta">
          ${labelBadge(seg.label)}
          <span class="segment-index">Párrafo ${i + 1}</span>
        </div>
        <div class="segment-confidence">
          ${renderConfBar(seg.confidence, 200)}
        </div>
      </div>
      <div class="segment-text">${escHtml(seg.text)}</div>
    </div>`).join("");

  // Botón siguiente
  const btnNext = document.getElementById("btn-contributions");
  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (analysis.contributions) {
        window.location.href = "contributions.html";
      } else {
        window.location.href = "index.html";
      }
    });
  }
}

/* ------------------------------------------------------------------ */
/* Vista 3 — Contribuciones                                             */
/* ------------------------------------------------------------------ */

function initContributionsView() {
  const analysis = loadAnalysis();

  if (!analysis || !analysis.contributions) {
    document.getElementById("fragments-container").innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3>Sin datos de contribuciones</h3>
        <p><a href="index.html">Realiza un nuevo análisis</a></p>
      </div>`;
    return;
  }

  const { fragments, stats } = analysis.contributions;

  // Tarjetas de resumen
  const sc = document.getElementById("summary-cards");
  if (sc) {
    sc.innerHTML = `
      <div class="summary-card neutral">
        <div class="summary-card-value">${stats.total_fragments}</div>
        <div class="summary-card-label">Fragmentos analizados</div>
      </div>
      <div class="summary-card positive">
        <div class="summary-card-value">${stats.positive}</div>
        <div class="summary-card-label">Contribuciones encontradas</div>
      </div>
      <div class="summary-card neutral">
        <div class="summary-card-value">${confPct(stats.avg_confidence_positive)}</div>
        <div class="summary-card-label">Confianza media</div>
      </div>
      <div class="summary-card neutral">
        <div class="summary-card-value">${stats.negative}</div>
        <div class="summary-card-label">Sin contribución</div>
      </div>`;
  }

  // Lista de fragmentos
  const container = document.getElementById("fragments-container");
  container.innerHTML = fragments.map((frag, i) => {
    const isPos = frag.is_contribution;
    const bodyText = isPos && frag.highlight
      ? highlightText(frag.text, frag.highlight)
      : escHtml(frag.text);

    return `
      <div class="fragment-card ${isPos ? "" : "no-contribution"}">
        <div class="fragment-header">
          <div class="fragment-header-left">
            <span class="${isPos ? "badge-contribution" : "badge-no-contribution"}">
              ${isPos ? "✓ Contribución detectada" : "— Sin contribución"}
            </span>
            ${isPos ? `<span class="badge-type">${frag.contribution_type}</span>` : ""}
            <span class="para-index">Párrafo ${frag.paragraph_index + 1}</span>
          </div>
          <div class="fragment-header-right">
            <div class="fragment-conf-wrapper">
              ${renderConfBar(frag.confidence, 160)}
            </div>
          </div>
        </div>
        <div class="fragment-body">${bodyText}</div>
        <div class="fragment-footer">
          <span class="footer-label">Sección:</span>
          ${labelBadge(frag.source_label)}
        </div>
      </div>`;
  }).join("");

  // Botón exportar
  const btnExport = document.getElementById("btn-export");
  if (btnExport) {
    btnExport.addEventListener("click", () => exportJson(analysis, "scitext_analysis.json"));
  }
}

/* ------------------------------------------------------------------ */
/* Vista 4 — Comparación                                               */
/* ------------------------------------------------------------------ */

async function initComparisonView() {
  const analysis = loadAnalysis();

  if (!analysis) {
    showError("Sin análisis cargado. Realiza primero un análisis.", "comparison-error");
    return;
  }

  showLoading("Cargando métricas comparativas…");
  try {
    const metrics = await apiCompare(analysis.id);
    hideLoading();
    renderComparison(metrics);
  } catch (err) {
    hideLoading();
    showError(err.message, "comparison-error");
  }

  // Exportar reporte
  const btnExport = document.getElementById("btn-export-report");
  if (btnExport) {
    btnExport.addEventListener("click", async () => {
      try {
        const m = await apiCompare(analysis.id);
        exportJson({ analysis_id: analysis.id, model: analysis.model, ...m }, "scitext_report.json");
      } catch (e) {
        showError(e.message, "comparison-error");
      }
    });
  }
}

function renderComparison(metrics) {
  const { task1_metrics, task2_metrics, cost_per_doc, total_time } = metrics;
  const models = ["encoder", "llm", "api"];
  const modelIcons = { encoder: "⚡", llm: "🧠", api: "☁️" };
  const modelColors = { encoder: "#1565C0", llm: "#7B1FA2", api: "#E65100" };

  // Tabla helper
  function renderTable(tableMetrics, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const metricKeys = ["f1", "precision", "recall", "latency"];
    const metricLabels = { f1: "F1-Score", precision: "Precisión", recall: "Recall", latency: "Latencia (s)" };

    // Encontrar el mejor valor por métrica
    const bestPerMetric = {};
    metricKeys.forEach(k => {
      const vals = models.map(m => tableMetrics[m][k]);
      bestPerMetric[k] = k === "latency" ? Math.min(...vals) : Math.max(...vals);
    });

    el.innerHTML = `
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Modelo</th>
            ${metricKeys.map(k => `<th>${metricLabels[k]}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${models.map(m => `
            <tr>
              <td>
                <div class="model-cell">
                  <span class="model-dot" style="background:${modelColors[m]}"></span>
                  ${modelIcons[m]} ${m.charAt(0).toUpperCase() + m.slice(1)}
                </div>
              </td>
              ${metricKeys.map(k => {
                const val = tableMetrics[m][k];
                const isBest = Math.abs(val - bestPerMetric[k]) < 0.001;
                const display = k === "latency" ? val.toFixed(2) + "s" : (val * 100).toFixed(1) + "%";
                return `<td class="metric-value ${isBest ? "best-value" : ""}">
                  ${display}${isBest ? '<span class="best-star">★</span>' : ""}
                </td>`;
              }).join("")}
            </tr>`).join("")}
        </tbody>
      </table>`;
  }

  renderTable(task1_metrics, "table-task1");
  renderTable(task2_metrics, "table-task2");

  // Gráficos de barras — Costo
  renderBarChart("chart-cost", cost_per_doc, models, modelColors,
    m => `$${cost_per_doc[m].toFixed(4)}`, "USD");

  // Gráficos de barras — Tiempo
  renderBarChart("chart-time", total_time, models, modelColors,
    m => `${total_time[m].toFixed(1)}s`, "Segundos");

  // Trade-offs
  renderTradeoffs(metrics);
}

function renderBarChart(containerId, values, models, colors, labelFn, unit) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const max = Math.max(...models.map(m => values[m]));
  const modelNames = { encoder: "Encoder", llm: "LLM", api: "API" };
  const modelIcons = { encoder: "⚡", llm: "🧠", api: "☁️" };

  el.innerHTML = models.map(m => {
    const pct = Math.round((values[m] / max) * 100);
    return `
      <div class="bar-row">
        <div class="bar-label">${modelIcons[m]} ${modelNames[m]}</div>
        <div class="bar-track">
          <div class="bar-fill bar-${m}" style="width:${pct}%"></div>
        </div>
        <div class="bar-value">${labelFn(m)}</div>
      </div>`;
  }).join("");
}

function renderTradeoffs(metrics) {
  const el = document.getElementById("tradeoff-cards");
  if (!el) return;

  const cards = [
    {
      model: "encoder", icon: "⚡", name: "Encoder",
      color: "#1565C0",
      pros: ["Más rápido (" + metrics.total_time.encoder.toFixed(1) + "s)", "Menor costo ($" + metrics.cost_per_doc.encoder.toFixed(4) + ")", "Sin dependencia de API externa", "Reproducible y auditable"],
      cons: ["Menor F1 en Tarea 2 vs API", "Limitado a idiomas entrenados", "Requiere GPU para escala"],
    },
    {
      model: "llm", icon: "🧠", name: "LLM Open-Weight",
      color: "#7B1FA2",
      pros: ["Balance calidad-costo", "Sin costo por llamada de API", "Flexibilidad de fine-tuning", "Buen recall en Tarea 2"],
      cons: ["Mayor latencia (" + metrics.total_time.llm.toFixed(1) + "s)", "Requiere hardware dedicado", "Más costoso en infraestructura"],
    },
    {
      model: "api", icon: "☁️", name: "API Comercial",
      color: "#E65100",
      pros: ["Mayor F1 general", "Sin setup de infraestructura", "Modelos actualizados", "Mejor en textos cortos"],
      cons: ["Mayor costo por doc ($" + metrics.cost_per_doc.api.toFixed(4) + ")", "Dependencia de proveedor externo", "Sin control sobre el modelo"],
    },
  ];

  el.innerHTML = cards.map(c => `
    <div class="tradeoff-card">
      <div class="tradeoff-card-title" style="color:${c.color}">
        ${c.icon} ${c.name}
      </div>
      <ul class="tradeoff-pros">
        ${c.pros.map(p => `<li>${p}</li>`).join("")}
      </ul>
      <div class="tradeoff-divider"></div>
      <ul class="tradeoff-cons">
        ${c.cons.map(p => `<li>${p}</li>`).join("")}
      </ul>
    </div>`).join("");
}

/* ------------------------------------------------------------------ */
/* Utilidades                                                           */
/* ------------------------------------------------------------------ */

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlightText(fullText, highlight) {
  if (!highlight) return escHtml(fullText);
  const escaped = escHtml(fullText);
  const escapedHighlight = escHtml(highlight);
  // Buscar la subcadena (case-insensitive) y envolverla con <mark>
  const idx = escaped.toLowerCase().indexOf(escapedHighlight.toLowerCase());
  if (idx === -1) return escaped;
  return (
    escaped.slice(0, idx) +
    `<mark>${escaped.slice(idx, idx + escapedHighlight.length)}</mark>` +
    escaped.slice(idx + escapedHighlight.length)
  );
}

function exportJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ */
/* Auto-init según página                                               */
/* ------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  const page = window.location.pathname.split("/").pop() || "index.html";

  if (page === "index.html" || page === "")        initInputView();
  else if (page === "segmentation.html")           initSegmentationView();
  else if (page === "contributions.html")          initContributionsView();
  else if (page === "comparison.html")             initComparisonView();
});
