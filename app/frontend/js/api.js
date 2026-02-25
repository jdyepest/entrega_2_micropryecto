/**
 * api.js — Cliente HTTP para el backend SciText-ES
 *
 * Estrategia de fallback:
 *   1. Intenta llamar al backend real (http://localhost:5000).
 *   2. Si el servidor no está disponible (Failed to fetch, CORS, timeout),
 *      genera los datos localmente con el mock de navegador.
 *   3. El resto de la app (app.js) recibe exactamente el mismo formato
 *      sin importar si los datos vienen del servidor o del mock local.
 */

const API_BASE = "http://localhost:5000";

/* ================================================================
   Mock de navegador (replica la lógica del backend en Python)
   ================================================================ */

const _MOCK = (() => {

  // ── Configuración de modelos ──────────────────────────────────
  const MODELS = {
    encoder: { delay: 500,  task1F1: 0.83, task2F1: 0.78, cost: 0.001 },
    llm:     { delay: 2000, task1F1: 0.79, task2F1: 0.82, cost: 0.008 },
    api:     { delay: 1500, task1F1: 0.87, task2F1: 0.91, cost: 0.035 },
  };

  // ── Keywords por categoría retórica ──────────────────────────
  const KEYWORDS = {
    INTRO: ["introducción","este trabajo","en este artículo","el presente","motivación","objetivo","propósito","presente trabajo","este estudio"],
    BACK:  ["antecedentes","trabajos previos","revisión","estado del arte","investigaciones anteriores","han propuesto","fue propuesto","en la literatura","según","de acuerdo con"],
    METH:  ["metodología","método","procedimiento","experimento","implementación","arquitectura","entrenamiento","corpus","dataset","conjunto de datos","evaluamos","utilizamos","se utilizó","fine-tuning","hiperparámetros","configuración"],
    RES:   ["resultado","tabla","figura","obtuvo","obtuvimos","rendimiento","accuracy","f1","precisión","recall","métricas","muestra","se observa","se puede ver","en la tabla","en la figura"],
    DISC:  ["discusión","análisis","interpretamos","esto sugiere","esto indica","podemos inferir","se debe a","explicar","probable","parece","comparado con","en comparación"],
    CONTR: ["contribución","aporte","propuesta","novedad","innovación","original","nueva","nuevo","presentamos","proponemos","a diferencia de","primer trabajo"],
    LIM:   ["limitación","limitaciones","trabajo futuro","futuras investigaciones","no se consideró","fuera del alcance","restricción","sesgo"],
    CONC:  ["conclusión","conclusiones","en conclusión","en resumen","resumiendo","concluimos","hemos mostrado","hemos demostrado","finalmente","en definitiva"],
  };

  const ORDER = ["INTRO","BACK","METH","RES","DISC","CONTR","LIM","CONC"];

  const LABEL_TO_TYPE = {
    METH:"Metodológica", RES:"Empírica", CONTR:"Recurso",
    DISC:"Conceptual",   INTRO:"Conceptual", BACK:"Conceptual",
    LIM:"Conceptual",    CONC:"Metodológica",
  };

  // ── PRNG determinista (xorshift) ─────────────────────────────
  function seededRng(seed) {
    let s = (seed >>> 0) || 1;
    return () => {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      return ((s >>> 0) / 0xFFFFFFFF);
    };
  }

  function strHash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < Math.min(str.length, 120); i++) {
      h ^= str.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    return h;
  }

  function jitter(rng, base, spread) {
    return Math.min(0.99, Math.max(0.10, base + (rng() * 2 - 1) * spread));
  }

  // ── Dividir texto en párrafos ─────────────────────────────────
  function splitParagraphs(text) {
    return text.split(/\n\s*\n/)
      .map(p => p.trim())
      .filter(p => p.length > 20);
  }

  // ── Puntaje por keywords ──────────────────────────────────────
  function keywordScores(para) {
    const lower = para.toLowerCase();
    const scores = {};
    for (const [lbl, kws] of Object.entries(KEYWORDS)) {
      scores[lbl] = kws.filter(k => lower.includes(k)).length;
    }
    return scores;
  }

  // ── Label + confianza para un párrafo ────────────────────────
  function labelParagraph(para, idx, total, model, rng) {
    const scores = keywordScores(para);
    const maxScore = Math.max(...Object.values(scores));
    let label, baseConf;

    if (idx === 0 && maxScore < 2) {
      label = "INTRO"; baseConf = 0.82;
    } else if (idx === total - 1 && maxScore < 2) {
      label = "CONC"; baseConf = 0.80;
    } else if (maxScore > 0) {
      label = Object.entries(scores).sort((a,b) => b[1]-a[1])[0][0];
      baseConf = 0.72 + Math.min(maxScore * 0.05, 0.22);
    } else {
      const ratio = idx / Math.max(total - 1, 1);
      label = ORDER[Math.round(ratio * (ORDER.length - 1))];
      baseConf = 0.62 + rng() * 0.10;
    }

    const noise = model === "encoder" ? (rng()*0.12-0.06)
                : model === "llm"     ? (rng()*0.09-0.04)
                :                       (rng()*0.06-0.02);

    return { label, confidence: Math.round(jitter(rng, baseConf + noise, 0) * 100) / 100 };
  }

  // ── Highlight: primera frase de ≥6 palabras ──────────────────
  function findHighlight(text) {
    const sentences = text.split(/[.;]/);
    for (const s of sentences) {
      const words = s.trim().split(/\s+/);
      if (words.length >= 6) return words.slice(0, 10).join(" ");
    }
    return text.slice(0, 80).trim();
  }

  // ── Generar UUID simple ───────────────────────────────────────
  function uuid() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  // ── sleep ─────────────────────────────────────────────────────
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Mock /api/analyze ─────────────────────────────────────────
  async function mockAnalyze(text, model, tasks) {
    const cfg = MODELS[model] || MODELS.encoder;
    await sleep(cfg.delay);

    const rng   = seededRng(strHash(text.slice(0,100)) ^ strHash(model));
    const paras = splitParagraphs(text);
    const total = paras.length || 1;
    const id    = uuid();

    // Tarea 1 ─ segmentación
    const segments = paras.map((para, i) => {
      const { label, confidence } = labelParagraph(para, i, total, model, rng);
      return { paragraph_index: i, text: para, label, confidence };
    });

    const totalWords  = paras.reduce((s, p) => s + p.split(/\s+/).length, 0);
    const avgConf     = Math.round(segments.reduce((s,g) => s + g.confidence, 0) / total * 1000) / 1000;
    const timeSec     = Math.round((cfg.delay / 1000 + rng() * 0.4) * 100) / 100;

    const segmentation = {
      segments,
      stats: { total_paragraphs: total, total_words: totalWords, avg_confidence: avgConf, time_seconds: timeSec },
    };

    // Tarea 2 ─ contribuciones
    const HIGH = new Set(["CONTR","METH","RES"]);
    const MED  = new Set(["DISC","CONC","INTRO"]);

    const fragments = segments.map(seg => {
      const baseProb = HIGH.has(seg.label) ? 0.80 : MED.has(seg.label) ? 0.40 : 0.20;
      const probNoise = model === "encoder" ? rng()*0.15-0.10
                      : model === "llm"     ? rng()*0.13-0.05
                      :                       rng()*0.10;
      const isContrib = rng() < baseProb + probNoise;

      let baseC = isContrib
        ? 0.78 + rng()*0.18 + (model === "api" ? 0.04 : model === "encoder" ? -0.03 : 0)
        : 0.30 + rng()*0.30;
      baseC = Math.min(0.99, Math.max(0.10, Math.round(baseC * 100) / 100));

      const ctype    = LABEL_TO_TYPE[seg.label] || "Conceptual";
      const highlight = isContrib ? findHighlight(seg.text) : "";

      return {
        paragraph_index: seg.paragraph_index,
        text:           seg.text,
        is_contribution: isContrib,
        contribution_type: isContrib ? ctype : null,
        confidence:     baseC,
        highlight,
        source_label:   seg.label,
      };
    });

    const positives = fragments.filter(f => f.is_contribution);
    const avgCP = positives.length
      ? Math.round(positives.reduce((s,f) => s + f.confidence, 0) / positives.length * 1000) / 1000
      : 0;

    const contributions = {
      fragments,
      stats: { total_fragments: fragments.length, positive: positives.length, negative: fragments.length - positives.length, avg_confidence_positive: avgCP },
    };

    return {
      id,
      model,
      model_name: { encoder:"Encoder (BETO/RoBERTa)", llm:"LLM Open-Weight", api:"API Comercial" }[model],
      segmentation: tasks.includes("segmentation") ? segmentation : null,
      contributions: tasks.includes("contributions") ? contributions : null,
    };
  }

  // ── Mock /api/compare ─────────────────────────────────────────
  async function mockCompare(analysisId) {
    const rng = seededRng(strHash(analysisId));

    function jit(base, spread = 0.04) {
      return Math.round(Math.min(0.99, Math.max(0.50, base + (rng()*2-1)*spread)) * 100) / 100;
    }
    function latJit(base) {
      return Math.round((base + (rng()*2-1)*0.2) * 100) / 100;
    }

    const cfgs = {
      encoder: { t1f1: 0.83, t2f1: 0.78, delay: 0.5,  cost: 0.001 },
      llm:     { t1f1: 0.79, t2f1: 0.82, delay: 2.0,  cost: 0.008 },
      api:     { t1f1: 0.87, t2f1: 0.91, delay: 1.5,  cost: 0.035 },
    };

    const t1 = {}, t2 = {};
    for (const [m, c] of Object.entries(cfgs)) {
      const f1a = jit(c.t1f1); t1[m] = { f1: f1a, precision: jit(f1a, 0.03), recall: jit(f1a, 0.03), latency: latJit(c.delay) };
      const f1b = jit(c.t2f1); t2[m] = { f1: f1b, precision: jit(f1b, 0.03), recall: jit(f1b, 0.03), latency: latJit(c.delay * 0.6) };
    }

    return {
      analysis_id: analysisId,
      task1_metrics: t1,
      task2_metrics: t2,
      cost_per_doc: { encoder: Math.round(cfgs.encoder.cost*(0.9+rng()*0.2)*10000)/10000, llm: Math.round(cfgs.llm.cost*(0.9+rng()*0.2)*10000)/10000, api: Math.round(cfgs.api.cost*(0.9+rng()*0.2)*10000)/10000 },
      total_time:   { encoder: Math.round((t1.encoder.latency+t2.encoder.latency)*100)/100, llm: Math.round((t1.llm.latency+t2.llm.latency)*100)/100, api: Math.round((t1.api.latency+t2.api.latency)*100)/100 },
    };
  }

  return { mockAnalyze, mockCompare };
})();

/* ================================================================
   Funciones públicas con fallback automático
   ================================================================ */

/**
 * POST /api/analyze — con fallback a mock de navegador.
 */
async function apiAnalyze(text, model, tasks) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000); // timeout 8s

    const resp = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model, tasks }),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Error del servidor (${resp.status})`);
    }
    return resp.json();

  } catch (fetchErr) {
    // Servidor no disponible → usar mock de navegador
    console.warn("[SciText-ES] Backend no disponible, usando mock local:", fetchErr.message);
    return _MOCK.mockAnalyze(text, model, tasks);
  }
}

/**
 * GET /api/compare/:id — con fallback a mock de navegador.
 */
async function apiCompare(analysisId) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);

    const resp = await fetch(`${API_BASE}/api/compare/${analysisId}`, {
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Error del servidor (${resp.status})`);
    }
    return resp.json();

  } catch (fetchErr) {
    console.warn("[SciText-ES] Backend no disponible, usando mock local:", fetchErr.message);
    return _MOCK.mockCompare(analysisId);
  }
}
