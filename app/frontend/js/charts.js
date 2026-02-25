/**
 * charts.js — Animaciones de barras y métricas visuales
 *
 * Este módulo se carga en todas las páginas pero solo actúa
 * cuando encuentra elementos relevantes en el DOM.
 */

/**
 * Anima las barras de confianza para que se llenen progresivamente.
 * Busca todos los elementos .confidence-bar-fill con style.width definido.
 */
function animateConfidenceBars() {
  const fills = document.querySelectorAll(".confidence-bar-fill");
  fills.forEach(fill => {
    const targetWidth = fill.style.width;
    fill.style.width = "0%";
    // Trigger reflow
    void fill.offsetWidth;
    requestAnimationFrame(() => {
      fill.style.width = targetWidth;
    });
  });
}

/**
 * Anima los gráficos de barras horizontales (vista 4).
 */
function animateBarCharts() {
  const fills = document.querySelectorAll(".bar-fill");
  fills.forEach(fill => {
    const targetWidth = fill.style.width;
    fill.style.width = "0%";
    void fill.offsetWidth;
    // Escalonar la animación
    const delay = Math.random() * 150;
    setTimeout(() => {
      fill.style.width = targetWidth;
    }, delay);
  });
}

/**
 * Observador de intersección para animar elementos cuando entran en pantalla.
 */
function setupAnimationObserver() {
  if (!("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target;
          if (el.classList.contains("bar-fill") || el.classList.contains("confidence-bar-fill")) {
            const targetWidth = el.dataset.targetWidth || el.style.width;
            el.style.width = "0%";
            void el.offsetWidth;
            requestAnimationFrame(() => { el.style.width = targetWidth; });
          }
          observer.unobserve(el);
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll(".bar-fill, .confidence-bar-fill").forEach(el => {
    el.dataset.targetWidth = el.style.width;
    observer.observe(el);
  });
}

// Ejecutar animaciones iniciales con un pequeño delay para que el CSS ya esté pintado
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    animateConfidenceBars();
    animateBarCharts();
  }, 80);
});
