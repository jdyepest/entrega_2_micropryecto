"""
SciText-ES — Servidor principal (Flask)
Ejecutar: python main.py
Acceder:  http://localhost:5000
"""

import os
import logging
from pathlib import Path
from flask import Flask, send_from_directory
from flask_cors import CORS

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # Optional dependency; backend still works if env vars are exported.
    pass

from routes.analysis import analysis_bp
from routes.comparison import comparison_bp
from flasgger import Swagger
from services.errors import UpstreamServiceError

# ---------------------------------------------------------------------------
# Crear app
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend"),
    static_url_path="",
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_level = (os.environ.get("LOG_LEVEL") or "INFO").upper()
_log_file = (os.environ.get("LOG_FILE") or "").strip()
handlers: list[logging.Handler] = [logging.StreamHandler()]
if _log_file:
    Path(_log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(_log_file))

logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=handlers,
)

# ---------------------------------------------------------------------------
# Swagger (OpenAPI) UI
# ---------------------------------------------------------------------------
Swagger(
    app,
    config={
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec_1",
                "route": "/apispec_1.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/",
    },
    template={
        "swagger": "2.0",
        "info": {
            "title": "SciText-ES API",
            "description": "API para segmentación retórica (Task1) y detección binaria de contribuciones (Task2).",
            "version": "1.0.0",
        },
    },
)

# ---------------------------------------------------------------------------
# Registrar blueprints
# ---------------------------------------------------------------------------
app.register_blueprint(analysis_bp)
app.register_blueprint(comparison_bp)

# ---------------------------------------------------------------------------
# Servir el frontend estático
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    # Sirve cualquier archivo del frontend (html, css, js, etc.)
    full_path = os.path.join(app.static_folder, path)
    if os.path.isfile(full_path):
        return send_from_directory(app.static_folder, path)
    # Fallback a index.html para rutas desconocidas
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return {"error": "Recurso no encontrado."}, 404

@app.errorhandler(UpstreamServiceError)
def upstream_error(e: UpstreamServiceError):
    return {"error": f"Servicio no disponible: {e.service}", "detail": e.detail}, int(e.status_code)

@app.errorhandler(500)
def server_error(e):
    original = getattr(e, "original_exception", None)
    detail = str(original) if original else str(e)
    return {"error": "Error interno del servidor.", "detail": detail}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "1") == "1"
    print(f"[SciText-ES] Servidor iniciando en http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
