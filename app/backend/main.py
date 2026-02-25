"""
SciText-ES — Servidor principal (Flask)
Ejecutar: python main.py
Acceder:  http://localhost:5000
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS

from routes.analysis import analysis_bp
from routes.comparison import comparison_bp

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

@app.errorhandler(500)
def server_error(e):
    return {"error": "Error interno del servidor.", "detail": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "1") == "1"
    print(f"[SciText-ES] Servidor iniciando en http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
