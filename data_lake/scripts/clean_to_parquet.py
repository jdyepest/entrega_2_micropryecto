from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# =========================
# Heurística de "secciones"
# =========================
# Señales rápidas de que el texto parece un documento académico con estructura.
SECTION_RE = re.compile(
    r"\b(resumen|introducci[oó]n|metodolog[ií]a|resultados|discusi[oó]n|conclusiones|referencias)\b",
    flags=re.IGNORECASE,
)

def doc_id_from_path(path: str) -> str:
    """ID estable por documento (no depende del contenido)."""
    return hashlib.sha1(path.encode("utf-8")).hexdigest()

def clean_text(s: str) -> str:
    """
    Limpieza mínima (barata):
    - elimina null bytes (rompen herramientas)
    - normaliza saltos de línea
    - colapsa espacios/tabs repetidos
    - colapsa saltos de línea excesivos
    """
    s = s.replace("\x00", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)        # espacios/tabs repetidos -> 1 espacio
    s = re.sub(r"\n{3,}", "\n\n", s)     # 3+ saltos -> 2 saltos
    return s.strip()

def char_stats(s: str) -> dict:
    """
    Estadísticas para filtrar ruido:
    - letters_ratio: proporción de letras (a-z, ñ, etc.)
      * baja proporción = OCR malo, tablas, símbolos, basura
    - digits_ratio: proporción de dígitos
      * muy alta puede ser logs/tablas, pero no lo filtramos directo
    - spaces_ratio: proporción de espacios/saltos
      * extremos suelen indicar texto raro
    """
    n = len(s)
    if n == 0:
        return {"letters_ratio": 0.0, "digits_ratio": 0.0, "spaces_ratio": 0.0}
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    return {
        "letters_ratio": letters / n,
        "digits_ratio": digits / n,
        "spaces_ratio": spaces / n,
    }

def relevance_score(s: str) -> int:
    """
    Score barato para priorizar textos que parecen papers/informes:
    + secciones académicas (Resumen, Introducción, Metodología, etc.)
    + señales típicas: et al., DOI, Figura/Tabla, citas (20xx)
    NO filtra por score; solo lo guarda para que luego puedas seleccionar.
    """
    score = 0

    # 1) Secciones: contar ocurrencias (con tope para que no domine)
    hits = len(SECTION_RE.findall(s))
    score += min(hits, 6)

    # 2) Señales académicas comunes
    if re.search(r"\bet al\.\b", s, re.IGNORECASE):
        score += 1
    if re.search(r"\bdoi\b", s, re.IGNORECASE):
        score += 1
    if re.search(r"\bfigura\b|\btabla\b", s, re.IGNORECASE):
        score += 1
    if re.search(r"\(\s*(19|20)\d{2}\s*\)", s):  # (2021) etc.
        score += 1

    return score

def write_parquet(rows: list[dict], out_path: Path) -> None:
    """Escribe un shard parquet (compresión zstd)."""
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")

def iter_paths(manifest: Path) -> Iterable[str]:
    """Itera paths desde un archivo manifest (1 path por línea)."""
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                yield p

def main(
    manifest: str,
    out_dir: str,
    shard_size: int = 5000,
    max_docs: int | None = None,
    min_chars: int = 1500,
    min_letters_ratio: float = 0.55,
):
    """
    Pipeline principal.

    FILTROS (estos SÍ descartan archivos):
    1) min_chars:
       - descarta textos muy cortos (ruido, trozos, metadatos, archivos vacíos)
    2) min_letters_ratio:
       - descarta textos con poca letra (OCR malo, tablas, símbolos, basura)

    NO FILTRA (solo guarda para selección posterior):
    - quality_score: score de "parece académico"
    - has_sections: si detecta palabras de secciones
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    shard = 0
    scanned = 0
    kept = 0

    for p in tqdm(iter_paths(Path(manifest)), desc="Cleaning"):
        if max_docs is not None and scanned >= max_docs:
            break
        scanned += 1

        try:
            # Lee bytes y decodifica de manera tolerante (por si hay txt con encoding raro)
            b = Path(p).read_bytes()
            text = b.decode("utf-8", errors="replace")
            text = clean_text(text)

            # -------------------
            # FILTRO 1: longitud
            # -------------------
            if len(text) < min_chars:
                continue

            # ------------------------
            # FILTRO 2: ratio de letras
            # ------------------------
            stats = char_stats(text)
            if stats["letters_ratio"] < min_letters_ratio:
                continue

            # Señales (NO filtran; solo guardan metadatos útiles)
            score = relevance_score(text)
            has_sections = bool(SECTION_RE.search(text))

            rows.append({
                "doc_id": doc_id_from_path(p),
                "path": p,
                "quality_score": score,
                "has_sections": has_sections,
                "letters_ratio": stats["letters_ratio"],
                "digits_ratio": stats["digits_ratio"],
                "spaces_ratio": stats["spaces_ratio"],
                "n_chars": len(text),
                "n_words": len(text.split()),
                "text": text,
            })
            kept += 1

            # Escribe shard cuando acumula shard_size docs "buenos"
            if len(rows) >= shard_size:
                out_path = out_dir / f"part-{shard:05d}.parquet"
                write_parquet(rows, out_path)
                rows = []
                shard += 1

        except Exception:
            # Si un archivo está corrupto o ilegible: se salta
            continue

    # último shard parcial
    if rows:
        out_path = out_dir / f"part-{shard:05d}.parquet"
        write_parquet(rows, out_path)

    print(f"Done. scanned={scanned}, kept={kept}, shards_written={shard + (1 if kept else 0)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--min-letters-ratio", type=float, default=0.55)
    args = ap.parse_args()

    main(
        manifest=args.manifest,
        out_dir=args.out,
        shard_size=args.shard_size,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        min_letters_ratio=args.min_letters_ratio,
    )