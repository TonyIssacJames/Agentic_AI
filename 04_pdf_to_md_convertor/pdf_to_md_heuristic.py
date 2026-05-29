#!/usr/bin/env python3
"""
pdf_to_md.py - A generic PDF -> Markdown converter with real extraction logic.

NO content is hard-coded: everything below is derived from the PDF at run time.

What it does well
-----------------
* SPACING: many LaTeX/academic PDFs glue words together ("Thedominant...").
  This tool rebuilds text from individual glyphs, so spacing is correct.
* SUB/SUPERSCRIPTS: by inspecting per-glyph font size and baseline it recovers
  d_k, d_{model}, n^2, K^T, etc. in both prose and tables.
* PARAGRAPHS: reconstructed from line geometry (vertical gaps + short final
  lines), not dumped as one block per section.
* TABLES: "booktabs" academic tables (horizontal rules only) are found by
  anchoring on their "Table N:" caption and rebuilding columns from text
  alignment, then rendered as GitHub-flavored Markdown. No false tables from the
  title page, references or figures.
* FIGURES (optional, --figures): each figure is cropped from the region above
  its "Figure N:" caption.

Equations
---------
A plain extractor cannot turn rendered math back into LaTeX. Two honest choices:

  * Default: display-equation lines are isolated onto their own line with
    sub/superscripts recovered (readable, but not LaTeX).
  * --latex-ocr: if `pix2tex` is installed locally, each display-equation region
    is cropped and OCR'd into real LaTeX ($$...$$). Needs `pip install pix2tex`
    and `pip install pymupdf`. Runs on CPU (no GPU required), just slower.

If you want a fully automatic, equation-aware conversion with zero coding, run
Meta's Nougat instead (also CPU-capable, just slow):
    pip install nougat-ocr
    nougat your.pdf -o out_dir --full-precision

Usage
-----
    python pdf_to_md.py input.pdf
    python pdf_to_md.py input.pdf -o out.md
    python pdf_to_md.py input.pdf --figures
    python pdf_to_md.py input.pdf --latex-ocr      # real LaTeX if pix2tex present

Dependencies
------------
    pip install pdfplumber
    pip install pymupdf      # for --figures and --latex-ocr
    pip install pix2tex      # for --latex-ocr
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pdfplumber

HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z].{0,80})$")
CAPTION_NUM = re.compile(r"^\d+[:.]?$")
SECTION_HEAD_NUM = re.compile(r"^\d+(?:\.\d+)*$")
MATH_SYMS = set("=∑√∈≤≥×·≠≈→±∞∂∇⊙")

TEXT_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_y_tolerance": 4,
    "join_x_tolerance": 12,
}


@dataclass
class Block:
    top: float
    kind: str          # "md" (already-formatted markdown) | "table"
    payload: object


# --------------------------------------------------------------------------- #
# Glyph-level rendering (spacing + sub/superscripts)
# --------------------------------------------------------------------------- #
def page_body_size(page) -> float:
    sizes = [round(c["size"], 1) for c in page.chars]
    return statistics.mode(sizes) if sizes else 10.0


def _compact_scripts(text: str) -> str:
    text = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", text)
    text = re.sub(r"\^\{([A-Za-z0-9])\}", r"^\1", text)
    return text


def render_chars(chars, body_size) -> str:
    """Render a run of chars on one visual line, recovering scripts + spacing."""
    chars = sorted(chars, key=lambda c: c["x0"])
    if not chars:
        return ""
    norm = [c for c in chars if c["size"] >= 0.85 * body_size] or chars
    base_bottom = statistics.median(c["bottom"] for c in norm)
    base_top = statistics.median(c["top"] for c in norm)
    space_gap = 0.18 * body_size
    out, mode, prev = [], "n", None

    def close():
        nonlocal mode
        if mode != "n":
            out.append("}")
            mode = "n"

    prev_kept = None
    for c in chars:
        if prev_kept is not None and c["text"] == prev_kept["text"] \
                and abs(c["x0"] - prev_kept["x0"]) < 0.6:
            continue  # drop overprinted duplicate glyph
        prev_kept = c
        small = c["size"] < 0.85 * body_size
        role = "n"
        if small:
            if c["bottom"] > base_bottom - 1.6:
                role = "b"
            elif c["top"] < base_top + 1.6:
                role = "p"
            else:
                role = "b"
        if prev is not None:
            gap = c["x0"] - prev["x1"]
            if gap > space_gap and not (small and role != "n" and gap < space_gap * 2.5):
                close()
                out.append(" ")
        if role != mode:
            close()
            if role == "b":
                out.append("_{")
            elif role == "p":
                out.append("^{")
            mode = role
        out.append(c["text"])
        prev = c
    close()
    return _compact_scripts("".join(out)).strip()


def cluster_lines(chars, body_size):
    """Group chars into visual lines keyed on the baseline of full-size glyphs."""
    if not chars:
        return []
    norm = [c for c in chars if c["size"] >= 0.85 * body_size] or chars
    baselines = []
    for c in sorted(norm, key=lambda c: c["bottom"]):
        if baselines and abs(baselines[-1] - c["bottom"]) <= 3:
            continue
        baselines.append(c["bottom"])
    # merge baselines that are very close
    merged = []
    for b in baselines:
        if merged and abs(merged[-1] - b) <= 4:
            continue
        merged.append(b)
    lines = {b: [] for b in merged}
    tol = 0.9 * body_size
    for c in chars:
        nearest = min(merged, key=lambda b: abs(b - c["bottom"]))
        if abs(nearest - c["bottom"]) <= tol:
            lines[nearest].append(c)
    res = []
    for b in merged:
        cs = lines[b]
        if not cs:
            continue
        nn = [c for c in cs if c["size"] >= 0.85 * body_size] or cs
        res.append({
            "top": statistics.median(c["top"] for c in nn),
            "bottom": b,
            "x0": min(c["x0"] for c in cs),
            "x1": max(c["x1"] for c in cs),
            "chars": cs,
        })
    return sorted(res, key=lambda l: l["top"])


# --------------------------------------------------------------------------- #
# Prose / paragraph reconstruction
# --------------------------------------------------------------------------- #
def _is_heading(text):
    m = HEADING_RE.match(text.strip())
    if not m:
        return None
    num, title = m.group(1), m.group(2).strip()
    return f"{'#' * (min(num.count('.') + 1, 6) + 1)} {num} {title}"


def _is_equation(text, line, region_left, region_right):
    words = [w for w in re.split(r"\s+", text) if len(re.sub(r"[^A-Za-z]", "", w)) >= 3]
    has_math = any(s in text for s in MATH_SYMS) or "_" in text or "^" in text
    if not has_math:
        return False
    width = max(region_right - region_left, 1)
    center = (line["x0"] + line["x1"]) / 2
    centered = abs(center - (region_left + region_right) / 2) < 0.20 * width
    tagged = bool(re.search(r"\(\d+\)\s*$", text))
    return (len(words) <= 4) or centered or tagged


def build_blocks(lines, body_size, ocr=None, page=None):
    """Turn clustered lines into markdown blocks (paragraphs/headings/eqs/bullets)."""
    if not lines:
        return []
    region_left = min(l["x0"] for l in lines)
    region_right = max(l["x1"] for l in lines)
    gaps = [lines[i + 1]["top"] - lines[i]["top"] for i in range(len(lines) - 1)]
    typical = statistics.median(gaps) if gaps else body_size * 1.2

    blocks, para = [], []

    def flush():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    for i, ln in enumerate(lines):
        text = render_chars(ln["chars"], body_size)
        if not text:
            continue
        first = ln["chars"][0]["text"]

        h = _is_heading(text)
        if h:
            flush()
            blocks.append(("h", h))
            continue
        if first in "•‣●◦":
            flush()
            blocks.append(("b", "- " + text.lstrip("•‣●◦ ").strip()))
            continue
        if _is_equation(text, ln, region_left, region_right):
            flush()
            eq = None
            if ocr is not None and page is not None:
                eq = ocr(page, ln)
            blocks.append(("eq", eq if eq else text))
            continue

        if para and para[-1].endswith("-"):
            para[-1] = para[-1][:-1] + text
        else:
            # paragraph break heuristics
            if para:
                gap = ln["top"] - lines[i - 1]["top"]
                prev_short = lines[i - 1]["x1"] < region_right - 0.12 * (region_right - region_left)
                if gap > 1.5 * typical or prev_short:
                    flush()
            para.append(text)
    flush()

    md = []
    for kind, text in blocks:
        if kind == "eq":
            md.append(f"$$\n{text}\n$$" if text.strip().startswith("\\") or "\\" in text else text)
        else:
            md.append(text)
    return md


# --------------------------------------------------------------------------- #
# Tables (caption-anchored, script-aware cells)
# --------------------------------------------------------------------------- #
def _line_metrics(ws):
    ws = sorted(ws, key=lambda w: w["x0"])
    gaps = [ws[i + 1]["x0"] - ws[i]["x1"] for i in range(len(ws) - 1)]
    toks = [w["text"] for w in ws]
    short_num = sum(1 for t in toks
                    if t.strip("().,:;[]").replace(".", "").replace(",", "").isdigit()
                    or len(t) <= 3)
    return len(ws), (max(gaps) if gaps else 0.0), (short_num / len(toks) if toks else 0.0)


def _table_band(word_lines, cap_top, page_h):
    """Return (grid_top, bottom): where the table grid starts (after the caption
    description) and where it ends (first full-width prose line after it)."""
    state, grid_top, bottom = "cap", None, page_h - 50
    for top, ws in word_lines:
        if top <= cap_top + 1:
            continue
        n, maxgap, frac = _line_metrics(ws)
        tabular = (n <= 7) or (maxgap > 35) or (frac >= 0.4)
        if state == "cap":
            if tabular:
                state, grid_top = "grid", top
        elif state == "grid":
            if (not tabular) and n >= 9:
                bottom = top - 2
                break
    return grid_top, bottom


def _cell_text(page, bbox, body_size):
    if not bbox:
        return ""
    x0, t, x1, b = bbox
    cs = [c for c in page.chars
          if c["x0"] >= x0 - 0.5 and c["x1"] <= x1 + 0.5
          and t - 0.5 <= (c["top"] + c["bottom"]) / 2 <= b + 0.5]
    return render_chars(cs, body_size)


def _accept_table(rows, min_rows=2, min_cols=2):
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if len(rows) < min_rows:
        return False
    if max((len(r) for r in rows), default=0) < min_cols:
        return False
    nonempty = [c.strip() for r in rows for c in r if (c or "").strip()]
    total = sum(len(r) for r in rows) or 1
    if len(nonempty) / total < 0.35:
        return False
    if sum(1 for c in nonempty if len(c) <= 2) / len(nonempty) > 0.55:
        return False
    return True


def _drop_empty_columns(rows):
    if not rows:
        return rows
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    keep = [j for j in range(ncols) if any(r[j].strip() for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def _has_small(page, bbox, body_size):
    if not bbox:
        return False
    x0, t, x1, b = bbox
    return any(c["size"] < 0.85 * body_size for c in page.chars
               if c["x0"] >= x0 - 0.5 and c["x1"] <= x1 + 0.5
               and t - 0.5 <= (c["top"] + c["bottom"]) / 2 <= b + 0.5)


def _cell_value(page, bbox, raw_text, body_size):
    """Use robust column text by default; re-render only compact/script cells so
    multi-word headers (e.g. "BLEU", "Training Cost (FLOPs)") stay intact while
    data cells recover scripts (O(n^2.d), d_k)."""
    t = (raw_text or "").replace("\n", " ").strip()
    if not bbox:
        return t
    x0, ct, x1, cb = bbox
    tight = (x1 - x0) < 0.45 * page.width and (cb - ct) < 2.0 * body_size
    long_words = sum(1 for w in t.split() if len(re.sub(r"[^A-Za-z]", "", w)) >= 3)
    if tight and long_words <= 1 and _has_small(page, bbox, body_size):
        return _cell_text(page, bbox, body_size)
    return t


def find_caption_tables(page, body_size, max_band=430.0):
    words = page.extract_words(x_tolerance=1.5)
    if not words:
        return []
    word_lines = {}
    for w in words:
        word_lines.setdefault(round(w["top"]), []).append(w)
    word_lines = sorted(word_lines.items())
    for _, ws in word_lines:
        ws.sort(key=lambda w: w["x0"])
    left = min(w["x0"] for w in words)

    captions, headings = [], []
    for top, ws in word_lines:
        if ws[0]["text"] == "Table" and len(ws) > 1 and CAPTION_NUM.match(ws[1]["text"]):
            captions.append(top)
        if abs(ws[0]["x0"] - left) < 4 and SECTION_HEAD_NUM.match(ws[0]["text"]):
            headings.append(top)

    results = []
    for cap_top in captions:
        grid_top, band_bottom = _table_band(word_lines, cap_top, page.height)
        if grid_top is None:
            continue
        below = [h for h in headings if h > grid_top + 5]
        boundary = min([min(below) if below else page.height - 50,
                        band_bottom, cap_top + max_band])
        crop = page.crop((max(left - 8, 0), cap_top, page.width - 40, boundary))
        try:
            tbls = crop.find_tables(table_settings=TEXT_TABLE_SETTINGS)
        except Exception:
            tbls = []
        best, best_cells = None, 0
        for t in tbls:
            cells = sum(len(r) for r in t.extract())
            if cells > best_cells:
                best, best_cells = t, cells
        if not best:
            continue
        extracted = best.extract(x_tolerance=1.5)
        rows = []
        for row_obj, row_txt in zip(best.rows, extracted):
            cells = []
            for cell_bbox, cell_txt in zip(row_obj.cells, row_txt):
                cells.append(_cell_value(page, cell_bbox, cell_txt, body_size))
            rows.append(cells)
        if not _accept_table(rows):
            continue
        cap_chars = [c for c in page.chars
                     if cap_top - 1 <= (c["top"] + c["bottom"]) / 2 < best.bbox[1] - 1]
        cap_text = " ".join(render_chars(l["chars"], body_size)
                            for l in cluster_lines(cap_chars, body_size))
        results.append((cap_top, best.bbox[1], best.bbox[3], cap_text, rows))
    return sorted(results, key=lambda r: r[0])


def table_to_markdown(rows):
    rows = _drop_empty_columns(rows)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    norm = [r + [""] * (ncols - len(r)) for r in rows]

    def esc(c):
        return c.replace("|", "\\|").strip()

    header = norm[0]
    if sum(1 for c in header if c.strip()) < max(2, ncols // 2):
        header = [f"Col {i+1}" for i in range(ncols)]
        body = norm
    else:
        body = norm[1:]
    out = ["| " + " | ".join(esc(c) for c in header) + " |",
           "| " + " | ".join("---" for _ in range(ncols)) + " |"]
    for r in body:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Optional pix2tex equation OCR
# --------------------------------------------------------------------------- #
def make_ocr():
    try:
        from pix2tex.cli import LatexOCR
        import fitz  # noqa: F401  (needed for rasterizing)
    except Exception as e:
        print(f"  [latex-ocr] disabled ({e}); install: pip install pix2tex pymupdf",
              file=sys.stderr)
        return None
    model = LatexOCR()
    import fitz
    from PIL import Image
    import io

    def ocr(page, line):
        try:
            x0 = max(line["x0"] - 6, 0)
            x1 = min(line["x1"] + 6, page.width)
            top = max(line["top"] - 4, 0)
            bot = min(line["bottom"] + 4, page.height)
            doc = fitz.open(page.pdf.stream) if hasattr(page, "pdf") else None
        except Exception:
            doc = None
        try:
            d = fitz.open(PAGE_PDF_PATH[0])
            rect = fitz.Rect(x0, top, x1, bot)
            pix = d[page.page_number - 1].get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            latex = model(img).strip()
            return f"$$ {latex} $$" if latex else None
        except Exception as ex:
            print(f"  [latex-ocr] line skipped: {ex}", file=sys.stderr)
            return None
    return ocr


PAGE_PDF_PATH = [None]  # set in convert() so the OCR rasterizer can reopen the PDF


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #
def region_blocks(page, top, bottom, body_size, ocr):
    chars = [c for c in page.chars if c["top"] >= top - 0.5 and c["bottom"] <= bottom + 1]
    lines = cluster_lines(chars, body_size)
    return build_blocks(lines, body_size, ocr=ocr, page=page)


def convert_page(page, ocr=None):
    body = page_body_size(page)
    tables = find_caption_tables(page, body)
    out, cursor = [], 0.0
    for (cap_top, grid_top, grid_bottom, cap_text, rows) in tables:
        if grid_top < cursor:
            continue
        out.extend(region_blocks(page, cursor, cap_top, body, ocr))
        if cap_text.strip():
            out.append(f"**{cap_text.strip()}**")
        md = table_to_markdown(rows)
        if md:
            out.append(md)
        cursor = grid_bottom
    out.extend(region_blocks(page, cursor, page.height, body, ocr))
    return out


def convert(pdf_path, latex_ocr=False, page_markers=False):
    PAGE_PDF_PATH[0] = os.path.abspath(pdf_path)
    ocr = make_ocr() if latex_ocr else None
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if page_markers:
                parts.append(f"<!-- page {i + 1} -->")
            parts.extend(convert_page(page, ocr=ocr))
    return "\n\n".join(p for p in parts if p and p.strip()) + "\n"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def export_figures(pdf_path, out_dir, dpi=180):
    try:
        import fitz
    except ImportError:
        print("  [figures] PyMuPDF not installed; skipping.", file=sys.stderr)
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved, zoom = [], dpi / 72.0
    with pdfplumber.open(pdf_path) as pl:
        doc = fitz.open(pdf_path)
        for pidx, page in enumerate(pl.pages):
            caps = [w["top"] for w in page.extract_words() if w["text"].startswith("Figure")]
            if not caps:
                continue
            top, bottom = 50.0, min(caps) - 4
            if bottom - top < 40:
                continue
            rect = fitz.Rect(50, top, page.width - 50, bottom)
            pix = doc[pidx].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
            name = f"figure_page{pidx + 1}.png"
            pix.save(os.path.join(out_dir, name))
            saved.append((pidx + 1, os.path.join("figures", name)))
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generic PDF -> Markdown converter.")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--output")
    ap.add_argument("--figures", action="store_true", help="export cropped figure PNGs")
    ap.add_argument("--latex-ocr", action="store_true",
                    help="use pix2tex for real LaTeX equations (needs pix2tex + pymupdf)")
    ap.add_argument("--page-markers", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.pdf):
        print(f"error: file not found: {args.pdf}", file=sys.stderr)
        return 2
    out_path = args.output or os.path.splitext(args.pdf)[0] + ".md"
    print(f"Converting {args.pdf} -> {out_path}")
    md = convert(args.pdf, latex_ocr=args.latex_ocr, page_markers=args.page_markers)
    if args.figures:
        fig_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "figures")
        saved = export_figures(args.pdf, fig_dir)
        if saved:
            md += "\n## Figures\n\n"
            for pageno, rel in saved:
                md += f"![Figure from page {pageno}]({rel})\n\n"
            print(f"  exported {len(saved)} figure image(s) -> {fig_dir}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Done. Wrote {len(md):,} characters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
