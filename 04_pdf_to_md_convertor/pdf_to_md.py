#!/usr/bin/env python3
"""
pdf_to_md.py - Convert a PDF to Markdown (LaTeX equations + tables) with the
Nougat OCR model. Pip-installable, runs on CPU (no GPU required).

Nougat is a single model trained to turn academic PDF pages into Markdown,
including real equations and tables. Its native output is "Mathpix Markdown"
(math as \\(...\\) / \\[...\\], tables as LaTeX \\begin{tabular}). Plain Markdown
viewers show that as raw code, so by default this script POST-PROCESSES Nougat's
output into GitHub-flavored Markdown: $...$ / $$...$$ math and pipe tables.
Use --raw to keep Nougat's native LaTeX output instead.

Install (CPU is fine)
---------------------
    pip install "transformers>=4.36" pillow pymupdf
    pip install torch --index-url https://download.pytorch.org/whl/cpu
Weights auto-download from Hugging Face on first run (~250 MB nougat-small).

Usage
-----
    python pdf_to_md.py input.pdf                 # GitHub-flavored Markdown
    python pdf_to_md.py input.pdf --raw           # Nougat's native LaTeX/.mmd
    python pdf_to_md.py input.pdf --model facebook/nougat-base   # higher quality
    python pdf_to_md.py input.pdf --pages 1-3     # quick test on a few pages
    python pdf_to_md.py input.pdf --dpi 200

Offline / blocked network
-------------------------
Pre-download the model on a machine with access:
    huggingface-cli download facebook/nougat-small --local-dir nougat-small
copy the folder over, then:
    set HF_HUB_OFFLINE=1   (Windows;  export on macOS/Linux)
    python pdf_to_md.py input.pdf --model .\\nougat-small
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional


NOUGAT_FLAG_DEFAULTS = {
    "do_crop_margin": True,
    "do_resize": True,
    "do_thumbnail": True,
    "do_align_long_axis": False,
    "do_pad": True,
    "do_rescale": True,
    "do_normalize": True,
}


# --------------------------------------------------------------------------- #
# Dependencies / IO helpers
# --------------------------------------------------------------------------- #
def _require(modules):
    missing = []
    for mod, pip_name in modules:
        try:
            __import__(mod)
        except Exception:
            missing.append(pip_name)
    if missing:
        print("Missing dependencies: " + ", ".join(sorted(set(missing))), file=sys.stderr)
        print("Install them with:", file=sys.stderr)
        print('    pip install "transformers>=4.36" pillow pymupdf', file=sys.stderr)
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu",
              file=sys.stderr)
        return False
    return True


def parse_pages(spec: Optional[str]):
    """'1-8' / '3' / '2-5,9' -> sorted 0-based indices, or None for all pages."""
    if not spec:
        return None
    idx = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            idx.update(range(int(a) - 1, int(b)))
        elif part:
            idx.add(int(part) - 1)
    return sorted(idx)


def render_pages(pdf_path, dpi=150, page_range=None):
    """Render PDF pages to PIL images (PyMuPDF + Pillow; no torch needed)."""
    import fitz
    from PIL import Image

    images = []
    doc = fitz.open(pdf_path)
    indices = page_range if page_range is not None else range(len(doc))
    for i in indices:
        if i < 0 or i >= len(doc):
            continue
        pix = doc[i].get_pixmap(dpi=dpi)
        images.append((i + 1, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
    return images


# --------------------------------------------------------------------------- #
# Nougat model
# --------------------------------------------------------------------------- #
def load_model(model_name):
    import torch
    from transformers import AutoProcessor, VisionEncoderDecoderModel

    try:
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_name)
    # Some saved Nougat configs leave boolean preprocessing flags as null, which
    # newer transformers reject. Fill any None ones with Nougat's defaults.
    ip = getattr(processor, "image_processor", processor)
    for attr, default in NOUGAT_FLAG_DEFAULTS.items():
        if hasattr(ip, attr) and getattr(ip, attr) is None:
            setattr(ip, attr, default)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return processor, model, device


def ocr_page(processor, model, device, image, max_new_tokens):
    import torch

    pixel_values = processor(images=image, return_tensors="pt",
                             **NOUGAT_FLAG_DEFAULTS).pixel_values
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            min_length=1,
            max_new_tokens=max_new_tokens,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )
    seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return processor.post_process_generation(seq, fix_markdown=False)


# --------------------------------------------------------------------------- #
# Post-processing: Nougat (Mathpix) -> GitHub-flavored Markdown
# --------------------------------------------------------------------------- #
def _strip_junk(text):
    text = text.replace("\x00", "")
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def _clean_cell(c):
    c = c.strip()
    c = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{(.*?)\}", r"\1", c)
    # inline math first, so \mathbf etc. stay valid LaTeX inside the math span
    c = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", c)
    c = c.replace("\\hline", "").replace("\\\\", "").replace("\\&", "&")
    return c.strip()


def _expand_row(row, ncols):
    parts = re.split(r"(?<!\\)&", row)   # split on & but not \&
    cells = []
    for p in parts:
        m = re.search(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*?)\}", p)
        if m:
            cells.append(_clean_cell(m.group(2)))
            cells.extend([""] * (int(m.group(1)) - 1))
        else:
            cells.append(_clean_cell(p))
    if len(cells) < ncols:
        cells += [""] * (ncols - len(cells))
    return cells[:ncols]


def _tabular_to_md(colspec, body):
    ncols = sum(1 for ch in colspec if ch in "lcr")
    body = re.sub(r"\\cline\{[^}]*\}", "", body).replace("\\hline", "")
    rows = [r.strip() for r in body.split("\\\\") if r.strip()]
    grid = [_expand_row(r, ncols) for r in rows]
    grid = [g for g in grid if any(c.strip() for c in g)]
    if not grid:
        return ""
    out = ["| " + " | ".join(grid[0]) + " |",
           "| " + " | ".join(["---"] * ncols) + " |"]
    for g in grid[1:]:
        out.append("| " + " | ".join(g) + " |")
    return "\n".join(out)


def _convert_tables(text):
    return re.sub(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                  lambda m: _tabular_to_md(m.group(1), m.group(2)), text, flags=re.S)


def _convert_math(text):
    text = re.sub(r"\\\[(.+?)\\\]",
                  lambda m: "\n$$" + m.group(1).strip() + "$$\n", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)",
                  lambda m: "$" + m.group(1).strip() + "$", text, flags=re.S)
    return text


def to_github_markdown(text):
    text = _strip_junk(text)
    text = _convert_tables(text)
    text = _convert_math(text)
    text = re.sub(r"\\begin\{table\}|\\end\{table\}|\\centering", "", text)
    text = re.sub(r"(?m)^\s*&\s*", "", text)   # drop leaked author "&" separators
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def convert(pdf_path, model_name="facebook/nougat-small", dpi=150,
            pages=None, max_new_tokens=3584, raw=False):
    page_idx = parse_pages(pages)
    print(f"Rendering pages from {pdf_path} (dpi={dpi}) ...")
    images = render_pages(pdf_path, dpi=dpi, page_range=page_idx)
    print(f"  {len(images)} page(s) to process.")

    print(f"Loading model {model_name} (first run downloads weights) ...")
    processor, model, device = load_model(model_name)
    print(f"  running on: {device}")

    parts = []
    for pageno, img in images:
        print(f"  OCR page {pageno} ...", flush=True)
        try:
            md = ocr_page(processor, model, device, img, max_new_tokens)
        except Exception as e:
            md = f"<!-- page {pageno}: OCR failed: {e} -->"
        parts.append(md.strip())

    text = "\n\n".join(p for p in parts if p) + "\n"
    if not raw:
        text = to_github_markdown(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="PDF -> Markdown (LaTeX + tables) with Nougat OCR (CPU OK).")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default="facebook/nougat-small",
                    help="facebook/nougat-small (default, faster) or facebook/nougat-base")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pages", help="subset, e.g. '1-8' or '2,5,9' (1-based)")
    ap.add_argument("--max-new-tokens", type=int, default=3584)
    ap.add_argument("--raw", action="store_true",
                    help="keep Nougat's native LaTeX/Mathpix output (no GitHub-Markdown conversion)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.pdf):
        print(f"error: file not found: {args.pdf}", file=sys.stderr)
        return 2
    needed = [("fitz", "pymupdf"), ("PIL", "pillow"),
              ("torch", "torch"), ("transformers", "transformers>=4.36")]
    if not _require(needed):
        return 3

    out_path = args.output or os.path.splitext(args.pdf)[0] + ".md"
    md = convert(args.pdf, model_name=args.model, dpi=args.dpi, pages=args.pages,
                 max_new_tokens=args.max_new_tokens, raw=args.raw)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Done. Wrote {out_path} ({len(md):,} characters).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
