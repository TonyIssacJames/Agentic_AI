#!/usr/bin/env python3
"""
mmd_to_md.py - Convert Nougat's .mmd (Mathpix Markdown) into GitHub-flavored
Markdown (.md): \\(...\\)/\\[...\\] math -> $...$/$$...$$, LaTeX \\begin{tabular}
-> Markdown pipe tables, and strip junk bytes.

Self-contained (no torch / no model needed) - pure text processing.

Usage
-----
    python mmd_to_md.py input.mmd                 # -> input.md
    python mmd_to_md.py input.mmd -o output.md
"""

from __future__ import annotations
import argparse
import os
import re
import sys


def _strip_junk(text):
    text = text.replace("\x00", "")
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def _clean_cell(c):
    c = c.strip()
    c = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{(.*?)\}", r"\1", c)
    c = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", c)
    c = c.replace("\\hline", "").replace("\\\\", "").replace("\\&", "&")
    return c.strip()


def _expand_row(row, ncols):
    parts = re.split(r"(?<!\\)&", row)
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


def mmd_to_markdown(text):
    text = _strip_junk(text)
    text = _convert_tables(text)
    text = _convert_math(text)
    text = re.sub(r"\\begin\{table\}|\\end\{table\}|\\centering", "", text)
    text = re.sub(r"(?m)^\s*&\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert Nougat .mmd to GitHub-flavored .md")
    ap.add_argument("mmd", help="input .mmd file")
    ap.add_argument("-o", "--output", help="output .md (default: same name, .md)")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.mmd):
        print(f"error: file not found: {args.mmd}", file=sys.stderr)
        return 2
    raw = open(args.mmd, "rb").read().decode("utf-8", "replace")
    md = mmd_to_markdown(raw)
    out_path = args.output or os.path.splitext(args.mmd)[0] + ".md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Done. Wrote {out_path} ({len(md):,} characters).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
