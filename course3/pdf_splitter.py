"""Split a PDF into size-bounded chunks at page / chapter / section boundaries.

Usage:
    python pdf_splitter.py <input.pdf> <max_size> [--outdir DIR]

    <max_size> accepts: 500KB, 1MB, 2MB, 1.5MB, 750000 (raw bytes), ...
    (KB/MB/GB are binary: 1KB = 1024 bytes, 1MB = 1024 KB.)

Rules implemented:
  1. No output chunk exceeds <max_size> (verified against the actual saved bytes).
  2. Break points are chosen by strict priority, using the PDF table of contents:
       a. CHAPTER end (level-1 entry) -- preferred; chunks hold whole chapters.
       b. MAJOR-SECTION end (level-2) -- used only when a single chapter is too
          big to fit in one chunk, so it must be split internally.
       c. RAW PAGE boundary -- used only when even a single major section exceeds
          <max_size>. This is flagged with a warning (raise --max_size to avoid).
     Minor subsections (level 3+) are never used as break points. No page is
     duplicated across chunks (no overlap).
  3. Pages are copied verbatim, so text / image / layout alignment is untouched.
     Splits never happen mid-page.
  4. Original page numbers are preserved as PDF page labels: a chunk that begins
     at original page 40 is numbered 40, 41, ... in a viewer.
  5. Chunks are named <stem>_part01.pdf, <stem>_part02.pdf, ...

A single page larger than <max_size> cannot be split without breaking rule 3,
so it is written on its own and a warning is printed.
"""

import argparse
import os
import sys

import pymupdf  # PyMuPDF

# Saved/serialized with these options; measuring with the same options means the
# size we check equals the size on disk.
SAVE_OPTS = dict(garbage=4, deflate=True)


def parse_size(text: str) -> int:
    """Parse '500KB' / '1MB' / '2MB' / '750000' into a byte count (binary units)."""
    s = text.strip().upper().replace(" ", "")
    mult = 1
    for suffix, factor in (("KB", 1024), ("MB", 1024 ** 2), ("GB", 1024 ** 3), ("B", 1)):
        if s.endswith(suffix):
            mult = factor
            s = s[: -len(suffix)]
            break
    try:
        value = float(s)
    except ValueError:
        raise SystemExit(f"Could not parse max size: {text!r} (try e.g. 500KB, 1MB, 2MB)")
    return int(value * mult)


def toc_break_starts(doc: pymupdf.Document):
    """0-based pages that begin a chapter / major section, from the TOC.

    Returns (chapter_starts, section_starts):
      * chapter_starts  -> level-1 TOC entries (chapters)
      * section_starts  -> level-1 and level-2 entries (chapters + major sections)
    Subsections (level 3+) are deliberately excluded so we never treat a minor
    subsection start as an acceptable break point.
    """
    chapter_starts, section_starts = set(), set()
    for level, _title, page in doc.get_toc():  # page is 1-based, may be -1 if unknown
        if page <= 0:
            continue
        if level == 1:
            chapter_starts.add(page - 1)
        if level <= 2:
            section_starts.add(page - 1)
    return chapter_starts, section_starts


def print_summary(records: list) -> None:
    """Print a boxed summary table of the chunks that were written.

    records: list of (chunk_label, first_page, last_page, size_bytes, break_kind).
    """
    pretty = {"chapter": "chapter end", "section": "major-section end",
              "page": "page boundary"}
    headers = ["Chunk", "Original pages", "Size", "Break at"]
    rows = [[label, f"{first}–{last}", f"{size:,} B", pretty.get(kind, kind)]
            for (label, first, last, size, kind) in records]

    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(c) for c in col) for col in cols]

    def rule(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(cells, center=False):
        rendered = [" " + (c.center(w) if center else c.ljust(w)) + " "
                    for c, w in zip(cells, widths)]
        return "│" + "│".join(rendered) + "│"

    print(rule("┌", "┬", "┐"))
    print(line(headers, center=True))
    print(rule("├", "┼", "┤"))
    for idx, row in enumerate(rows):
        print(line(row))
        if idx < len(rows) - 1:
            print(rule("├", "┼", "┤"))
    print(rule("└", "┴", "┘"))


def chunk_bytes(src: pymupdf.Document, start: int, end: int) -> bytes:
    """Serialize pages [start, end] (inclusive, 0-based) to PDF bytes."""
    tmp = pymupdf.open()
    tmp.insert_pdf(src, from_page=start, to_page=end)
    data = tmp.tobytes(**SAVE_OPTS)
    tmp.close()
    return data


def split_pdf(input_path: str, max_bytes: int, outdir: str) -> None:
    src = pymupdf.open(input_path)
    n = src.page_count
    chapter_starts, section_starts = toc_break_starts(src)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(outdir, exist_ok=True)

    print(f"Input: {input_path}  ({n} pages, {os.path.getsize(input_path):,} bytes)")
    print(f"Max chunk size: {max_bytes:,} bytes")
    print(f"Chapters (level 1): {len(chapter_starts)} | "
          f"chapters+major sections (level<=2): {len(section_starts)}\n")
    warnings = []

    records = []  # (chunk_label, first_page, last_page, size_bytes)
    part = 1
    i = 0  # 0-based index of the first page of the current chunk
    while i < n:
        # Grow the chunk one page at a time until the next page would overflow.
        end = i
        data = chunk_bytes(src, i, end)
        if len(data) > max_bytes:
            print(f"  ! page {i + 1} alone is {len(data):,} bytes > limit; "
                  f"writing it on its own (cannot split a single page).")
        j = i + 1
        while j < n:
            candidate = chunk_bytes(src, i, j)
            if len(candidate) > max_bytes:
                break
            data, end = candidate, j
            j += 1

        # `end` is now the furthest page that fits by size (the size ceiling).
        # Choose the break point by strict priority:
        #   1. chapter end (level 1)        -> pack whole chapters
        #   2. major-section end (level 2)  -> only if a chapter is too big to fit
        #   3. raw page                     -> only if a section alone exceeds size
        max_end = end
        chapter_candidates = [c for c in chapter_starts if i < c <= max_end + 1]
        section_candidates = [s for s in section_starts if i < s <= max_end + 1]

        if chapter_candidates:
            end = max(chapter_candidates) - 1
            break_kind = "chapter"
        elif section_candidates:
            end = max(section_candidates) - 1
            break_kind = "section"
            warnings.append(
                f"part{part:02d} (pages {i + 1}-{end + 1}) ends mid-chapter at a "
                f"major-section boundary: chapter starting at page {i + 1} is larger "
                f"than the {max_bytes:,}-byte limit.")
        else:
            end = max_end
            break_kind = "page"
            if i != n - 1:  # a lone final page isn't really an abrupt cut
                warnings.append(
                    f"part{part:02d} (pages {i + 1}-{end + 1}) had to be cut at a raw "
                    f"page boundary: a single major section exceeds the "
                    f"{max_bytes:,}-byte limit. Consider a larger --max_size.")

        if not (i <= end <= max_end):      # safety net
            end, break_kind = max_end, "page"

        # Build the final chunk document for this range.
        chunk = pymupdf.open()
        chunk.insert_pdf(src, from_page=i, to_page=end)

        # Preserve original page numbers via page labels (decimal, starting at i+1).
        try:
            chunk.set_page_labels([{"startpage": 0, "style": "D", "firstpagenum": i + 1}])
        except Exception as exc:  # pragma: no cover - depends on PyMuPDF build
            print(f"  (note: could not set page labels: {exc})")

        chunk.set_metadata({**src.metadata,
                            "title": f"{stem} (pages {i + 1}-{end + 1})"})

        out_name = f"{stem}_part{part:02d}.pdf"
        out_path = os.path.join(outdir, out_name)
        chunk.save(out_path, **SAVE_OPTS)
        chunk.close()

        size = os.path.getsize(out_path)
        flag = "  <-- OVER LIMIT" if size > max_bytes else ""
        label = {"chapter": "[chapter end]",
                 "section": "[major-section end]",
                 "page": "[page boundary]"}[break_kind]
        print(f"  {out_name}: original pages {i + 1}-{end + 1} "
              f"({end - i + 1} pages), {size:,} bytes {label}{flag}")

        records.append((f"part{part:02d}", i + 1, end + 1, size, break_kind))
        part += 1
        i = end + 1

    src.close()
    print(f"\nDone. Wrote {part - 1} chunk(s) to {outdir}\n")
    print_summary(records)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ! {w}")


def main() -> None:
    # The summary table uses box-drawing characters; make sure stdout can emit
    # them on Windows consoles that default to a legacy code page (cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Split a PDF into size-bounded chunks.")
    ap.add_argument("input", help="Path to the input PDF")
    ap.add_argument("max_size", help="Maximum chunk size, e.g. 500KB, 1MB, 2MB")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: alongside the input file)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    max_bytes = parse_size(args.max_size)
    if max_bytes <= 0:
        raise SystemExit("Max size must be positive.")

    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.input)))
    split_pdf(args.input, max_bytes, outdir)


if __name__ == "__main__":
    main()
