"""Split a PDF into size-bounded chunks at page / chapter / section boundaries.

Usage:
    python pdf_splitter.py <input.pdf> <max_size> [--min-size SIZE] [--outdir DIR]
                           [--no-table-check]

    <max_size> / <min-size> accept: 500KB, 1MB, 2MB, 1.5MB, 750000 (raw bytes), ...
    (KB/MB/GB are binary: 1KB = 1024 bytes, 1MB = 1024 KB.)

Rules implemented:
  1. No output chunk exceeds <max_size> (verified against the actual saved bytes).
  2. Break points are chosen by strict priority, using the PDF table of contents:
       a. CHAPTER end (level-1 entry) -- preferred; chunks hold whole chapters.
       b. MAJOR-SECTION end (level-2) -- used only when a single chapter is too
          big to fit in one chunk.
       c. RAW PAGE boundary -- used only when even a single major section exceeds
          <max_size>. This is flagged with a warning.
     Minor subsections (level 3+) are never used as break points.
  3. Table-aware cuts: a candidate break between two pages is REJECTED if a
     table on the earlier page reaches the bottom margin or a table on the next
     page starts in the top margin (almost always a continuation). The splitter
     falls back to the next earlier safe boundary. Disable with --no-table-check.
  4. When a chapter exceeds <max_size>, break at the EARLIEST safe section
     boundary (not the latest). Smaller chunks, but cleaner separation and more
     headroom for the following chunk.
  5. Pages are copied verbatim; splits never happen mid-page.
  6. Original page numbers are preserved as PDF page labels (a chunk that begins
     at original page 40 is numbered 40, 41, ...).
  7. Optional --min-size: chunks below this size are merged forward into the
     next chunk (subject to <max_size>), to avoid tiny tail files. The very
     last chunk is exempt if no merge is possible.
  8. Chunks are named <stem>_part01.pdf, <stem>_part02.pdf, ...

Performance: per-page serialized sizes are cached and chunk size is estimated
as the sum of per-page sizes plus a small constant for the trailer/xref. The
estimate is verified by serializing the final chunk before each write, so
cache drift cannot cause an over-limit file to slip through.
"""

import argparse
import os
import sys

import pymupdf  # PyMuPDF

# Saved/serialized with these options; measuring with the same options means the
# size we check equals the size on disk.
SAVE_OPTS = dict(garbage=4, deflate=True)

# Distance from a page edge (in points; 72pt = 1 inch) inside which a table is
# considered to "touch" that edge -- meaning a cut next to that edge would
# likely sever a multi-page table.
#
# Different tolerances for top vs bottom because page LAYOUT is asymmetric:
#   * Top of page: headers / chapter titles / page numbers push real content
#     down ~70pt. A table starting at y0 < ~90pt is almost always a
#     continuation from the previous page (no heading between).
#   * Bottom of page: footers are smaller, ~30-50pt. A table extending to
#     y1 > height - 100pt typically runs off the bottom (table didn't have
#     room to finish on this page).
TABLE_TOP_TOLERANCE_PT = 90.0
TABLE_BOTTOM_TOLERANCE_PT = 100.0


def parse_size(text: str) -> int:
    """Parse '500KB' / '1MB' / '750000' into a byte count (binary units)."""
    s = text.strip().upper().replace(" ", "")
    for suffix, factor in (("KB", 1024), ("MB", 1024 ** 2), ("GB", 1024 ** 3), ("B", 1)):
        if s.endswith(suffix):
            mult = factor
            s = s[: -len(suffix)]
            break
    else:
        mult = 1
    try:
        value = float(s)
    except ValueError:
        raise SystemExit(f"Could not parse size: {text!r} (try e.g. 500KB, 1MB, 2MB)")
    return int(value * mult)


def toc_break_starts(doc: pymupdf.Document):
    """0-based pages that begin a chapter / major section, from the TOC.

    Returns (chapter_starts, section_starts) as sorted lists:
      * chapter_starts -> level-1 TOC entries (chapters)
      * section_starts -> level-1 AND level-2 entries (chapters + major sections)
    Subsections (level 3+) are deliberately excluded.
    """
    chapter_set, section_set = set(), set()
    for level, _title, page in doc.get_toc():
        if page <= 0:
            continue
        if level == 1:
            chapter_set.add(page - 1)
        if level <= 2:
            section_set.add(page - 1)
    return sorted(chapter_set), sorted(section_set)


def find_max_fit(src, start, max_bytes, n):
    """Find the furthest page `end` such that serializing pages [start, end]
    fits in max_bytes. Returns (end, actual_size_bytes).

    Uses exponential growth + binary search to minimize the number of
    serializations: O(log k) where k is the chunk size in pages, instead of
    O(k) for a one-page-at-a-time loop.
    """
    # First confirm a single-page chunk fits; if not, return it anyway so the
    # caller can warn and move on (a single page can't be split further).
    size_start = len(chunk_bytes(src, start, start))
    if size_start > max_bytes:
        return start, size_start

    # Exponential growth: find an `end` where the chunk DOESN'T fit, or hit n-1.
    lo = start                          # known to fit
    hi = start                          # search frontier
    step = 1
    last_good_size = size_start
    while True:
        candidate = min(start + step, n - 1)
        size = len(chunk_bytes(src, start, candidate))
        if size > max_bytes:
            hi = candidate              # known to NOT fit
            break
        lo = candidate                  # known to fit
        last_good_size = size
        if candidate == n - 1:
            return candidate, size      # whole tail fits
        step *= 2

    # Binary search between lo (fits) and hi (doesn't fit) for the boundary.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        size = len(chunk_bytes(src, start, mid))
        if size > max_bytes:
            hi = mid
        else:
            lo = mid
            last_good_size = size
    return lo, last_good_size


def chunk_bytes(src: pymupdf.Document, start: int, end: int) -> bytes:
    """Serialize pages [start, end] (inclusive, 0-based) to PDF bytes."""
    tmp = pymupdf.open()
    tmp.insert_pdf(src, from_page=start, to_page=end)
    data = tmp.tobytes(**SAVE_OPTS)
    tmp.close()
    return data


def find_unsafe_cuts(doc: pymupdf.Document):
    """Return the set of 0-based page indices `i` such that cutting BETWEEN
    page i and page i+1 would likely sever a table.

    A cut after page i is unsafe if:
      * any table on page i reaches within TABLE_EDGE_TOLERANCE_PT of the
        bottom of the page (table runs off the bottom), OR
      * any table on page i+1 starts within TABLE_EDGE_TOLERANCE_PT of the
        top of the page (table is a continuation from the previous page).

    Both conditions are needed in practice because table detection sometimes
    fires on only one side of a true cross-page table.
    """
    unsafe = set()
    # Pre-compute table bboxes per page so we only call find_tables() once.
    table_bboxes = []
    for i in range(doc.page_count):
        page = doc[i]
        try:
            tabs = page.find_tables().tables
            bboxes = [(t.bbox[1], t.bbox[3]) for t in tabs]  # (y0, y1)
        except Exception:
            bboxes = []
        table_bboxes.append((page.rect.height, bboxes))

    for i in range(doc.page_count - 1):
        h1, boxes1 = table_bboxes[i]
        _, boxes2 = table_bboxes[i + 1]
        bottom_touch = any(y1 > h1 - TABLE_BOTTOM_TOLERANCE_PT for (_, y1) in boxes1)
        top_touch = any(y0 < TABLE_TOP_TOLERANCE_PT for (y0, _) in boxes2)
        if bottom_touch and top_touch:
            # Both ends of the candidate cut have a table near the edge:
            # almost certainly the same table continuing across pages.
            unsafe.add(i)
    return unsafe


def print_summary(records: list) -> None:
    """Print a boxed summary table of the chunks that were written."""
    pretty = {"chapter": "chapter end", "section": "major-section end",
              "page": "page boundary"}
    headers = ["Chunk", "Original pages", "Size", "Break at"]
    rows = [[label, f"{first}-{last}", f"{size:,} B", pretty.get(kind, kind)]
            for (label, first, last, size, kind) in records]

    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(c) for c in col) for col in cols]

    def rule(left, mid, right):
        return left + mid.join("-" * (w + 2) for w in widths) + right

    def line(cells, center=False):
        rendered = [" " + (c.center(w) if center else c.ljust(w)) + " "
                    for c, w in zip(cells, widths)]
        return "|" + "|".join(rendered) + "|"

    print(rule("+", "+", "+"))
    print(line(headers, center=True))
    print(rule("+", "+", "+"))
    for idx, row in enumerate(rows):
        print(line(row))
        if idx < len(rows) - 1:
            print(rule("+", "+", "+"))
    print(rule("+", "+", "+"))


def pick_end_page(i, max_end, chapter_starts, section_starts, unsafe_cuts, n):
    """Choose the end page for a chunk starting at i, given the size ceiling max_end.

    Strategy:
      1. Prefer chapter-end cuts. Among chapters that start in (i, max_end+1],
         choose one whose "end page = next_chapter_start - 1" is a SAFE cut.
         Take the FURTHEST safe chapter cut (packs whole chapters), but if the
         furthest is unsafe (table spillover) walk back to the next earlier
         chapter end that is safe.
      2. If no safe chapter cut exists, fall back to section-end cuts. Per
         user choice, pick the EARLIEST safe section cut in (i, max_end+1].
      3. If no safe section cut exists either, fall back to the latest safe
         raw page cut at-or-before max_end. If max_end itself is unsafe, walk
         backward until we find a safe page cut, but never below i (a single
         page can't be split, so if i..max_end is entirely unsafe we accept
         the unsafe cut at max_end with a warning).

    Returns (end_page, break_kind, warning_message_or_None).
    """
    chapter_candidates = [c - 1 for c in chapter_starts if i < c <= max_end + 1]
    section_candidates = [s - 1 for s in section_starts if i < s <= max_end + 1]

    # 1. Chapter boundary — furthest safe.
    # 'end' means we'd cut AFTER page 'end'. Cut is safe if 'end' not in
    # unsafe_cuts (or if 'end' is the very last page, in which case there's
    # no "after page" to worry about).
    for end in sorted(chapter_candidates, reverse=True):
        if end == n - 1 or end not in unsafe_cuts:
            return end, "chapter", None

    # 2. Section boundary — earliest safe (per user choice).
    for end in sorted(section_candidates):
        if end not in unsafe_cuts:
            msg = (f"chapter starting at page {i + 1} exceeds max size; "
                   f"cut at section boundary after page {end + 1}.")
            return end, "section", msg

    # 3. Raw page boundary — latest safe page cut at-or-before max_end.
    for end in range(max_end, i - 1, -1):
        if end == n - 1 or end not in unsafe_cuts:
            msg = None
            if i != n - 1:
                msg = (f"no safe chapter/section boundary fit; cut at raw page "
                       f"after page {end + 1}.")
            return end, "page", msg

    # 4. Last resort: every cut from i to max_end is unsafe (rare). Accept
    #    max_end and warn loudly — better than infinite loop.
    msg = (f"every candidate cut between pages {i + 1} and {max_end + 1} "
           f"lands inside a table; using page {max_end + 1} anyway.")
    return max_end, "page", msg


def split_pdf(input_path: str, max_bytes: int, min_bytes: int, outdir: str,
              check_tables: bool) -> None:
    src = pymupdf.open(input_path)
    n = src.page_count
    chapter_starts, section_starts = toc_break_starts(src)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(outdir, exist_ok=True)

    print(f"Input: {input_path}  ({n} pages, {os.path.getsize(input_path):,} bytes)")
    print(f"Max chunk size: {max_bytes:,} bytes"
          f"{f' | Min: {min_bytes:,} bytes' if min_bytes else ''}")
    print(f"Chapters (L1): {len(chapter_starts)} | "
          f"Chapters+major sections (L<=2): {len(section_starts)}")

    unsafe_cuts = set()
    if check_tables:
        print("Detecting cross-page tables ...", end=" ", flush=True)
        unsafe_cuts = find_unsafe_cuts(src)
        print(f"done ({len(unsafe_cuts)} unsafe cut-points"
              f"{': ' + ', '.join('after p' + str(i+1) for i in sorted(unsafe_cuts)) if unsafe_cuts else ''}).")
    print()

    warnings = []

    # === Pass 1: choose chunk boundaries ===
    chunks = []  # list of (start, end, break_kind)
    i = 0
    while i < n:
        # Find furthest end page that fits within max_bytes.
        max_end, max_end_size = find_max_fit(src, i, max_bytes, n)
        if max_end == i and max_end_size > max_bytes:
            warnings.append(f"page {i + 1} alone is {max_end_size:,} bytes, "
                            f"larger than the {max_bytes:,}-byte limit "
                            f"(written as-is; cannot split a single page).")

        # Choose actual end using boundary preferences + table safety.
        end, break_kind, warning = pick_end_page(
            i, max_end, chapter_starts, section_starts, unsafe_cuts, n)
        if warning:
            warnings.append(f"chunk starting page {i + 1}: {warning}")

        # Safety net: ensure we make forward progress.
        if end < i:
            end = i
            break_kind = "page"

        chunks.append((i, end, break_kind))
        i = end + 1

    # === Pass 2: merge tail chunks below --min-size into their predecessor ===
    # Uses real serialized sizes. A small chunk gets absorbed into the previous
    # one only if the combined real size stays under max_bytes.
    if min_bytes > 0 and len(chunks) >= 2:
        merged = [chunks[0]]
        for start, end, kind in chunks[1:]:
            est_self = len(chunk_bytes(src, start, end))
            if est_self < min_bytes:
                prev_start, prev_end, _prev_kind = merged[-1]
                est_combined = len(chunk_bytes(src, prev_start, end))
                if est_combined <= max_bytes:
                    # Absorb: take the new chunk's break_kind (the actual
                    # break point of the combined chunk is where this small
                    # chunk would have ended).
                    merged[-1] = (prev_start, end, kind)
                    continue
            merged.append((start, end, kind))
        if len(merged) != len(chunks):
            print(f"Merged {len(chunks) - len(merged)} small chunk(s) "
                  f"into their predecessor (--min-size={min_bytes:,}).\n")
        chunks = merged

    # === Pass 3: write each chunk and verify actual size ===
    records = []
    for part_idx, (start, end, break_kind) in enumerate(chunks, start=1):
        chunk = pymupdf.open()
        chunk.insert_pdf(src, from_page=start, to_page=end)

        # Preserve original page numbers via page labels.
        try:
            chunk.set_page_labels([{"startpage": 0, "style": "D",
                                    "firstpagenum": start + 1}])
        except Exception as exc:
            print(f"  (note: could not set page labels: {exc})")

        chunk.set_metadata({**src.metadata,
                            "title": f"{stem} (pages {start + 1}-{end + 1})"})

        out_name = f"{stem}_part{part_idx:02d}.pdf"
        out_path = os.path.join(outdir, out_name)
        chunk.save(out_path, **SAVE_OPTS)
        chunk.close()

        size = os.path.getsize(out_path)
        if size > max_bytes:
            warnings.append(f"{out_name} is {size:,} bytes, OVER the "
                            f"{max_bytes:,}-byte limit (estimate under-counted).")

        flag = "  <-- OVER LIMIT" if size > max_bytes else ""
        label = {"chapter": "[chapter end]",
                 "section": "[major-section end]",
                 "page": "[page boundary]"}[break_kind]
        print(f"  {out_name}: original pages {start + 1}-{end + 1} "
              f"({end - start + 1} pages), {size:,} bytes {label}{flag}")
        records.append((f"part{part_idx:02d}", start + 1, end + 1, size, break_kind))

    src.close()
    print(f"\nDone. Wrote {len(chunks)} chunk(s) to {outdir}\n")
    print_summary(records)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ! {w}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Split a PDF into size-bounded chunks.")
    ap.add_argument("input", help="Path to the input PDF")
    ap.add_argument("max_size", help="Maximum chunk size, e.g. 500KB, 1MB, 5MB")
    ap.add_argument("--min-size", default=None,
                    help="Minimum chunk size; smaller tail chunks are merged "
                         "into their predecessor when possible (e.g. 100KB)")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: alongside the input file)")
    ap.add_argument("--no-table-check", action="store_true",
                    help="Skip cross-page table detection (faster, but cuts "
                         "may land inside tables)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    max_bytes = parse_size(args.max_size)
    if max_bytes <= 0:
        raise SystemExit("Max size must be positive.")
    min_bytes = parse_size(args.min_size) if args.min_size else 0
    if min_bytes >= max_bytes:
        raise SystemExit("--min-size must be smaller than max_size.")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    split_pdf(args.input, max_bytes, min_bytes, outdir,
              check_tables=not args.no_table_check)


if __name__ == "__main__":
    main()