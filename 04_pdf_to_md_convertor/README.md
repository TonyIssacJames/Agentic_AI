# PDF → Markdown Converter

Converts a PDF to GitHub-flavored Markdown with caption-anchored table
detection, paragraph reflow, heading promotion, and optional figure export.

**Main script:** `pdf_to_md.py`

---

## Requirements

- Python 3.10+ (tested with Python 3.13)
- Packages pinned in `requirements.txt`

### Key dependencies

| Package | Purpose |
|---|---|
| `transformers` (4.x) | Nougat OCR model (VisionEncoderDecoder) |
| `torch` (CPU build) | model inference |
| `PyMuPDF` (`fitz`) | renders PDF pages to images |
| `pillow`, `numpy` | image handling |
| `nltk`, `python-Levenshtein` | required by Nougat's tokenizer post-processing |

### Important notes

- `transformers` is pinned to the 4.x line (`4.57.6`). Nougat is **NOT** compatible
  with `transformers` 5.x — the image processor/tokenizer raise validation
  errors (`do_thumbnail` / `do_crop_margin` expected `bool`, got `None`). Do not
  upgrade `transformers` past 4.x for this project.
- `torch` is the CPU-only build (`torch==2.12.0+cpu`). `requirements.txt` includes
  the PyTorch CPU index URL so a plain `pip install -r requirements.txt`
  resolves it correctly.

---

## How it works

This converter is a thin pipeline around the **Nougat** model
(`facebook/nougat-small`), an OCR transformer from Meta AI specifically trained
on academic PDFs. Nougat reads a **page image** and writes **Markdown** — not
character-by-character OCR, but a learned page-to-markup translation that
preserves equations, tables, headings, and reading order.

The script's job is to (1) turn the PDF into page images Nougat can see,
(2) drive the model page-by-page, and (3) rewrite Nougat's native
*Mathpix Markdown* into clean GitHub-flavored Markdown.

### Pipeline (block diagram)

```
        input.pdf
            │
            ▼
   ┌─────────────────────┐
   │ PyMuPDF  (fitz)     │   open PDF, iterate pages
   │ render_pages()      │   rasterize each page at --dpi
   └─────────┬───────────┘
             │   raw pixel buffer
             ▼
   ┌─────────────────────┐
   │ Pillow  (PIL.Image) │   wrap buffer as an RGB image
   └─────────┬───────────┘
             │   List[PIL.Image]  (one per page)
             ▼
   ┌─────────────────────────────┐
   │ transformers AutoProcessor  │   crop margins, resize, pad,
   │ (Nougat image processor)    │   normalize → pixel tensor
   └─────────┬───────────────────┘
             │   torch.Tensor  (pixel_values)
             ▼
   ┌─────────────────────────────────────┐
   │ Nougat VisionEncoderDecoderModel    │
   │  • encoder: Swin-style image vit    │   image  →  visual features
   │  • decoder: autoregressive LM       │   features → token stream
   │  weights: .\nougat-small (local)    │
   │  runtime: torch  (CPU build here)   │
   └─────────┬───────────────────────────┘
             │   token ids
             ▼
   ┌─────────────────────────────┐
   │ processor.batch_decode +    │   ids → Mathpix Markdown text
   │ post_process_generation     │   ( \(…\), \[…\], \begin{tabular} )
   └─────────┬───────────────────┘
             │   raw .mmd-style markdown
             ▼
   ┌─────────────────────────────┐
   │ to_github_markdown()        │   \begin{tabular} → | pipe | tables |
   │ (this script)               │   \(x\)  → $x$
   │                             │   \[x\]  → $$x$$
   │                             │   strip \hline, \centering, blank runs
   └─────────┬───────────────────┘
             │
             ▼
        input.md   (GitHub-flavored Markdown)
```

### What each piece does

| Layer | Library / function | Role |
|---|---|---|
| **PDF I/O** | `PyMuPDF` (imported as `fitz`) | Opens the PDF, walks pages, renders each page to a pixel buffer at the chosen DPI. No text extraction — we want the page as Nougat *sees* it. |
| **Image wrapper** | `Pillow` (`PIL.Image`) | Turns the raw RGB buffer from PyMuPDF into a `PIL.Image` object — that's the input type the HF processor expects. |
| **Numeric backbone** | `numpy` | Used implicitly by Pillow and the HF processor for array math during image preprocessing. |
| **Preprocessing** | `transformers.AutoProcessor` (Nougat's image processor) | Margin-crops the page, resizes to the model's expected resolution, pads, rescales pixel values, normalizes — outputs a `torch.Tensor` of shape `(1, 3, H, W)`. |
| **Model runtime** | `torch` (CPU build) | Provides the tensor engine + autograd-free inference (`torch.no_grad()`). CPU build is enough; GPU is auto-detected if present. |
| **The model** | `VisionEncoderDecoderModel` loaded from `facebook/nougat-small` | A vision encoder (Swin Transformer) + text decoder (BART-style). The encoder turns the page image into a sequence of visual tokens; the decoder generates Markdown tokens autoregressively. |
| **Tokenizer / detok** | Nougat tokenizer (inside `AutoProcessor`) | Decodes the generated token IDs back to text and runs `post_process_generation` for light fixups. |
| **NLP helpers** | `nltk`, `python-Levenshtein` | Pulled in by the Nougat tokenizer's post-processing (sentence segmentation, fuzzy fixups). Not called directly by this script, but the tokenizer fails to load without them. |
| **Markdown rewrite** | `to_github_markdown()` in `pdf_to_md.py` | Converts Nougat's *Mathpix Markdown* into GitHub-flavored Markdown: `\begin{tabular}` → pipe tables, `\(…\)` → `$…$`, `\[…\]` → `$$…$$`, strips LaTeX scaffolding. Skipped if you pass `--raw`. |
| **CLI / orchestration** | `argparse` + `main()` / `convert()` | Parses flags, walks pages, runs OCR per page, joins the page outputs, writes the `.md` file. |

### End-to-end flow, step by step

1. **Parse arguments.** `--pages 1-8` becomes a sorted list of 0-based page
   indices; `--dpi` controls the render resolution; `--model` points at the
   local `.\nougat-small` folder so weights aren't re-downloaded.
2. **Render pages → images.** `render_pages()` opens the PDF with PyMuPDF,
   calls `page.get_pixmap(dpi=…)` for each page, and wraps the bitmap as a
   `PIL.Image`. Output: `[(page_number, PIL.Image), …]`.
3. **Load the model once.** `load_model()` loads the processor and the
   `VisionEncoderDecoderModel` from the local weights folder, patches any
   `None` preprocessing flags (a known incompatibility between older Nougat
   checkpoints and newer `transformers`), and moves the model to `cuda` if
   available, else `cpu`.
4. **OCR each page.** `ocr_page()` runs the image through the processor to
   get `pixel_values`, then calls `model.generate(...)` to decode up to
   `--max-new-tokens` Markdown tokens. The result is *Mathpix Markdown* — a
   Markdown dialect where math is in `\(…\)` / `\[…\]` and tables are raw
   `\begin{tabular}` blocks.
5. **Join + post-process.** The per-page strings are joined with blank
   lines. Unless `--raw` is set, `to_github_markdown()` rewrites tables and
   math into GitHub-flavored Markdown and strips LaTeX scaffolding
   (`\hline`, `\centering`, `\begin{table}`, etc.).
6. **Write `<input>.md`.** UTF-8, one file, ready for any Markdown renderer
   (GitHub, VS Code preview, MkDocs, etc.).

### Why these specific pins matter

- **`transformers==4.57.6`** — Nougat's processor config in the published
  checkpoint leaves some boolean flags as `null`. `transformers` 4.x tolerates
  this (and the script also force-fills them); `transformers` 5.x rejects it
  with a validation error. Hence the hard 4.x pin.
- **`torch==2.12.0+cpu`** — keeps install size and runtime predictable on
  machines without a GPU. The script auto-uses CUDA if `torch.cuda.is_available()`
  returns true, so swapping in a CUDA build later "just works."
- **`HF_HUB_OFFLINE=1` + `--model .\nougat-small`** — once weights are
  downloaded, this combination guarantees `transformers` never tries the
  network, so the run is fully reproducible and air-gap friendly.

---

## Setup (Windows / PowerShell)

1. **Create the virtual environment** (already created in `.venv`):

   ```powershell
   py -m venv .venv
   ```

2. **Install dependencies:**

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. **Download the Nougat model weights locally (CRUCIAL STEP).**
   With the venv active (or using its `huggingface-cli`), download the model
   into a local `.\nougat-small` folder so `--model` can point at it:

   ```powershell
   pip install huggingface_hub
   huggingface-cli download facebook/nougat-small --local-dir nougat-small
   ```

   > On newer `huggingface_hub` the command is `hf download facebook/nougat-small --local-dir nougat-small` — both do the same thing.

   #### If the download is rate-limited or times out

   You only need to do anything if it **FAILS** with a rate-limit error (HTTP 429)
   or times out. A free Hugging Face token raises the limit and speeds it up:

   1. Create a token at <https://huggingface.co/settings/tokens>
      (a "Read" token is enough).
   2. Authenticate, either by logging in:

      ```powershell
      huggingface-cli login
      ```

      (paste the token when asked), **OR** set it for the session:

      ```powershell
      set HF_TOKEN=hf_your_token_here
      ```

   3. Re-run the same download command — it resumes where it left off:

      ```powershell
      huggingface-cli download facebook/nougat-small --local-dir nougat-small
      ```

4. **(Optional) Activate the environment:**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

---

## Model weights

The Nougat model lives in `.\nougat-small` (downloaded in setup step 3). Point
`--model` at that folder so it is loaded locally and not re-downloaded from the
Hugging Face Hub.

---

## Usage

Run with the venv's Python (or activate first, then use `python`):

```powershell
# basic run -> writes attention_is_all_you_need.md
python pdf_to_md.py attention_is_all_you_need.pdf --model .\nougat-small

# writes input.md
.\.venv\Scripts\python.exe pdf_to_md.py input.pdf --model .\nougat-small

# custom output
python pdf_to_md.py input.pdf --model .\nougat-small -o out.md

# subset of pages
python pdf_to_md.py input.pdf --model .\nougat-small --pages 1-8

# higher render DPI
python pdf_to_md.py input.pdf --model .\nougat-small --dpi 200
```

### Options

| Option | Description |
|---|---|
| `-o`, `--output` | output `.md` path (default: `<input>.md`) |
| `--model` | HF model id or LOCAL path (use `.\nougat-small` here; `facebook/nougat-base` is higher quality but slower) |
| `--dpi` | page render DPI (default `150`; try `200` if blurry) |
| `--pages` | subset, e.g. `'1-8'` or `'2,5,9'` (1-based) |
| `--max-new-tokens` | decoder token cap per page (default `3584`) |

**Speed (CPU):** each page is decoded autoregressively — roughly seconds to
~1–2 min per page. Test with `--pages 1-2` before running a whole PDF.

---

## Example

```powershell
.\.venv\Scripts\python.exe pdf_to_md.py attention_is_all_you_need.pdf --model .\nougat-small --pages 1-2
```

---

## Additional setup notes / alternative workflows

> Verbatim — do not change commands.

### Core method

Download the model locally, force offline mode so `transformers`/HF won't try
the network, then run the converter against the local model folder.

> **NOTE:** `--model` path below is `.\models\nougat-small`; adjust it to wherever you
> downloaded the model (this project uses `.\nougat-small`).

```powershell
pip install huggingface_hub
huggingface-cli download facebook/nougat-small --local-dir nougat-small
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python pdf_to_md.py attention_is_all_you_need.pdf --model .\models\nougat-small
```

### Alternate method

Uses the standalone Nougat CLI instead of this script. `albumentations` is
pinned to `1.3.1` for compatibility with `nougat-ocr`. `--full-precision` helps CPU.

```powershell
pip install "albumentations==1.3.1"
nougat attention_is_all_you_need.pdf -o . --full-precision
```

**Step 1** — PDF → `.mmd` (Nougat CLI):

```powershell
pip install nougat-ocr
nougat attention_is_all_you_need.pdf -o . --full-precision
```

**Step 2** — `.mmd` → `.md` (the converter):

```powershell
python mmd_to_md.py attention_is_all_you_need.mmd
```

#### Full two-step run

```powershell
nougat attention_is_all_you_need.pdf -o . --full-precision
python mmd_to_md.py attention_is_all_you_need.mmd
```

---

## Appendix — sample run (verbatim terminal output)

A successful end-to-end run on `attention_is_all_you_need.pdf` (15 pages, CPU)
looks like the transcript below. Use it as a sanity reference — the printed
DPI, page count, device, and "Done." line confirm the script picked up the
flags you actually intended. (Easy to miss when vibe-coding: the `dpi=150`
default and the `running on: cpu` line are worth eyeballing.)

```bash
(.venv) D:\git_repos\Agentic_AI\04_pdf_to_md_convertor>python pdf_to_md.py attention_is_all_you_need.pdf --model .\nougat-small
Rendering pages from attention_is_all_you_need.pdf (dpi=150) ...
  15 page(s) to process.
Loading model .\nougat-small (first run downloads weights) ...
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
  running on: cpu
  OCR page 1 ...
  OCR page 2 ...
  OCR page 3 ...
  OCR page 4 ...
  OCR page 5 ...
  OCR page 6 ...
  OCR page 7 ...
  OCR page 8 ...
  OCR page 9 ...
  OCR page 10 ...
  OCR page 11 ...
  OCR page 12 ...
  OCR page 13 ...
  OCR page 14 ...
  OCR page 15 ...
Done. Wrote attention_is_all_you_need.md (39,933 characters).
```

### What to check in the output

| Line | What it tells you |
|---|---|
| `Rendering pages ... (dpi=150)` | DPI actually used. If you passed `--dpi 200` and still see `150`, the flag didn't land. |
| `15 page(s) to process.` | Page count after `--pages` filtering. A small number here when you expected the whole PDF means `--pages` is narrower than you thought. |
| `Loading model .\nougat-small` | Confirms the **local** weights folder is being used, not a Hugging Face Hub download. |
| `Using a slow image processor ...` | Harmless `transformers` warning — the slow (Python) image processor is intentional for Nougat compatibility. |
| `running on: cpu` | Device auto-detected. Shows `cuda` only if a GPU + CUDA torch build are present. |
| `OCR page N ...` (one per page) | Progress heartbeat. On CPU each line can take seconds to ~1–2 min. |
| `Done. Wrote ....md (N characters)` | Final success line + output filename + size. A suspiciously small character count usually means decoding hit `--max-new-tokens` early. |

---

## Appendix 2 — the heuristic converter (`pdf_to_md_heuristic.py`)

A second, **ML-free** converter ships alongside the Nougat one. It does the
same job — PDF → Markdown — but via **geometry and font heuristics** instead
of a neural model. Keep it around: it's the right tool for several cases
where Nougat is overkill or too slow.

### What it does

It opens the PDF with `pdfplumber`, reads the embedded character glyphs
(position, font size, baseline), and reconstructs Markdown by analyzing layout:

| Feature | How it's recovered |
|---|---|
| **Word spacing** | Inter-glyph x-gap relative to body font size — fixes the classic `"Thedominant…"` glue-up in LaTeX PDFs. |
| **Sub/superscripts** (`d_k`, `n^2`, `K^T`) | Glyphs smaller than the body size whose baseline sits below (sub) or above (super) the median line. |
| **Paragraphs** | Line clustering by baseline + breaks on large vertical gaps or short final lines. |
| **Headings** (`3.1 Background`) | Regex on numbered prefix at the start of a line. |
| **Tables** | Anchored on `"Table N:"` captions, then `pdfplumber.find_tables()` with text-alignment strategy inside the band below the caption. |
| **Figures** (`--figures`) | Crops the region above each `"Figure N:"` caption to PNG. |
| **Equations** | By default, isolated on their own line (no real LaTeX). With `--latex-ocr`, each display equation is cropped and OCR'd to LaTeX via `pix2tex`. |

### Nougat vs heuristic — at a glance

|   | `pdf_to_md.py` (Nougat) | `pdf_to_md_heuristic.py` |
|---|---|---|
| **Approach** | Vision transformer reads page **images** | Reads embedded text glyphs + geometry |
| **Deps** | `torch`, `transformers`, `PyMuPDF`, ~250 MB model | Just `pdfplumber` (+ optional `pix2tex`, `pymupdf`) |
| **Speed (CPU)** | Seconds to ~1–2 min **per page** | A second or two **per page** |
| **Equations** | Real LaTeX (`$$…$$`) out of the box | Plain text by default; LaTeX only with `--latex-ocr` |
| **Works on scanned PDFs?** | Yes (it's OCR) | **No** — needs embedded text |
| **Fidelity on weird layouts** | Better — model handles it | Worse — heuristics can misclassify |
| **Reproducibility** | Model-version-dependent | Deterministic (pure code) |

### When to run which

Use this as a quick decision guide so the heuristic path doesn't get forgotten:

| Situation | Run this |
|---|---|
| Math-heavy academic PDF, equations matter, quality > speed | **Nougat** — `python pdf_to_md.py paper.pdf --model .\nougat-small` |
| Scanned PDF / image-only PDF (no embedded text) | **Nougat** — heuristic can't read scans |
| Text-based PDF, prose + tables are the goal, no need for LaTeX | **Heuristic** — fast, no model, no GPU. `python pdf_to_md_heuristic.py paper.pdf` |
| Bulk run over many PDFs and you want it to finish today | **Heuristic** first; only re-run Nougat on the few that needed math fidelity |
| Need real LaTeX but only on a handful of equations | **Heuristic** with `--latex-ocr` (requires `pip install pix2tex pymupdf`) |
| You also want figure images extracted | **Heuristic** with `--figures` — Nougat doesn't export figure crops |
| Air-gapped / no model weights available | **Heuristic** — pure `pdfplumber` works without HF Hub or weights |
| Reproducible CI runs (same input → same output, bit-for-bit) | **Heuristic** — Nougat outputs can shift with model/transformers versions |

### Heuristic usage

```powershell
# basic run -> writes attention_is_all_you_need.md
python pdf_to_md_heuristic.py attention_is_all_you_need.pdf

# custom output path
python pdf_to_md_heuristic.py attention_is_all_you_need.pdf -o out.md

# also export cropped figure PNGs into .\figures\
python pdf_to_md_heuristic.py attention_is_all_you_need.pdf --figures

# real LaTeX for display equations (needs pix2tex + pymupdf)
python pdf_to_md_heuristic.py attention_is_all_you_need.pdf --latex-ocr

# add <!-- page N --> markers between pages (handy for diffing)
python pdf_to_md_heuristic.py attention_is_all_you_need.pdf --page-markers
```

### Heuristic options

| Option | Description |
|---|---|
| `-o`, `--output` | output `.md` path (default: `<input>.md`) |
| `--figures` | crop the region above each `"Figure N:"` caption to PNG in `.\figures\` |
| `--latex-ocr` | run `pix2tex` on each display equation to recover real LaTeX |
| `--page-markers` | insert `<!-- page N -->` between pages |

### Extra deps for the heuristic path

```powershell
pip install pdfplumber         # required
pip install pymupdf            # for --figures and --latex-ocr (already installed for the Nougat path)
pip install pix2tex            # only for --latex-ocr
```

> **Rule of thumb:** start with the **heuristic** for any text-based PDF —
> if the math, tables, or layout come out wrong, *then* fall back to
> **Nougat**. Don't pay Nougat's CPU cost when geometry would have been
> enough.
