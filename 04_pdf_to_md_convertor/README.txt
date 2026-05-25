PDF -> Markdown Converter
=========================

Converts a PDF to GitHub-flavored Markdown with caption-anchored table
detection, paragraph reflow, heading promotion, and optional figure export.

Main script: pdf_to_md.py


Requirements
------------
- Python 3.10+ (tested with Python 3.13)
- Packages pinned in requirements.txt

Key dependencies:
- transformers (4.x) : Nougat OCR model (VisionEncoderDecoder)
- torch (CPU build)  : model inference
- PyMuPDF (fitz)     : renders PDF pages to images
- pillow, numpy      : image handling
- nltk, python-Levenshtein : required by Nougat's tokenizer post-processing

IMPORTANT NOTES:
- transformers is pinned to the 4.x line (4.57.6). Nougat is NOT compatible
  with transformers 5.x -- the image processor/tokenizer raise validation
  errors (do_thumbnail / do_crop_margin expected bool, got None). Do not
  upgrade transformers past 4.x for this project.
- torch is the CPU-only build (torch==2.12.0+cpu). requirements.txt includes
  the PyTorch CPU index URL so a plain `pip install -r requirements.txt`
  resolves it correctly.


Setup (Windows / PowerShell)
----------------------------
1. Create the virtual environment (already created in .venv):
       py -m venv .venv

2. Install dependencies:
       .\.venv\Scripts\python.exe -m pip install -r requirements.txt

3. Download the Nougat model weights locally (CRUCIAL STEP).
   With the venv active (or using its huggingface-cli), download the model
   into a local .\nougat-small folder so --model can point at it:

       pip install huggingface_hub
       huggingface-cli download facebook/nougat-small --local-dir nougat-small

   (On newer huggingface_hub the command is `hf download facebook/nougat-small
   --local-dir nougat-small` -- both do the same thing.)

   --- If the download is rate-limited or times out ---
   You only need to do anything if it FAILS with a rate-limit error (HTTP 429)
   or times out. A free Hugging Face token raises the limit and speeds it up:

     a. Create a token at https://huggingface.co/settings/tokens
        (a "Read" token is enough).
     b. Authenticate, either by logging in:
            huggingface-cli login
        (paste the token when asked), OR set it for the session:
            set HF_TOKEN=hf_your_token_here
     c. Re-run the same download command -- it resumes where it left off:
            huggingface-cli download facebook/nougat-small --local-dir nougat-small

4. (Optional) Activate the environment:
       .\.venv\Scripts\Activate.ps1


Model weights
-------------
The Nougat model lives in .\nougat-small (downloaded in setup step 3). Point
--model at that folder so it is loaded locally and not re-downloaded from the
Hugging Face Hub.


Usage
-----
Run with the venv's Python (or activate first, then use `python`):

    .\.venv\Scripts\python.exe pdf_to_md.py input.pdf --model .\nougat-small
        -> writes input.md

    python pdf_to_md.py input.pdf --model .\nougat-small -o out.md   # custom output
    python pdf_to_md.py input.pdf --model .\nougat-small --pages 1-8 # subset of pages
    python pdf_to_md.py input.pdf --model .\nougat-small --dpi 200   # higher render DPI

Options:
    -o, --output        output .md path (default: <input>.md)
    --model             HF model id or LOCAL path (use .\nougat-small here;
                        facebook/nougat-base is higher quality but slower)
    --dpi               page render DPI (default 150; try 200 if blurry)
    --pages             subset, e.g. '1-8' or '2,5,9' (1-based)
    --max-new-tokens    decoder token cap per page (default 3584)

Speed (CPU): each page is decoded autoregressively -- roughly seconds to
~1-2 min per page. Test with --pages 1-2 before running a whole PDF.


Example
-------
    .\.venv\Scripts\python.exe pdf_to_md.py attention_is_all_you_need.pdf --model .\nougat-small --pages 1-2


===========================================================================
Additional setup notes / alternative workflows (verbatim, do not change cmds)
===========================================================================

Core Method
-----------
# Download the model locally, force offline mode so transformers/HF won't try
# the network, then run the converter against the local model folder.
# NOTE: --model path below is .\models\nougat-small; adjust it to wherever you
# downloaded the model (this project uses .\nougat-small).
pip install huggingface_hub
huggingface-cli download facebook/nougat-small --local-dir nougat-small
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python pdf_to_md.py attention_is_all_you_need.pdf --model .\models\nougat-small



Alternate method
----------------
# Uses the standalone Nougat CLI instead of this script. albumentations is
# pinned to 1.3.1 for compatibility with nougat-ocr. --full-precision helps CPU.
pip install "albumentations==1.3.1"
nougat attention_is_all_you_need.pdf -o . --full-precision

Step 1 - PDF -> .mmd (Nougat CLI):
pip install nougat-ocr
nougat attention_is_all_you_need.pdf -o . --full-precision

Step 2 - .mmd -> .md (the converter I just made):
python mmd_to_md.py attention_is_all_you_need.mmd

# Full two-step run:
nougat attention_is_all_you_need.pdf -o . --full-precision
python mmd_to_md.py attention_is_all_you_need.mmd
