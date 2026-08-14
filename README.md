# Indic Manuscript Synthesis Pipeline

An automated, end to end Python pipeline that synthesizes realistic historical
manuscript folios (page images) for three Indic scripts — **Devanagari**,
**Modi**, and **Sharada**  along with fully synchronized ground-truth
annotations. Built to produce training data for OCR models on historical
handwritten manuscripts.

## Overview

The pipeline generates synthetic manuscript page images from raw corpus text,
producing paired `.png` images and `.md` ground-truth annotation files. Each
generated folio includes:

- A realistic background (aged handmade paper, palm leaf, or vellum/parchment)
  with authentic texture, aging (stains, foxing speckles, warm tint), and
  physical deformation (page curl, folds, surface warping)
- Handwriting style calligraphic text rendering with natural waviness, slant,
  and per-word irregularity
- Traditional manuscript layout: ruled writing margins, marginal annotations,
  section/punctuation markers, and highlighted (red ink) text
- Ink artifacts: bleed, smudges, and occasional faded strokes
- A torn/deckled page edge

## Dataset

The generated dataset is hosted publicly on Hugging Face:

**[Shad0w1nonly/indic-manuscript-synthetic](https://huggingface.co/datasets/Shad0w1nonly/indic-manuscript-synthetic)**

It contains three subsets — `devanagari`, `modi`, `sharada`  each with:

| Split      | Images |
|------------|--------|
| train      | 85     |
| validation | 10     |
| test       | 5      |

Each image (`Image_N.png`) has a matching ground truth annotation file
(`Image_N.md`) containing the exact rendered text, including any marginal
note.

## Repository Structure

```
manuscript-generator/
├── generate.py           # Main pipeline — run this to generate the dataset
├── hb_render.py           # HarfBuzz + FreeType word shaping/rendering helper
├── fonts/                 # Font files used for each script
│   ├── Kalam-Regular.ttf          (Devanagari — genuine cursive handwriting font)
│   ├── MarathiCursiveT.ttf        (Modi — genuine cursive Modi font)
│   └── NotoSansModi-Regular.ttf / NotoSansSharada-Regular.ttf
├── data/                  # Source corpus text files (one per script)
│   ├── devanagari_md.md
│   ├── Modi_md.md
│   └── sharada_md.md
├── dataset_output/        # Generated output (gitignored - see Hugging Face link above)
│   └── <script>/<split>/Image_N.png + Image_N.md
└── requirements.txt
```

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/umateabhijeet12-eng/manuscript-generator.git
cd manuscript-generator
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

**Fonts:** Devanagari (Kalam) and Modi (MarathiCursiveT) fonts are genuine
open source handwriting-style fonts and should be placed in `fonts/`. See
comments in `generate.py`'s `SCRIPTS` config for exact filenames and sources.

## Usage

Run the full pipeline:

```bash
python generate.py
```

This generates 100 images per script (85/10/5 train/validation/test split),
saved under `dataset_output/<script>/<split>/`.

### Configurable Parameters

Key parameters live at the top of `generate.py` and in the `SCRIPTS` dict:

- `WIDTH`, `HEIGHT` — output image dimensions
- `SCRIPTS` — add/remove scripts here; each entry needs a `font` path, a
  `corpus` text file, a `font_size`, and a `needs_distortion` flag (see
  Limitations below)
- `aging_intensity` (randomized per image in `generate_dataset`) — controls
  background wear/staining strength
- Background material weights (`paper` / `palmleaf` / `vellum`) in
  `get_master_background`

To add a new script, add an entry to `SCRIPTS` with its font and corpus file
 no other code changes are required.

## Limitations

- **Sharada calligraphy:** no open-source cursive/handwriting-style Sharada
  font currently exists (Sharada is a rare, largely liturgical script with
  minimal digital typography investment). As a fallback, Sharada glyphs are
  rendered with a print font and post processed with a stroke-distortion
  filter (`apply_stroke_distortion`) — mild elastic warp and stroke-thickness
  jitter  to reduce mechanical uniformity. This does not fully replicate
  genuine handwriting and is a documented, deliberate trade-off given no
  better font resource is currently available.
- The scanned real-paper background path is cached after first generation
  (`data/bg_paper.png`) for performance; delete this file to force
  regeneration from the source sample image.

## License

Fonts used retain their original licenses (SIL OFL for Kalam; M+ Fonts
license for MarathiCursiveT). Generated dataset content is derived from the
source text corpora provided in `data/`.
