# SudokuScan

COMP 482 Computer Vision — Project 1. Reads a phone photograph of a printed sudoku and returns
the puzzle as a 9×9 grid: Harris corner detection, affine and projective (homography)
rectification, cell extraction and Tesseract OCR, then a sudoku validity check.

## Installation

Requires Python 3.10+ and the Tesseract OCR engine (pytesseract is only a Python wrapper).

1. Install Tesseract:

   | OS | Command |
   |---|---|
   | macOS | `brew install tesseract` |
   | Ubuntu / Debian | `sudo apt-get install tesseract-ocr` |
   | Windows | installer from https://github.com/UB-Mannheim/tesseract/wiki, then add it to `PATH` |

   Check with `tesseract --version` (developed with 5.5.4).

2. Create a virtual environment and install the Python packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

```bash
python sudoku_scan.py --input photo.jpg --outdir out/
python sudoku_scan.py --batch images_dir/ --outdir results/
```

`--batch` processes every `.jpg`, `.jpeg` and `.png` file in the directory. Runs on CPU only,
about 2 s per image. An image that fails never stops the run: it gets a result file with
`"verdict": "not_found"`.

### Output files (per input image `<name>`)

| File | Content |
|---|---|
| `<name>_grid.txt` | 9 lines of 9 digits, `0` for empty cells (from the H4 rectification) |
| `<name>_result.json` | corners (TL, TR, BR, BL, original-image pixels), `A3`, `A4`, `H4` matrices, `a3_corner_err_px`, `a4_rms_px`, `grid`, `verdict` (`valid` / `invalid` / `not_found`), `runtime_s` |
| `<name>_harris.png` | Harris response map with all candidates (green) and the four selected corners (red) |
| `<name>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | the three 450×450 rectifications |
| `<name>_debug.png` | H4 rectification with cell boundaries, recognised digits and conflicting cells in red |

## Pipeline

| Stage | Function(s) in `sudoku_scan.py` | What it does |
|---|---|---|
| 1 Preprocessing | `preprocess` | downscale to 1200 px, greyscale, denoise, divide by a closing-based background estimate to remove shadows |
| 2 Corners | `harris_candidates`, `select_corners`, `to_original` | `cv2.cornerHarris` + threshold + non-maximum suppression; region = most grid-like quadrilateral, each corner = nearest Harris candidate; `cornerSubPix` refinement |
| 3 Rectification | `rectify` | A3 (`getAffineTransform`, TL/TR/BL), A4 (`estimateAffine2D`, least squares), H4 (`getPerspectiveTransform`) to 450×450 |
| 5 Recognition | `read_cells`, `digit_mask`, `prepare_for_ocr`, `ocr_digit` | 81 cells with a 7 px margin, empty/filled decision from connected components, cell preparation, Tesseract `--psm 10` with digit whitelist |
| 6 Validity | `check_validity` | duplicate digits in any row, column or 3×3 box |

## Evaluation

Section 7 metrics (corner error, cell accuracy, empty-cell F1, grid exact match, robustness),
overall and per condition C1–C5:

```bash
python sudoku_scan.py --batch path/to/images --outdir results/
python tools/evaluate.py --results results/ --annotations path/to/annotations --csv results/eval.csv
```

Development set (20 images, detected corners):

| Corner error | Cell accuracy | Empty-cell F1 | Grid exact match | Robustness |
|---|---|---|---|---|
| 0.50 % | 99.81 % | 0.998 | 90 % | 100 % |

## Affine versus homography analysis (Stage 4)

```bash
python analysis/affine_vs_h.py --images path/to/images --annotations path/to/annotations --outdir analysis/out
```

Writes `stage4.csv` and the figures used in the report (`fig_residuals.png`,
`fig_grid_deviation.png`, `fig_cell_accuracy.png`, `fig_depth_spread.png`,
`fig_side_by_side.png`). Add `--gt-corners` to rectify from the annotated corners.
The written analysis is in `docs/stage4_analysis.md`.

## Repository layout

```
sudoku_scan.py            the pipeline and the command-line interface
tools/evaluate.py         Section 7 metrics
analysis/affine_vs_h.py   Stage 4 measurements and figures
docs/                     report material
requirements.txt          pinned Python dependencies
```

The course development set is not part of the repository.
