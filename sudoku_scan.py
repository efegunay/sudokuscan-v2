"""SudokuScan: photograph of a printed sudoku -> 9x9 digital grid.

Usage:
    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# MARK: - Exceptions

class PuzzleNotFound(Exception):
    """Raised by any stage that cannot continue; the image gets verdict 'not_found'."""


# MARK: - Stage 1: Preprocessing

def preprocess(image_bgr):
    """Greyscale, denoise, normalise illumination. Returns (prepared_image, scale)."""
    raise NotImplementedError("Stage 1")


# MARK: - Stage 2: Harris corners + corner selection (written by Efe)

def harris_candidates(prepared):
    """Harris response map -> list of candidate corner points."""
    raise NotImplementedError("Stage 2a")


def select_corners(candidates, prepared):
    """Pick the four outer puzzle corners: TL, TR, BR, BL."""
    raise NotImplementedError("Stage 2b")


# MARK: - Stage 3: Rectification (A3, A4, H4)

def rectify(image_bgr, corners):
    """Returns the three 450x450 rectifications, their matrices and residuals."""
    raise NotImplementedError("Stage 3")


# MARK: - Stage 5: Cells + OCR

def read_grid(rectified_h4):
    """Returns the 9x9 grid of givens, 0 for empty cells."""
    raise NotImplementedError("Stage 5")


# MARK: - Stage 6: Validity (written by Efe)

def check_validity(grid):
    """Returns (verdict, conflicting_cells)."""
    raise NotImplementedError("Stage 6")


# MARK: - Pipeline

def run_pipeline(image_bgr):
    """Runs all stages on one image and returns the fields of the result JSON."""
    prepared, scale = preprocess(image_bgr)
    candidates = harris_candidates(prepared)
    corners = select_corners(candidates, prepared)
    rect = rectify(image_bgr, corners)
    grid = read_grid(rect["H4_image"])
    verdict, conflicts = check_validity(grid)
    # Output images and the remaining JSON fields are filled in as the stages get written.
    return {"corners": corners, "grid": grid, "verdict": verdict}


# MARK: - Output

def empty_result(name):
    """Result in the Appendix A schema with every field unset and verdict 'not_found'."""
    return {
        "image": name,
        "corners": None,
        "A3": None,
        "A4": None,
        "H4": None,
        "a3_corner_err_px": None,
        "a4_rms_px": None,
        "grid": None,
        "verdict": "not_found",
        "runtime_s": None,
    }


def write_result(outdir, stem, result):
    with open(outdir / f"{stem}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    if result["grid"] is not None:
        with open(outdir / f"{stem}_grid.txt", "w") as f:
            f.write("\n".join("".join(str(d) for d in row) for row in result["grid"]) + "\n")


# MARK: - Single image (never crashes)

def process_image(path, outdir):
    """Process one image; any failure becomes verdict 'not_found' instead of a crash."""
    start = time.perf_counter()
    result = empty_result(path.name)
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise PuzzleNotFound(f"cannot read image: {path}")
        result.update(run_pipeline(image))
    except Exception as exc:
        result = empty_result(path.name)
        print(f"[{path.name}] not_found: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not isinstance(exc, (PuzzleNotFound, NotImplementedError)):
            traceback.print_exc()
    result["runtime_s"] = round(time.perf_counter() - start, 3)
    write_result(outdir, path.stem, result)
    print(f"[{path.name}] {result['verdict']} in {result['runtime_s']:.2f}s")
    return result


# MARK: - CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Read a sudoku puzzle from a photograph.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="a single image file")
    source.add_argument("--batch", type=Path, help="a directory of images (grader mode)")
    parser.add_argument("--outdir", type=Path, required=True, help="directory for all output files")
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.input:
        paths = [args.input]
    else:
        paths = sorted(p for p in args.batch.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for path in paths:
        process_image(path, args.outdir)


if __name__ == "__main__":
    main()
