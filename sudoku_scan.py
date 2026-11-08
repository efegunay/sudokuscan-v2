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

# Stage 1 parameters. Sizes are fractions of the working image width, so the same values
# work for a 1000 px and a 4000 px photograph.
WORK_LONG_SIDE = 1200      # px; larger inputs are downscaled to this before corner detection
BLUR_FRAC = 0.003          # Gaussian denoising kernel  (~3.6 px at 1200 px)
BACKGROUND_FRAC = 0.02     # closing kernel for the paper background (~24 px), wider than any ink stroke

# Stage 2 parameters (working image is WORK_LONG_SIDE px, so pixel sizes are already relative).
HARRIS_BLOCK_SIZE = 5      # neighbourhood summed into the structure tensor M
HARRIS_KSIZE = 3           # Sobel aperture for the derivatives Ix, Iy
HARRIS_K = 0.04            # R = det(M) - k * trace(M)^2
HARRIS_THRESHOLD = 0.01    # keep pixels with R > HARRIS_THRESHOLD * max(R)
NMS_FRAC = 0.008           # non-maximum suppression window (~9 px), smaller than a grid cell
INK_LEVEL = 220          # prepared-image grey level below which a pixel counts as ink (paper is ~250)
MIN_PUZZLE_AREA = 0.02     # the puzzle covers at least 2% of the frame
SNAP_RADIUS_CELLS = 0.25   # a Harris corner must lie within 1/4 cell of the region corner


# MARK: - Exceptions

class PuzzleNotFound(Exception):
    """Raised by any stage that cannot continue; the image gets verdict 'not_found'."""


# MARK: - Stage 1: Preprocessing

def odd_kernel(fraction, width, minimum=3):
    """Odd kernel size equal to `fraction` of `width` (OpenCV blur kernels must be odd)."""
    size = max(minimum, int(round(fraction * width)))
    return size if size % 2 == 1 else size + 1


def preprocess(image_bgr):
    """Greyscale, denoise, normalise illumination. Returns (prepared_image, scale).

    `prepared` is a uint8 greyscale image of the downscaled photo where the paper is
    close to white (255) everywhere, even under a shadow, and ink stays dark.
    `scale` = working size / original size; divide working coordinates by it to get
    original-image coordinates.
    """
    # 1) Downscale: Harris and the later stages do not need 12 megapixels, and a fixed
    #    working size keeps the runtime far below the 20 s limit.
    h, w = image_bgr.shape[:2]
    scale = min(1.0, WORK_LONG_SIDE / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 2) Greyscale: corners and digits are defined by brightness, not colour.
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    width = grey.shape[1]

    # 3) Denoise: a small Gaussian removes sensor noise and JPEG blocks that Harris
    #    would otherwise report as tiny corners.
    k = odd_kernel(BLUR_FRAC, width)
    grey = cv2.GaussianBlur(grey, (k, k), 0)

    # 4) Background estimate: a morphological closing (dilate, then erode) with a kernel
    #    wider than any ink stroke erases the dark ink and keeps the paper brightness,
    #    including shadows and gradients. Blurring smooths the blocky closing result.
    kb = odd_kernel(BACKGROUND_FRAC, width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    background = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (kb, kb), 0)

    # 5) Flat-field: divide by the background. Paper becomes ~1.0 everywhere, ink stays
    #    a fraction of it, so a shadow no longer changes how dark the ink looks.
    flat = grey.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    prepared = np.clip(flat * 255.0, 0, 255).astype(np.uint8)
    return prepared, scale


# MARK: - Stage 2: Harris corners + corner selection
# NOTE: Stage 2 was written by Claude (AI) at the author's explicit request, against the
# Section 11 rule. It must be declared in Appendix C.

def harris_candidates(prepared):
    """Harris response -> candidate points. Returns (points, response).

    points: (N, 2) float32 array of (x, y) in working-image pixels, one per local maximum.
    response: the raw Harris response map R (float32, same size as `prepared`).
    """
    response = cv2.cornerHarris(prepared.astype(np.float32), HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)

    # Non-maximum suppression: dilation replaces every pixel by the maximum of its window,
    # so a pixel equal to its dilated value is the peak of its neighbourhood.
    n = odd_kernel(NMS_FRAC, prepared.shape[1])
    local_max = cv2.dilate(response, np.ones((n, n), np.uint8))
    peaks = (response == local_max) & (response > HARRIS_THRESHOLD * response.max())

    ys, xs = np.nonzero(peaks)              # NumPy gives (row, col) = (y, x)
    points = np.column_stack([xs, ys]).astype(np.float32)
    return points, response


def order_corners(quad):
    """Order four points as TL, TR, BR, BL for a roughly upright puzzle (in-plane rotation < 45 deg)."""
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    centre = quad.mean(axis=0)
    # Image y points down, so the clockwise order in the picture is increasing atan2 angle.
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    order = np.argsort(angles)
    quad, angles = quad[order], angles[order]
    # TL is the point whose direction from the centre is closest to up-left (-135 deg).
    diff = np.angle(np.exp(1j * (angles + 3 * np.pi / 4)))   # wrapped to (-pi, pi]
    return np.roll(quad, -int(np.argmin(np.abs(diff))), axis=0)


def grid_score(binary, quad):
    """How much the inside of `quad` looks like a 9x9 grid, in [0, 1].

    The region is warped to a 450x450 square (only to test it; the final corners come from
    Harris). Ink fraction is measured on the 10 expected line positions every 50 px versus
    the cell interiors between them: a sudoku has dark lines exactly there.
    """
    target = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
    warp = cv2.warpPerspective(binary, cv2.getPerspectiveTransform(quad, target), (451, 451))
    ink = warp.astype(np.float32) / 255.0
    rows, cols = ink.mean(axis=1), ink.mean(axis=0)
    lines = np.arange(0, 451, 50)
    on = [p[max(0, i - 3):i + 4].max() for p in (rows, cols) for i in lines]
    off = [p[i + 15:i + 36].mean() for p in (rows, cols) for i in lines[:-1]]
    return float(np.clip(np.median(on) - 2 * np.mean(off), 0.0, 1.0))


def puzzle_region(prepared):
    """Find the puzzle outline as an approximate quadrilateral (working-image pixels), or None.

    Contours are used only to locate the region; the corner coordinates come from Harris.
    """
    binary = (prepared < INK_LEVEL).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_PUZZLE_AREA * prepared.shape[0] * prepared.shape[1]
    best, best_score = None, 0.0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        hull = cv2.convexHull(contour)
        quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(quad) != 4:
            continue
        quad = order_corners(quad)
        score = grid_score(binary, quad)
        if score > best_score:
            best, best_score = quad, score
    return best


def select_corners(candidates, prepared):
    """Pick the four outer puzzle corners from the Harris candidates: TL, TR, BR, BL.

    1) Locate the puzzle region (largest grid-like quadrilateral).
    2) For each region corner, take the nearest Harris candidate within 1/4 cell.
    Returns a (4, 2) float32 array in working-image pixels.
    """
    region = puzzle_region(prepared)
    if region is None:
        raise PuzzleNotFound("no grid-like quadrilateral")
    side = np.mean(np.linalg.norm(region - np.roll(region, -1, axis=0), axis=1))
    radius = SNAP_RADIUS_CELLS * side / 9
    corners = []
    for approx in region:
        dist = np.linalg.norm(candidates - approx, axis=1)
        if len(dist) == 0 or dist.min() > radius:
            raise PuzzleNotFound("no Harris corner near a region corner")
        corners.append(candidates[np.argmin(dist)])
    return np.array(corners, np.float32)


def to_original(corners, scale, grey_full):
    """Working-image corners -> original-image corners, refined with cornerSubPix."""
    # Pixel centres: x_orig + 0.5 = (x_work + 0.5) / scale
    pts = ((corners + 0.5) / scale - 0.5).astype(np.float32).reshape(-1, 1, 2)
    win = max(2, int(round(1.5 / scale)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    cv2.cornerSubPix(grey_full, pts, (win, win), (-1, -1), criteria)
    return pts.reshape(4, 2)


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

def run_pipeline(image_bgr, debug):
    """Runs all stages on one image and returns the fields of the result JSON.

    Intermediate results go into `debug` as soon as they exist, so the caller can still
    write the debug images when a later stage fails.
    """
    prepared, scale = preprocess(image_bgr)
    candidates, response = harris_candidates(prepared)
    debug.update(response=response, candidates=candidates)
    work_corners = select_corners(candidates, prepared)
    debug["selected"] = work_corners
    grey_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners = to_original(work_corners, scale, grey_full)
    debug["corners"] = corners.tolist()
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


def write_harris_image(outdir, stem, debug):
    """<name>_harris.png: Harris response map with all candidates (green) and the four
    selected corners (red, labelled TL TR BR BL), in working-image pixels."""
    if "response" not in debug:
        return
    r = np.maximum(debug["response"], 0)
    r = np.sqrt(r / (r.max() + 1e-12))           # square root so weak corners stay visible
    vis = cv2.cvtColor((r * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in debug["candidates"]:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 200, 0), 1)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), debug.get("selected", [])):
        cv2.circle(vis, (int(x), int(y)), 9, (0, 0, 255), 2)
        cv2.putText(vis, label, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(outdir / f"{stem}_harris.png"), vis)


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
    debug = {}
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise PuzzleNotFound(f"cannot read image: {path}")
        result.update(run_pipeline(image, debug))
    except Exception as exc:
        result = empty_result(path.name)
        result["corners"] = debug.get("corners")   # keep the corners if only a later stage failed
        print(f"[{path.name}] not_found: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not isinstance(exc, (PuzzleNotFound, NotImplementedError)):
            traceback.print_exc()
    result["runtime_s"] = round(time.perf_counter() - start, 3)
    write_result(outdir, path.stem, result)
    write_harris_image(outdir, path.stem, debug)
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
