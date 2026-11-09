import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pytesseract

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

WORK_LONG_SIDE = 1200
BLUR_FRAC = 0.003
BACKGROUND_FRAC = 0.02

HARRIS_BLOCK_SIZE = 5
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.01
NMS_FRAC = 0.008
INK_LEVEL = 220
MIN_PUZZLE_AREA = 0.02
SNAP_RADIUS_CELLS = 0.25


class PuzzleNotFound(Exception):
    pass


def odd_kernel(fraction, width, minimum=3):
    size = max(minimum, int(round(fraction * width)))
    return size if size % 2 == 1 else size + 1


def preprocess(image_bgr):
    h, w = image_bgr.shape[:2]
    scale = min(1.0, WORK_LONG_SIDE / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    width = grey.shape[1]

    k = odd_kernel(BLUR_FRAC, width)
    grey = cv2.GaussianBlur(grey, (k, k), 0)

    kb = odd_kernel(BACKGROUND_FRAC, width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    background = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (kb, kb), 0)

    flat = grey.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    prepared = np.clip(flat * 255.0, 0, 255).astype(np.uint8)
    return prepared, scale


def harris_candidates(prepared):
    response = cv2.cornerHarris(prepared.astype(np.float32), HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)

    n = odd_kernel(NMS_FRAC, prepared.shape[1])
    local_max = cv2.dilate(response, np.ones((n, n), np.uint8))
    peaks = (response == local_max) & (response > HARRIS_THRESHOLD * response.max())

    ys, xs = np.nonzero(peaks)
    points = np.column_stack([xs, ys]).astype(np.float32)
    return points, response


def order_corners(quad):
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    order = np.argsort(angles)
    quad, angles = quad[order], angles[order]
    diff = np.angle(np.exp(1j * (angles + 3 * np.pi / 4)))
    return np.roll(quad, -int(np.argmin(np.abs(diff))), axis=0)


def grid_score(binary, quad):
    target = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
    warp = cv2.warpPerspective(binary, cv2.getPerspectiveTransform(quad, target), (451, 451))
    ink = warp.astype(np.float32) / 255.0
    rows, cols = ink.mean(axis=1), ink.mean(axis=0)
    lines = np.arange(0, 451, 50)
    on = [p[max(0, i - 3):i + 4].max() for p in (rows, cols) for i in lines]
    off = [p[i + 15:i + 36].mean() for p in (rows, cols) for i in lines[:-1]]
    return float(np.clip(np.median(on) - 2 * np.mean(off), 0.0, 1.0))


def puzzle_region(prepared):
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
    pts = ((corners + 0.5) / scale - 0.5).astype(np.float32).reshape(-1, 1, 2)
    win = max(2, int(round(1.5 / scale)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    cv2.cornerSubPix(grey_full, pts, (win, win), (-1, -1), criteria)
    return pts.reshape(4, 2)


RECT_SIZE = 450

TARGET = np.float32([[-0.5, -0.5], [RECT_SIZE - 0.5, -0.5],
                     [RECT_SIZE - 0.5, RECT_SIZE - 0.5], [-0.5, RECT_SIZE - 0.5]])


def apply_affine(matrix_2x3, points):
    return points @ matrix_2x3[:, :2].T + matrix_2x3[:, 2]


def rectify(image_bgr, corners):
    src = np.float32(corners)
    size = (RECT_SIZE, RECT_SIZE)

    a3 = cv2.getAffineTransform(src[[0, 1, 3]], TARGET[[0, 1, 3]])
    a3_br = apply_affine(a3, src[2:3])[0]
    a3_err = float(np.linalg.norm(a3_br - TARGET[2]))

    a4, _ = cv2.estimateAffine2D(src, TARGET, method=cv2.RANSAC, ransacReprojThreshold=1e6,
                                 refineIters=100)
    if a4 is None:
        raise PuzzleNotFound("A4 estimation failed (degenerate corners)")
    a4_res = np.linalg.norm(apply_affine(a4, src) - TARGET, axis=1)
    a4_rms = float(np.sqrt(np.mean(a4_res ** 2)))

    h4 = cv2.getPerspectiveTransform(src, TARGET)
    h4_proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h4).reshape(4, 2)
    h4_max_err = float(np.abs(h4_proj - TARGET).max())

    return {
        "A3_image": cv2.warpAffine(image_bgr, a3, size, flags=cv2.INTER_LINEAR),
        "A4_image": cv2.warpAffine(image_bgr, a4, size, flags=cv2.INTER_LINEAR),
        "H4_image": cv2.warpPerspective(image_bgr, h4, size, flags=cv2.INTER_LINEAR),
        "A3": a3.tolist(),
        "A4": a4.tolist(),
        "H4": h4.tolist(),
        "a3_corner_err_px": round(a3_err, 3),
        "a3_br_target": [round(float(v), 3) for v in a3_br],
        "a4_rms_px": round(a4_rms, 3),
        "h4_max_corner_err_px": h4_max_err,
    }


CELL = RECT_SIZE // 9
CELL_MARGIN = 7
RECT_BACKGROUND_KERNEL = 15
RECT_INK_LEVEL = 0.75
EMPTY_MIN_AREA = 0.03
EMPTY_MIN_HEIGHT = 0.35
OCR_HEIGHT = 48
OCR_PAD = 20
OCR_CONFIG = "--psm 10 --oem 1 -c tessedit_char_whitelist=123456789"
OCR_FALLBACK_CONFIG = "--psm 13 --oem 1 -c tessedit_char_whitelist=123456789"


def rect_ink(rect_bgr):
    grey = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RECT_BACKGROUND_KERNEL, RECT_BACKGROUND_KERNEL))
    background = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (RECT_BACKGROUND_KERNEL, RECT_BACKGROUND_KERNEL), 0)
    flat = np.clip(grey.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0), 0.0, 1.0)
    return grey, flat, (flat < RECT_INK_LEVEL).astype(np.uint8)


def cell_box(row, col, margin=CELL_MARGIN):
    y0, x0 = row * CELL, col * CELL
    return y0 + margin, y0 + CELL - margin, x0 + margin, x0 + CELL - margin


def digit_mask(ink_crop):
    h, w = ink_crop.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_crop, connectivity=8)
    best, best_area = 0, 0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x == 0 or y == 0 or x + cw == w or y + ch == h
        line_like = cw <= 0.15 * w or ch <= 0.15 * h
        if touches and line_like:
            continue
        if area > best_area:
            best, best_area = i, area
    if best == 0:
        return None
    x, y, cw, ch, area = stats[best]
    if area < EMPTY_MIN_AREA * h * w or ch < EMPTY_MIN_HEIGHT * h:
        return None
    mask = labels == best
    for i in range(1, n):
        if i == best:
            continue
        xi, yi, wi, hi, _ = stats[i]
        cx, cy = xi + wi / 2, yi + hi / 2
        if x <= cx <= x + cw and y - 0.2 * ch <= cy <= y + 1.2 * ch:
            mask |= labels == i
    return mask


def prepare_for_ocr(flat_crop, mask):
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - 1), ys.max() + 2
    x0, x1 = max(0, xs.min() - 1), xs.max() + 2
    keep = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    digit = np.where(keep, flat_crop, 1.0)[y0:y1, x0:x1]
    width = max(1, int(round(digit.shape[1] * OCR_HEIGHT / digit.shape[0])))
    digit = cv2.resize(digit, (width, OCR_HEIGHT), interpolation=cv2.INTER_CUBIC)
    digit = (np.clip((digit - 0.25) / 0.6, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.copyMakeBorder(digit, OCR_PAD, OCR_PAD, OCR_PAD, OCR_PAD, cv2.BORDER_CONSTANT, value=255)


def ocr_digit(image):
    for config in (OCR_CONFIG, OCR_FALLBACK_CONFIG):
        text = pytesseract.image_to_string(image, config=config)
        digits = [ch for ch in text if ch in "123456789"]
        if digits:
            return int(digits[0])
    return 0


def read_cells(rect_bgr, prepare=True):
    grey, flat, ink = rect_ink(rect_bgr)
    grid, crops = [], {}
    for r in range(9):
        row = []
        for c in range(9):
            y0, y1, x0, x1 = cell_box(r, c)
            mask = digit_mask(ink[y0:y1, x0:x1])
            if mask is None:
                row.append(0)
                continue
            crop = prepare_for_ocr(flat[y0:y1, x0:x1], mask) if prepare else grey[y0:y1, x0:x1]
            crops[(r, c)] = crop
            row.append(ocr_digit(crop))
        grid.append(row)
    return grid, crops


def read_grid(rectified_h4):
    return read_cells(rectified_h4)[0]


def sudoku_units():
    rows = [[(r, c) for c in range(9)] for r in range(9)]
    cols = [[(r, c) for r in range(9)] for c in range(9)]
    boxes = [[(br * 3 + i, bc * 3 + j) for i in range(3) for j in range(3)]
             for br in range(3) for bc in range(3)]
    return rows + cols + boxes


def check_validity(grid):
    conflicts = set()
    for unit in sudoku_units():
        seen = {}
        for r, c in unit:
            digit = grid[r][c]
            if digit == 0:
                continue
            seen.setdefault(digit, []).append((r, c))
        for cells in seen.values():
            if len(cells) > 1:
                conflicts.update(cells)
    verdict = "invalid" if conflicts else "valid"
    return verdict, sorted(conflicts)


def run_pipeline(image_bgr, debug):
    prepared, scale = preprocess(image_bgr)
    candidates, response = harris_candidates(prepared)
    debug.update(response=response, candidates=candidates)
    work_corners = select_corners(candidates, prepared)
    debug["selected"] = work_corners
    grey_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners = to_original(work_corners, scale, grey_full)
    fields = debug.setdefault("fields", {})
    fields["corners"] = [[round(float(x), 2), round(float(y), 2)] for x, y in corners]

    rect = rectify(image_bgr, corners)
    debug["rect"] = rect
    fields.update({key: rect[key] for key in ("A3", "A4", "H4", "a3_corner_err_px", "a4_rms_px")})

    grid = read_grid(rect["H4_image"])
    fields["grid"] = grid
    verdict, conflicts = check_validity(grid)
    debug.update(grid=grid, verdict=verdict, conflicts=conflicts)
    return {**fields, "grid": grid, "verdict": verdict}


def empty_result(name):
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
    if "response" not in debug:
        return
    r = np.maximum(debug["response"], 0)
    r = np.sqrt(r / (r.max() + 1e-12))
    vis = cv2.cvtColor((r * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in debug["candidates"]:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 200, 0), 1)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), debug.get("selected", [])):
        cv2.circle(vis, (int(x), int(y)), 9, (0, 0, 255), 2)
        cv2.putText(vis, label, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(outdir / f"{stem}_harris.png"), vis)


def write_rect_images(outdir, stem, debug):
    if "rect" not in debug:
        return
    for model in ("A3", "A4", "H4"):
        cv2.imwrite(str(outdir / f"{stem}_rect_{model}.png"), debug["rect"][f"{model}_image"])


DEBUG_SCALE = 2


def write_debug_image(outdir, stem, debug):
    if "rect" not in debug or "grid" not in debug:
        return
    s = DEBUG_SCALE
    size = RECT_SIZE * s
    vis = cv2.resize(debug["rect"]["H4_image"], (size, size), interpolation=cv2.INTER_LINEAR)
    vis = cv2.addWeighted(vis, 0.55, np.full_like(vis, 255), 0.45, 0)
    conflicts = set(debug.get("conflicts", []))
    overlay = vis.copy()
    for r, c in conflicts:
        cv2.rectangle(overlay, (c * CELL * s, r * CELL * s), ((c + 1) * CELL * s, (r + 1) * CELL * s), (60, 60, 230), -1)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    for k in range(10):
        p = k * CELL * s
        thick = 3 if k % 3 == 0 else 1
        cv2.line(vis, (p, 0), (p, size), (200, 120, 40), thick)
        cv2.line(vis, (0, p), (size, p), (200, 120, 40), thick)
    for r in range(9):
        for c in range(9):
            digit = debug["grid"][r][c]
            if digit == 0:
                continue
            colour = (0, 0, 200) if (r, c) in conflicts else (150, 60, 0)
            org = (c * CELL * s + 8, r * CELL * s + 30)
            cv2.putText(vis, str(digit), org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    banner_h = 44
    verdict = debug.get("verdict", "")
    text = f"{verdict.upper()}"
    if conflicts:
        text += "  conflicts: " + " ".join(f"r{r + 1}c{c + 1}" for r, c in sorted(conflicts))
    banner = np.full((banner_h, size, 3), 255, np.uint8)
    colour = (0, 140, 0) if verdict == "valid" else (0, 0, 200)
    cv2.putText(banner, text[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    cv2.imwrite(str(outdir / f"{stem}_debug.png"), np.vstack([banner, vis]))


def write_result(outdir, stem, result):
    with open(outdir / f"{stem}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    if result["grid"] is not None:
        with open(outdir / f"{stem}_grid.txt", "w") as f:
            f.write("\n".join("".join(str(d) for d in row) for row in result["grid"]) + "\n")


def process_image(path, outdir):
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
        result.update(debug.get("fields", {}))
        print(f"[{path.name}] not_found: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not isinstance(exc, (PuzzleNotFound, NotImplementedError)):
            traceback.print_exc()
    result["runtime_s"] = round(time.perf_counter() - start, 3)
    write_result(outdir, path.stem, result)
    write_harris_image(outdir, path.stem, debug)
    write_rect_images(outdir, path.stem, debug)
    write_debug_image(outdir, path.stem, debug)
    print(f"[{path.name}] {result['verdict']} in {result['runtime_s']:.2f}s")
    return result


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
