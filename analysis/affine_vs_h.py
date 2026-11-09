import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sudoku_scan as ss

MODELS = ("A3", "A4", "H4")
STYLE = {"A3": ("#2a78d6", "o"), "A4": ("#eb6834", "s"), "H4": ("#1baf7a", "^")}
CELL = 50
SEARCH = 20
LOST_PX = 25.0
GOOD_PX = 3.0


def side_ratio(corners):
    c = np.asarray(corners, float)
    side = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    return float(max(side[0] / side[2], side[2] / side[0], side[1] / side[3], side[3] / side[1]))


def depth_spread(h4):
    g = np.linalg.inv(np.asarray(h4, float))
    w = np.array([g[2] @ [x, y, 1.0] for x, y in ss.TARGET])
    return float((w.max() - w.min()) / w.mean())


def ink_mask(rect_bgr):
    grey = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(grey, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)


def line_deviations(rect_bgr):
    ink = ink_mask(rect_bgr).astype(np.float32)
    devs = []
    for img in (ink, ink.T):
        for j in range(9):
            profile = img[:, CELL * j + 10: CELL * j + 40].mean(axis=1)
            for k in range(10):
                devs.append(_locate(profile, k))
    return np.array(devs)


def _locate(profile, k):
    n = len(profile)
    if k in (0, 9):
        idx = np.arange(0, SEARCH + 5) if k == 0 else np.arange(n - 1, n - SEARCH - 6, -1)
        hit = idx[profile[idx] > 0.5]
        return abs(float(hit[0] - idx[0])) if len(hit) else LOST_PX
    ideal = CELL * k - 0.5
    lo, hi = int(ideal - SEARCH), int(ideal + SEARCH) + 1
    window = profile[lo:hi]
    if window.max() < 0.5:
        return LOST_PX
    peak = lo + int(np.argmax(window))
    a, b = max(0, peak - 3), min(n, peak + 4)
    centre = float(np.sum(np.arange(a, b) * profile[a:b]) / np.sum(profile[a:b]))
    return abs(centre - ideal)


def cell_accuracy(rect_bgr, truth):
    try:
        grid = ss.read_grid(rect_bgr)
    except NotImplementedError:
        return None
    return float(np.mean(np.array(grid) == np.array(truth)))


def analyse(image_path, ann, use_gt):
    image = cv2.imread(str(image_path))
    if use_gt:
        corners = np.float32(ann["corners"])
    else:
        prepared, scale = ss.preprocess(image)
        cands, _ = ss.harris_candidates(prepared)
        work = ss.select_corners(cands, prepared)
        corners = ss.to_original(work, scale, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    rect = ss.rectify(image, corners)
    row = {
        "image": image_path.stem,
        "conditions": " ".join(ann.get("conditions", [])),
        "side_ratio": side_ratio(corners),
        "depth_spread": depth_spread(rect["H4"]),
        "a3_corner_err_px": rect["a3_corner_err_px"],
        "a4_rms_px": rect["a4_rms_px"],
    }
    for m in MODELS:
        d = line_deviations(rect[f"{m}_image"])
        row[f"{m}_line_dev_mean"] = float(d.mean())
        row[f"{m}_line_dev_max"] = float(d.max())
        row[f"{m}_lines_in_place"] = float(np.mean(d <= GOOD_PX))
        row[f"{m}_cell_acc"] = cell_accuracy(rect[f"{m}_image"], ann["grid"])
    return row, rect


def _scatter(ax, x, y, model, label=None):
    colour, marker = STYLE[model]
    ax.scatter(x, y, s=42, c=colour, marker=marker, edgecolors="white", linewidths=1.2,
               label=label or model, zorder=3)


def _axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e4e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_all(rows, best_rect, outdir):
    x = np.array([r["side_ratio"] for r in rows])
    plt.rcParams.update({"font.size": 9})

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    _scatter(ax, x, [r["a3_corner_err_px"] for r in rows], "A3", "A3 fourth-corner error")
    _scatter(ax, x, [r["a4_rms_px"] for r in rows], "A4", "A4 RMS residual")
    _axes(ax, "Obliqueness (max opposite-side ratio)", "Residual in 450 px target (px)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "fig_residuals.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in MODELS:
        _scatter(ax, x, [r[f"{m}_line_dev_mean"] for r in rows], m)
    ax.axhline(GOOD_PX, color="#8a8a85", linewidth=1, linestyle="--")
    ax.text(x.max(), GOOD_PX + 0.4, f"{GOOD_PX:.0f} px", color="#5f5f5a", fontsize=8, ha="right")
    _axes(ax, "Obliqueness (max opposite-side ratio)", "Mean grid-line deviation (px)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "fig_grid_deviation.png", dpi=300)
    plt.close(fig)

    if all(r["H4_cell_acc"] is not None for r in rows):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for m in MODELS:
            _scatter(ax, x, [100 * r[f"{m}_cell_acc"] for r in rows], m)
        _axes(ax, "Obliqueness (max opposite-side ratio)", "Cell accuracy (%)")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / "fig_cell_accuracy.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.scatter(x, [r["depth_spread"] for r in rows], s=42, c="#2a78d6", edgecolors="white",
               linewidths=1.2, zorder=3)
    _axes(ax, "Obliqueness (max opposite-side ratio)", "Relative depth spread  Δz / z")
    fig.tight_layout()
    fig.savefig(outdir / "fig_depth_spread.png", dpi=300)
    plt.close(fig)

    name, rect = best_rect
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, m in zip(axes, MODELS):
        ax.imshow(cv2.cvtColor(rect[f"{m}_image"], cv2.COLOR_BGR2RGB))
        for k in range(10):
            ax.axhline(CELL * k - 0.5, color=STYLE[m][0], linewidth=0.6, alpha=0.8)
            ax.axvline(CELL * k - 0.5, color=STYLE[m][0], linewidth=0.6, alpha=0.8)
        ax.set_title(m)
        ax.axis("off")
    fig.suptitle(f"{name}: rectifications with the ideal 50 px grid overlaid", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_side_by_side.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Stage 4: affine (A3, A4) versus homography (H4) comparison.")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("analysis/out"))
    p.add_argument("--gt-corners", action="store_true", help="use annotated instead of detected corners")
    p.add_argument("--figure-image", help="image stem for the side-by-side figure (default: most oblique)")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows, best = [], (None, None)
    for ann_path in sorted(args.annotations.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        image_path = args.images / ann["image"]
        try:
            row, rect = analyse(image_path, ann, args.gt_corners)
        except ss.PuzzleNotFound as exc:
            print(f"{image_path.name}: skipped ({exc})")
            continue
        rows.append(row)
        if args.figure_image:
            if row["image"] == args.figure_image:
                best = (row["image"], rect)
        elif best[0] is None or row["side_ratio"] > max(r["side_ratio"] for r in rows[:-1]):
            best = (row["image"], rect)

    rows.sort(key=lambda r: r["side_ratio"])
    with open(args.outdir / "stage4.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_all(rows, best, args.outdir)

    print(f"{'image':8} {'ratio':>6} {'dz/z':>6} {'A3err':>7} {'A4rms':>6} | mean line dev (px)  A3 / A4 / H4 | in place (%) A3 / A4 / H4")
    for r in rows:
        print(f"{r['image']:8} {r['side_ratio']:6.3f} {r['depth_spread']:6.3f} {r['a3_corner_err_px']:7.1f} "
              f"{r['a4_rms_px']:6.1f} | {r['A3_line_dev_mean']:6.2f} {r['A4_line_dev_mean']:6.2f} "
              f"{r['H4_line_dev_mean']:6.2f} | {100*r['A3_lines_in_place']:5.0f} {100*r['A4_lines_in_place']:5.0f} "
              f"{100*r['H4_lines_in_place']:5.0f}")
    print(f"\n{len(rows)} images -> {args.outdir}/stage4.csv and fig_*.png")


if __name__ == "__main__":
    main()
