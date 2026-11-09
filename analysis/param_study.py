import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sudoku_scan as ss

DEFAULTS = {
    "HARRIS_BLOCK_SIZE": ss.HARRIS_BLOCK_SIZE,
    "HARRIS_KSIZE": ss.HARRIS_KSIZE,
    "HARRIS_K": ss.HARRIS_K,
    "HARRIS_THRESHOLD": ss.HARRIS_THRESHOLD,
}
SWEEPS = {
    "HARRIS_BLOCK_SIZE": [2, 3, 5, 7, 9, 13],
    "HARRIS_KSIZE": [1, 3, 5, 7],
    "HARRIS_K": [0.02, 0.04, 0.06, 0.08, 0.12, 0.16],
    "HARRIS_THRESHOLD": [0.001, 0.005, 0.01, 0.03, 0.1, 0.3],
}
LABELS = {
    "HARRIS_BLOCK_SIZE": "blockSize",
    "HARRIS_KSIZE": "ksize",
    "HARRIS_K": "k",
    "HARRIS_THRESHOLD": "threshold / max R",
}
COLOUR = "#2a78d6"


def load(images_dir, annotations_dir):
    items = []
    for ann_path in sorted(annotations_dir.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        image = cv2.imread(str(images_dir / ann["image"]))
        prepared, scale = ss.preprocess(image)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        items.append({"name": ann_path.stem, "prepared": prepared, "scale": scale, "grey": grey,
                      "truth": np.array(ann["corners"], float)})
    return items


def corner_error_pct(pred, truth):
    side = np.mean(np.linalg.norm(truth - np.roll(truth, -1, axis=0), axis=1))
    return float(100 * np.linalg.norm(pred - truth, axis=1).mean() / side)


def run_config(items, params):
    for name, value in DEFAULTS.items():
        setattr(ss, name, params.get(name, value))
    counts, errors, found, seconds = [], [], 0, []
    for it in items:
        start = time.perf_counter()
        try:
            cands, _ = ss.harris_candidates(it["prepared"])
        except cv2.error:
            counts.append(np.nan)
            errors.append(np.nan)
            continue
        counts.append(len(cands))
        try:
            work = ss.select_corners(cands, it["prepared"])
            corners = ss.to_original(work, it["scale"], it["grey"])
            errors.append(corner_error_pct(corners, it["truth"]))
            found += 1
        except ss.PuzzleNotFound:
            errors.append(np.nan)
        seconds.append(time.perf_counter() - start)
    for name, value in DEFAULTS.items():
        setattr(ss, name, value)
    errs = np.array(errors, float)
    return {
        "candidates_mean": float(np.nanmean(counts)) if np.isfinite(counts).any() else np.nan,
        "found": found,
        "images": len(items),
        "corner_err_mean_pct": float(np.nanmean(errs)) if np.isfinite(errs).any() else np.nan,
        "corner_err_median_pct": float(np.nanmedian(errs)) if np.isfinite(errs).any() else np.nan,
        "seconds_per_image": float(np.mean(seconds)) if seconds else np.nan,
    }


def plot(rows, outdir):
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
    for col, name in enumerate(SWEEPS):
        sub = [r for r in rows if r["parameter"] == name]
        x = [r["value"] for r in sub]
        positions = range(len(x))
        found_all = [r["found"] == r["images"] for r in sub]
        for row, key, ylabel in ((0, "candidates_mean", "Candidates per image"),
                                 (1, "corner_err_mean_pct", "Mean corner error (%)")):
            ax = axes[row, col]
            y = [r[key] for r in sub]
            ax.plot(positions, y, color=COLOUR, linewidth=2, zorder=2)
            for p, v, ok in zip(positions, y, found_all):
                ax.scatter(p, v, s=36, zorder=3, color=COLOUR if ok else "white",
                           edgecolors=COLOUR, linewidths=1.6)
            default_pos = x.index(DEFAULTS[name])
            ax.axvline(default_pos, color="#8a8a85", linewidth=1, linestyle="--", zorder=1)
            ax.set_xticks(list(positions))
            ax.set_xticklabels([f"{v:g}" for v in x], rotation=45 if col >= 2 else 0, fontsize=7)
            if row == 0:
                ax.set_yscale("log")
            if row == 1:
                ax.set_xlabel(LABELS[name])
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, color="#e4e4e0", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.text(0.99, 0.01, "dashed = chosen value; hollow marker = corners not found on every image",
             ha="right", fontsize=7, color="#5f5f5a")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outdir / "fig_param_study.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 parameter study: blockSize, ksize, k and threshold.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/out"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    items = load(args.images, args.annotations)
    print(f"{len(items)} annotated images")
    rows = []
    for name, values in SWEEPS.items():
        for value in values:
            stats = run_config(items, {name: value})
            rows.append({"parameter": name, "value": value, **stats})
            print(f"{LABELS[name]:30} {value:<7g} candidates {stats['candidates_mean']:8.0f}  "
                  f"found {stats['found']:2d}/{stats['images']}  corner err mean {stats['corner_err_mean_pct']:6.3f}%  "
                  f"median {stats['corner_err_median_pct']:6.3f}%  {stats['seconds_per_image']:.2f} s/img")

    with open(args.outdir / "param_study.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.outdir)
    print(f"-> {args.outdir}/param_study.csv, fig_param_study.png")


if __name__ == "__main__":
    main()
