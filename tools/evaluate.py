import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

TIME_LIMIT_S = 20.0
CONDITIONS = ("C1", "C2", "C3", "C4", "C5")


def corner_error_pct(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    side = np.mean(np.linalg.norm(truth - np.roll(truth, -1, axis=0), axis=1))
    return float(100 * np.linalg.norm(pred - truth, axis=1).mean() / side)


def evaluate_image(result, ann):
    truth = np.array(ann["grid"])
    grid = np.array(result["grid"]) if result and result.get("grid") is not None else np.zeros((9, 9), int)
    corners = result.get("corners") if result else None
    pred_filled, true_filled = grid > 0, truth > 0
    return {
        "image": Path(ann["image"]).stem,
        "conditions": " ".join(ann.get("conditions", [])),
        "verdict": result.get("verdict", "missing") if result else "missing",
        "runtime_s": result.get("runtime_s") if result else None,
        "robust": bool(result) and result.get("runtime_s") is not None and result["runtime_s"] <= TIME_LIMIT_S,
        "corner_err_pct": corner_error_pct(corners, ann["corners"]) if corners else None,
        "cell_acc": float(np.mean(grid == truth)),
        "exact": bool(np.all(grid == truth)),
        "tp": int(np.sum(pred_filled & true_filled)),
        "fp": int(np.sum(pred_filled & ~true_filled)),
        "fn": int(np.sum(~pred_filled & true_filled)),
        "confusions": [(int(t), int(p)) for t, p in zip(truth.ravel(), grid.ravel()) if t != p],
    }


def summarise(rows):
    if not rows:
        return None
    errs = [r["corner_err_pct"] for r in rows if r["corner_err_pct"] is not None]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {
        "images": len(rows),
        "corners_found": len(errs),
        "corner_err_mean_pct": float(np.mean(errs)) if errs else None,
        "corner_err_median_pct": float(np.median(errs)) if errs else None,
        "cell_acc": float(np.mean([r["cell_acc"] for r in rows])),
        "empty_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "exact_match": float(np.mean([r["exact"] for r in rows])),
        "robustness": float(np.mean([r["robust"] for r in rows])),
    }


def fmt(value, pct=False, digits=2):
    if value is None:
        return "-"
    return f"{100 * value:.{digits}f}%" if pct else f"{value:.{digits}f}"


def print_table(groups):
    header = f"{'set':10} {'n':>3} {'found':>5} {'corner err':>10} {'cell acc':>9} {'empty F1':>9} {'exact':>7} {'robust':>7}"
    print(header)
    print("-" * len(header))
    for name, s in groups:
        if s is None:
            continue
        print(f"{name:10} {s['images']:3d} {s['corners_found']:5d} "
              f"{fmt(s['corner_err_mean_pct']) + '%':>10} {fmt(s['cell_acc'], True):>9} "
              f"{fmt(s['empty_f1'], digits=4):>9} {fmt(s['exact_match'], True, 0):>7} {fmt(s['robustness'], True, 0):>7}")


def main():
    parser = argparse.ArgumentParser(description="Section 7 metrics for SudokuScan results against annotations.")
    parser.add_argument("--results", type=Path, required=True, help="output directory of sudoku_scan.py --batch")
    parser.add_argument("--annotations", type=Path, required=True, help="directory of Appendix A JSON annotations")
    parser.add_argument("--csv", type=Path, help="optional per-image CSV output")
    args = parser.parse_args()

    rows = []
    for ann_path in sorted(args.annotations.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        result_path = args.results / f"{Path(ann['image']).stem}_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        rows.append(evaluate_image(result, ann))
    if not rows:
        raise SystemExit(f"no annotations found in {args.annotations}")

    groups = [("all", summarise(rows))]
    groups += [(c, summarise([r for r in rows if c in r["conditions"].split()])) for c in CONDITIONS]
    print_table(groups)

    print("\nper image:")
    for r in rows:
        err = fmt(r["corner_err_pct"]) + "%" if r["corner_err_pct"] is not None else "-"
        wrong = f"{len(r['confusions'])} wrong" if r["confusions"] else "exact"
        print(f"  {r['image']:12} {r['conditions']:12} {r['verdict']:9} corner {err:>7}  "
              f"cells {fmt(r['cell_acc'], True):>8} ({wrong})  {fmt(r['runtime_s'])} s")

    confusions = Counter(pair for r in rows for pair in r["confusions"])
    if confusions:
        print("\nmost frequent confusions (true -> predicted, 0 = empty):")
        for (t, p), n in confusions.most_common(10):
            print(f"  {t} -> {p}: {n}")

    if args.csv:
        fields = [k for k in rows[0] if k != "confusions"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nper-image CSV -> {args.csv}")


if __name__ == "__main__":
    main()
