import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sudoku_scan as ss

WHITELIST = "-c tessedit_char_whitelist=123456789"
PSMS = (6, 7, 8, 10, 13)


def read(image, psm, whitelist):
    config = f"--psm {psm} --oem 1" + (f" {WHITELIST}" if whitelist else "")
    text = pytesseract.image_to_string(image, config=config)
    digits = [ch for ch in text if ch in "123456789"]
    return int(digits[0]) if digits else 0


def collect(images_dir, annotations_dir):
    cells = []
    for ann_path in sorted(annotations_dir.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        image = cv2.imread(str(images_dir / ann["image"]))
        rect = ss.rectify(image, np.float32(ann["corners"]))["H4_image"]
        grey, flat, ink = ss.rect_ink(rect)
        for r in range(9):
            for c in range(9):
                truth = ann["grid"][r][c]
                if truth == 0:
                    continue
                y0, y1, x0, x1 = ss.cell_box(r, c)
                mask = ss.digit_mask(ink[y0:y1, x0:x1])
                prepared = ss.prepare_for_ocr(flat[y0:y1, x0:x1], mask) if mask is not None else None
                cells.append({"image": ann_path.stem, "row": r, "col": c, "truth": truth,
                              "raw": grey[y0:y1, x0:x1], "prepared": prepared})
    return cells


def main():
    parser = argparse.ArgumentParser(description="Stage 5 OCR study: page-segmentation mode, whitelist and cell preparation.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/out"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cells = collect(args.images, args.annotations)
    print(f"{len(cells)} filled cells from {len({c['image'] for c in cells})} images (annotated corners, H4)")
    rows, worst = [], None
    for prep in ("prepared", "raw"):
        for whitelist in (True, False):
            for psm in PSMS:
                wrong = Counter()
                correct = 0
                for cell in cells:
                    image = cell[prep]
                    pred = read(image, psm, whitelist) if image is not None else 0
                    if pred == cell["truth"]:
                        correct += 1
                    else:
                        wrong[(cell["truth"], pred)] += 1
                acc = correct / len(cells)
                rows.append({"input": prep, "whitelist": whitelist, "psm": psm, "digit_accuracy": acc,
                             "unread": sum(n for (t, p), n in wrong.items() if p == 0),
                             "misread": sum(n for (t, p), n in wrong.items() if p != 0),
                             "top_confusions": " ".join(f"{t}->{p}:{n}" for (t, p), n in wrong.most_common(4))})
                print(f"{prep:9} whitelist={str(whitelist):5} psm={psm:2d}  digit accuracy {100 * acc:6.2f}%  "
                      f"unread {rows[-1]['unread']:3d}  misread {rows[-1]['misread']:3d}  {rows[-1]['top_confusions']}")

    with open(args.outdir / "ocr_study.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    chosen = [c for c in cells if c["prepared"] is not None]
    misses = [c for c in chosen if ss.ocr_digit(c["prepared"]) != c["truth"]]
    if misses:
        tiles = []
        for c in misses[:24]:
            raw = cv2.cvtColor(cv2.resize(c["raw"], (96, 96)), cv2.COLOR_GRAY2BGR)
            prep = cv2.cvtColor(cv2.resize(c["prepared"], (96, 96)), cv2.COLOR_GRAY2BGR)
            tile = np.hstack([raw, prep])
            pred = ss.ocr_digit(c["prepared"])
            cv2.putText(tile, f"{c['truth']}->{pred}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(args.outdir / "fig_ocr_failures.png"), sheet)
    print(f"pipeline config (prepared, whitelist, psm 10 + psm 13 fallback): "
          f"{100 * (1 - len(misses) / len(cells)):.2f}% of filled cells, {len(misses)} failures "
          f"-> {args.outdir}/ocr_study.csv, fig_ocr_failures.png")


if __name__ == "__main__":
    main()
