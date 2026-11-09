# Stage 4 — Affine versus homography (draft for report section 4)

> Written by Claude (AI) at the author's explicit request, against the Section 11 rule.
> Must be declared in Appendix C.

Main evidence: the **personal dataset** (12 phone photos, 3 puzzles, side ratio 1.05–1.66).
The 20 synthetic development images are used as a cross-check.
Reproduce with:

```bash
python analysis/affine_vs_h.py --images images --annotations annotations --outdir analysis/out_personal
python analysis/affine_vs_h.py --images images --annotations annotations --outdir analysis/out_personal_gt --gt-corners
```

## Setup

- **Models.** A3 = exact affine from TL, TR, BL; A4 = least-squares affine from all four
  corners; H4 = homography from all four corners. Target 450 × 450 px, 50 px per cell.
- **Obliqueness.** Primary measure: largest ratio of opposite sides of the detected
  quadrilateral (1.0 = no keystone). Secondary, physically motivated measure: the relative
  depth spread Δz / z̄ of the four puzzle corners, read from the third row of H4⁻¹ (for a
  plane, the projective denominator is proportional to camera depth). They agree (r = 0.90
  personal, 0.96 devset); Δz / z̄ is what weak perspective is about, and it is the better
  measure when the photo is both tilted and rotated (IMG_8801: side ratio 1.38 but
  Δz / z̄ = 0.58, the largest in the set).
- **Grid straightness.** The 10 horizontal and 10 vertical lines of each rectification are
  located separately in each of the 9 cell-wide strips (180 segments per image); reported as
  mean deviation from the ideal 50 px positions and share of segments within 3 px. The
  measure saturates near 15 px (±20 px search window).
- **Recognition.** The same Stage 5 OCR run on A3, A4 and H4; cell accuracy over all 81 cells.

## Results — personal dataset (detected corners)

| Image | Side ratio | Δz / z̄ | A3 4th-corner err (px) | A4 RMS (px) | Mean line dev A3 / A4 / H4 (px) | Cell accuracy A3 / A4 / H4 (%) |
|---|---|---|---|---|---|---|
| IMG_8802 | 1.046 | 0.049 | 20.7 | 5.1 | 5.1 / 3.9 / 0.7 | 95.1 / 100 / 96.3 |
| IMG_8795 | 1.063 | 0.062 | 28.4 | 6.9 | 6.2 / 4.0 / 1.3 | 92.6 / 100 / 100 |
| IMG_8800 | 1.070 | 0.092 | 33.3 | 8.1 | 9.0 / 4.8 / 1.7 | 77.8 / 100 / 100 |
| IMG_8794 | 1.087 | 0.089 | 39.2 | 9.4 | 8.1 / 6.0 / 2.2 | 81.5 / 100 / 98.8 |
| IMG_8805 | 1.099 | 0.106 | 45.0 | 10.6 | 8.3 / 6.7 / 1.2 | 84.0 / 95.1 / 100 |
| IMG_8803 | 1.115 | 0.114 | 51.7 | 12.2 | 8.6 / 5.8 / 1.1 | 71.6 / 91.4 / 100 |
| IMG_8804 | 1.176 | 0.169 | 78.8 | 18.1 | 12.1 / 8.4 / 1.2 | 65.4 / 86.4 / 100 |
| IMG_8796 | 1.276 | 0.245 | 124.5 | 27.1 | 14.5 / 11.6 / 2.1 | 53.1 / 63.0 / 98.8 |
| IMG_8798 | 1.330 | 0.286 | 148.0 | 31.5 | 14.6 / 12.6 / 0.6 | 64.2 / 58.0 / 98.8 |
| IMG_8801 | 1.380 | 0.583 | 205.5 | 46.7 | 13.8 / 14.6 / 1.6 | 72.8 / 66.7 / 100 |
| IMG_8797 | 1.463 | 0.377 | 208.4 | 41.6 | 14.1 / 10.7 / 1.6 | 67.9 / 70.4 / 100 |
| IMG_8799 | 1.658 | 0.497 | 296.6 | 54.1 | 15.3 / 13.9 / 7.0 | 49.4 / 46.9 / 93.8 |
| **Mean** | | | | | | **72.9 / 81.5 / 98.9** |

Per-image CSV: `analysis/out_personal/stage4.csv`. Figures in `analysis/out_personal/`:
`fig_residuals.png`, `fig_grid_deviation.png`, `fig_cell_accuracy.png`,
`fig_depth_spread.png`, `fig_side_by_side.png` (most oblique image, IMG_8799).

Development set (20 synthetic images, side ratio 1.04–1.64): mean cell accuracy
A3 72.0 %, A4 74.8 %, H4 99.8 %; A4 RMS ≈ 98 px × Δz / z̄ (r = 0.99).

## Observations

1. **Corner residuals grow linearly with depth spread.** Personal set: A4 RMS ≈ 93 px × Δz / z̄
   + 1.9 px (r = 0.97); devset: 98 px × Δz / z̄ (r = 0.99). The same slope on synthetic and
   real photos means the residual is a property of the geometry, not of the images. A3's
   fourth-corner error is a median 4.3 × the A4 RMS: A3 fits TL, TR, BL exactly and
   extrapolates to BR with no constraint there, while A4 spreads the error over all four.
2. **H4 is flat across obliqueness**, but not perfect on real paper: 0.6–2.2 px mean line
   deviation, and 7.0 px on IMG_8799. Repeating with the annotated corners gives the same
   numbers (0.7–2.5 px, 7.6 px on IMG_8799), so this is not corner error. It is the limit of
   the model itself: a 3 × 3 homography is exact only for a flat plane and a pinhole camera.
   Printed paper on a desk is slightly curled (strongly at the top of IMG_8799, visible in
   `fig_side_by_side.png`) and the phone lens adds a little radial distortion. The synthetic
   devset, which is perfectly planar, gives ~0.05 px with annotated corners.
3. **Recognition shows where the affine error starts to matter.** With A4, OCR is 100 % up to
   Δz / z̄ ≈ 0.09 (IMG_8802, 8795, 8800, 8794), drops to 91–95 % at 0.11, to 86 % at 0.17 and to
   47–70 % beyond 0.24. The break-even matches the grid-line deviation: A4 stays at ≤ 6 px
   (the 7 px cell margin) up to Δz / z̄ ≈ 0.09–0.11, after which grid lines cross into the cell
   crops and digits are cut or shifted. A3 is never reliable (72.9 % mean, 92–95 % at best),
   because all its error lands in the bottom-right part of the grid.
4. **Even the most frontal hand-held photos are not affine-perfect.** The smallest depth
   spread in the set is 0.049: holding a phone "straight above" the page still leaves ~5 % depth
   variation. Grid lines then deviate 4 px under A4 and only 52–60 % of segments are within
   3 px, against 90–98 % for H4.

## Conclusion

An affine model is an acceptable approximation only when the depth variation across the
puzzle is small compared with its distance from the camera. On our photos, A4 reads every cell
correctly up to a relative depth spread Δz / z̄ ≈ 0.09 (side ratio ≈ 1.09; with the puzzle filling about half
the frame width this is a camera tilt of roughly 10°), and loses accuracy quickly after Δz / z̄ ≈ 0.1. A3 is not acceptable at any
angle in our data, and H4 is needed for anything beyond near-frontal shots.

This is the **weak-perspective** condition from the lectures. Perspective projection divides
each point by its own depth z; weak perspective replaces every z by one average depth z̄,
which is valid when Δz ≪ z̄ (rule of thumb Δz < z̄ / 10). Under that approximation the
projection of a plane is affine: parallel lines stay parallel and the projective row
(h₃₁, h₃₂) of the homography is zero. Tilting the phone increases Δz / z̄ roughly in proportion
to the tilt, the grid's vanishing points move in from infinity, and the affine error grows in
proportion (residual ∝ Δz / z̄, r = 0.97). Our measured break-even, Δz / z̄ ≈ 0.1, is exactly
the rule-of-thumb limit of weak perspective.

## Limitations

- 12 images, 3 puzzles; only one image between Δz / z̄ = 0.17 and 0.24.
- No photo is truly frontal (minimum Δz / z̄ = 0.049).
- The line-deviation measure saturates at ~15 px; beyond side ratio ~1.25 use the share of
  segments within 3 px or the OCR accuracy instead.
- OCR accuracy mixes geometry with OCR noise (e.g. IMG_8802, where A4 scores 100 % and H4
  96.3 %): single misreads are not evidence for one model over the other.
- A planar homography cannot remove page curl (IMG_8799); this is the remaining H4 error.
