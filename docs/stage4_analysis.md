# Stage 4 — Affine versus homography (draft for report section 4)

> Written by Claude (AI) at the author's explicit request, against the Section 11 rule.
> Must be declared in Appendix C. Numbers are from the **development set** (20 synthetic images);
> the spec requires the final analysis on ≥ 8 images of the **personal dataset** — re-run
> `analysis/affine_vs_h.py` on it and update the numbers.

## Setup

- **Models.** A3 = exact affine from TL, TR, BL; A4 = least-squares affine from all four
  corners; H4 = homography from all four corners. Target: 450 × 450 px, 50 px per cell.
- **Obliqueness.** Primary measure: the largest ratio between opposite sides of the detected
  quadrilateral (1.0 = no keystone). Secondary, physically motivated measure: the relative
  depth spread Δz / z̄ of the four puzzle corners, read from the third row of H4⁻¹
  (for a plane, the projective denominator is proportional to camera depth). The two agree
  closely (r = 0.96 on the devset); Δz / z̄ is the quantity weak perspective is about.
- **Grid straightness.** In each rectification the 10 horizontal and 10 vertical lines are
  located separately in each of the 9 cell-wide strips (180 segments per image) from an
  ink-profile peak; border lines from their outer ink edge. Reported: mean deviation from the
  ideal position and the share of segments within 3 px. Deviations saturate at ~15 px because
  the search window is ±20 px, so beyond that the "in place" share is the better measure.
- **Recognition.** Cell accuracy of the Stage 5 OCR on A3, A4 and H4 — *pending Stage 5*.

## Results (devset, detected corners)

| Obliqueness (side ratio) | Δz / z̄ | A3 4th-corner err | A4 RMS | Mean line dev A3 / A4 / H4 | Segments within 3 px A3 / A4 / H4 |
|---|---|---|---|---|---|
| 1.04 – 1.06 (5 imgs)  | 0.06 – 0.13 | 20 – 37 px   | 5 – 10 px  | 2.3–7.8 / 3.4–5.1 / 0.2–3.3 px | 19–74 / 34–64 / 73–100 % |
| 1.10 – 1.16 (6 imgs)  | 0.14 – 0.23 | 46 – 79 px   | 12 – 19 px | 6.5–15 / 7.1–10.5 / 0.5–1.6 px | 12–36 / 14–36 / 92–100 % |
| 1.29 – 1.64 (9 imgs)  | 0.30 – 0.54 | 110 – 257 px | 29 – 54 px | 13–17 / 11–17 / 0.5–5.3 px    | 11–23 / 6–27 / 49–100 %  |

Full per-image values: `analysis/out/stage4.csv`. Figures: `fig_residuals.png`,
`fig_grid_deviation.png`, `fig_depth_spread.png`, `fig_side_by_side.png`.

Observations:

1. **Residuals grow linearly with depth spread.** A4 RMS ≈ 98 px × Δz / z̄ (r = 0.99), i.e.
   about 0.22 × the target side per unit of relative depth. A3's fourth-corner error is a
   median 3.8 × the A4 RMS: A3 fits three corners exactly and dumps the whole model error on
   BR, while A4 spreads it over all four corners.
2. **H4 is flat at ~0.5 px** at every angle. Re-running with the annotated corners
   (`--gt-corners`) drops it to ~0.05 px, so the remaining 0.5 px is Stage 2 corner error, not
   a model error. The two exceptions are dev_19 / dev_20 (page curl, 1–5 px): no 3 × 3 map can
   remove a non-planar sheet.
3. **Even the most frontal devset images are not affine-perfect.** At side ratio ≈ 1.03 with
   annotated corners, A4 still has ~2 px mean line deviation and 79–87 % of segments in place,
   against 100 % for H4. The devset has no image with ratio below 1.03.
4. **Visual.** In the most oblique image (fig_side_by_side) the affine rectifications keep
   parallel lines parallel, so the converging grid lines of the photo stay converging: cells at
   the far edge are compressed, cells at the near edge stretched, and grid lines cross cell
   interiors. H4 maps the vanishing points back to infinity and the grid lands on the ideal lines.

## Conclusion

An affine model is an acceptable approximation only when the depth variation across the puzzle
is small compared with its distance from the camera. On this data, mean grid-line deviation
stays at or below ~2 px for Δz / z̄ ≲ 0.05 (side ratio ≲ 1.03) and below ~5 px — a tenth of a
cell, inside a typical OCR margin — up to Δz / z̄ ≈ 0.1 (side ratio ≈ 1.05–1.07). Beyond
Δz / z̄ ≈ 0.15, lines drift by 7–15 px and cross into neighbouring cells, which would put grid
lines inside the OCR crops; a homography is then required.

This is exactly the **weak-perspective** condition from the lectures: when every scene point
has nearly the same depth (Δz ≪ z̄, the usual rule of thumb Δz < z̄ / 10), perspective division
by z can be replaced by division by a single average depth z̄, the projection becomes affine,
and parallel lines stay parallel. In homography terms the projective row (h₃₁, h₃₂) is then ≈ 0.
Tilting the phone increases Δz / z̄ linearly with the tilt, the vanishing points move in from
infinity, and the affine error grows in proportion, as the residual plot shows. A3 is the
worst choice because it extrapolates to BR with no constraint there, which is why its
fourth-corner error grows fastest with obliqueness.

## Limitations / to do

- Devset images are synthetic; repeat on the personal dataset (≥ 8 images, C1 to strong C2).
- Add the recognition-accuracy comparison once Stage 5 exists (the script already runs it).
- The line-deviation measure saturates at ~15 px; for strongly oblique images use the
  "within 3 px" share.
