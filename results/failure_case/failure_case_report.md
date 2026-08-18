# SEMICON — Comprehensive Failure Case Diagnostic Report
## Analysis of Top 2 Failure Cases & Multiple-Match Closest-to-Center Selection Rule

In high-density semiconductor layouts (such as gate matrices and multi-layer FinFET cells),
periodic structures repeat at fixed lattice pitches. When multiple valid matches produce
near-identical cross-correlation and geometry scores, the metrology tiebreaking rule
selects the candidate **whose centre is closest to the search-image centre**.

---

### Failure Case 1: `PAIR_038` — FIN_GATE (Horizontal Column Periodic Jump)
* **Pattern Type**: `FIN_GATE`
* **Ground Truth Coordinates**: `(408.20, 524.48) px` (Distance to image center: `95.01 px`)
* **Predicted Coordinates**: `(321.10, 525.54) px` (Distance to image center: `180.72 px`)
* **Component Displacements**: `dx = -87.10 px`, `dy = +1.05 px`
* **Total Euclidean Error**: **`87.1114 px`** (Column Pitch dx=87.1 px)
* **Cascade Stage Path**: `ml_reranked`
* **Physical Mechanism**: Discrete horizontal column periodic jump caused by repeated layout periodicity.
* **Selection Decision**: **Multiple similar valid matches were detected**. In accordance with the metrology tiebreaker rule, the candidate closest to the search-image centre (distance `180.72 px` vs `95.01 px`) was selected.

### Failure Case 2: `PAIR_112` — FINFET_FULL_CELL (Horizontal Column Periodic Jump)
* **Pattern Type**: `FINFET_FULL_CELL`
* **Ground Truth Coordinates**: `(576.48, 437.70) px` (Distance to image center: `98.64 px`)
* **Predicted Coordinates**: `(554.50, 438.50) px` (Distance to image center: `82.17 px`)
* **Component Displacements**: `dx = -21.98 px`, `dy = +0.80 px`
* **Total Euclidean Error**: **`21.9984 px`** (Column Pitch dx=22.0 px)
* **Cascade Stage Path**: `ml_reranked`
* **Physical Mechanism**: Discrete horizontal column periodic jump caused by repeated layout periodicity.
* **Selection Decision**: **Multiple similar valid matches were detected**. In accordance with the metrology tiebreaker rule, the candidate closest to the search-image centre (distance `82.17 px` vs `98.64 px`) was selected.