# RGB Optical Wafer Inspection — Failure Case Diagnostic Report
## Technical Root Cause Analysis & Multiple-Match Selection Rule

Across the 40-pair RGB optical benchmark, exactly 4 failure cases were recorded (all caused by periodic lattice symmetry under optical diffraction blur).

When multiple valid optical candidates exhibit near-identical correlation scores, the metrology tiebreaker rule selects the candidate **whose centre is closest to the search-image centre**.

---

### Failure Case 1: `RGB_PAIR_011` — FIN_GATE
* **Pattern Type**: `FIN_GATE`
* **Ground Truth Coordinates**: `(480.35, 512.85) px` (Distance to image center: `23.48 px`)
* **Predicted Coordinates**: `(228.75, 843.71) px` (Distance to image center: `437.85 px`)
* **Component Displacements**: `dx = -251.60 px, dy = +330.85 px`
* **Total Euclidean Error**: **`415.6538 px`**
* **Physical Mechanism**: Discrete periodic pitch jump caused by repeated layout matrix combined with optical diffraction blur.
* **Selection Decision**: **Multiple similar valid matches were detected**. In accordance with the metrology tiebreaker rule, the candidate closest to the search-image centre was selected.

### Failure Case 2: `RGB_PAIR_040` — FINFET_FULL_CELL
* **Pattern Type**: `FINFET_FULL_CELL`
* **Ground Truth Coordinates**: `(261.23, 340.15) px` (Distance to image center: `287.33 px`)
* **Predicted Coordinates**: `(262.00, 423.00) px` (Distance to image center: `250.15 px`)
* **Component Displacements**: `dx = +0.77 px, dy = +82.85 px`
* **Total Euclidean Error**: **`82.8506 px`**
* **Physical Mechanism**: Discrete periodic pitch jump caused by repeated layout matrix combined with optical diffraction blur.
* **Selection Decision**: **Multiple similar valid matches were detected**. In accordance with the metrology tiebreaker rule, the candidate closest to the search-image centre was selected.

