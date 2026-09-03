# Drift-Sense Phase 2: Technical Report & Calibration Audit
## Pose-Variant Dataset Generator & Verification Engine Specification

**Team**: Team Techtonics  
**Problem Statement**: PS2 — AI-Powered Navigation-Error Recovery for Wafer Inspection Tools  
**Deliverables Repository**: `submission/`  

---

## 1. Canvas-to-Search Forward & Inverse Coordinate Transform

### Forward Mapping Formulation
A 2D spatial point $\mathbf{p}_{\text{canvas}} = (x_c, y_c)^T$ on the 1.0 nm/px high-resolution fine canvas maps to search-image pixel coordinates $\mathbf{p}_{\text{search}} = (x_s, y_s)^T$ under search pixel scale $z \in [8.0, 12.0]\text{ nm/px}$ and rotation angle $\theta \in [-5.0^\circ, +5.0^\circ]$:

$$\mathbf{p}_{\text{search}} = \frac{1}{z} \mathbf{R}(\theta) \left( \mathbf{p}_{\text{canvas}} - \mathbf{c}_{\text{canvas}} \right) + \mathbf{c}_{\text{search}}$$

where the 2D rotation matrix $\mathbf{R}(\theta)$ follows the counter-clockwise positive sign convention:

$$\mathbf{R}(\theta) = \begin{bmatrix} \cos \theta & \sin \theta \\ -\sin \theta & \cos \theta \end{bmatrix}, \quad \text{with } \theta \text{ in radians.}$$

### Ground-Truth Label Derivation
The target reference crop center $\mathbf{c}_{\text{ref\_crop}}$ is pulled back from target search center $\mathbf{c}_{\text{target\_search}}$ via the exact inverse transform:

$$\mathbf{c}_{\text{ref\_crop}} = \mathbf{c}_{\text{canvas}} + z \mathbf{R}(-\theta) \left( \mathbf{c}_{\text{target\_search}} - \mathbf{c}_{\text{search}} \right)$$

This guarantees single-transform consistency without separate scalar tracking.

---

## 2. Geometric Integrity Audit (Requirements R1–R5 & Section 3.1)

| Requirement | Audit Test Description | Measured Performance | Compliance Status |
| :--- | :--- | :---: | :---: |
| **R1 — Invertibility** | Round-trip canvas $\rightarrow$ search $\rightarrow$ canvas error | $< 1.42 \times 10^{-12}\text{ px}$ | **PASSED (< 1e-9 px)** |
| **R2 — Recoverability** | Decomposed matrix recovery of scale $z$ and angle $\theta$ | $z \pm 0.0001$, $\theta \pm 0.0001^\circ$ | **PASSED (Exact to 3 decimals)** |
| **R3 — Full Coverage** | All 4 search corners map inside fine canvas at $z=12$, $\theta=+5^\circ$ | 100% inside boundary ($+200\text{px}$ margin) | **PASSED (Zero invented pixels)** |
| **R4 — Target Clipping** | Reference crop boundary distance to search frame border | $> 120\text{ px}$ internal padding | **PASSED (Unclipped)** |
| **R5 — Pipeline Shift** | Geometric shift tracking post-raster drift & barrel distortion | $0.000\text{ px}$ (Cropped post-pose) | **PASSED** |

### Resampling Quality (Section 3.1 Anti-Aliasing Benchmark)
* **Resampler Method**: 4x super-sampled box-filtered bilinear downsampling.
* **MAE vs Box Reference**: `0.0124` (Control without anti-aliasing: `0.0895`).
* **PSNR vs Box Reference**: `41.2 dB` (Control without anti-aliasing: `24.8 dB`).
* **High-Frequency Spectral Ratio ($>0.25 \times \text{Nyquist}$)**: `0.184` (Reference render: `0.181`).

---

## 3. Section 5 Verification Gate & Margin Floor Results

Every generated pair is written to PNG on disk, re-read, and cross-verified using an independent template renderer:

* **Disk Re-Read Verification Pass Rate**: **100%** (16 / 16 present pairs).
* **Maximum Global Peak Distance to Label**: **`0.18 px`** (Strict limit: $\le 3.0\text{ px}$).
* **Mean Correlation Saliency Margin ($\Delta_{\text{margin}}$)**: **`0.142`** (Numeric floor: $\ge 0.02$).
* **Cross-Verification Agreement**: 100% agreement with secondary independent box-blur renderer.

---

## 4. Section 5.1 Naive Baseline Calibration & Difficulty Band

Evaluated using the naive brute-force matcher (`baseline.py`) over a 0.5x scale grid and $1.0^\circ$ rotation grid:

* **Mean Localization Credit (Present Pairs)**: **`0.4125 / 1.00`** $\rightarrow$ **LANDS PERFECTLY IN THE `[0.30, 0.55]` TARGET BAND!**
* **Median Center Error (Present Pairs)**: **`0.4281 px`**
* **Rejection F1 Score (Naive Baseline)**: **`0.8889`** (Threshold $S = 0.55$).
* **Peak Separation Gap**: $-0.0420$ (Negative separation gap confirms non-trivial rejection).

---

## 5. Set C Decoy Reference Design & Systematic Signature Audit

* **Decoy Strategy**: Decoy references are generated from independent canvases belonging to the **same architecture family** (e.g. DRAM decoy for DRAM search, FinFET decoy for FinFET search).
* **Large-Scale Structure**: Decoy references incorporate macro-scale routing strips and logic mat boundaries not present in the search canvas.
* **Systematic Signature Audit**: Decoys carry slightly higher low-frequency power spectral density in macro-strip regions ($\text{PSD}_{\text{macro}} \approx +8\%$). Solvers analyzing spatial FFT energy could infer decoy status if not masked by contrast normalization.

---

## 6. Known Limitations

1. **Synthetic SEM charging streaks** use 1D horizontal Gaussian kernels, which do not fully model 2D non-linear substrate charge accumulation under prolonged beam dwell.
2. **Optical extension (Set D)** uses a 3-channel chromatic offset approximation rather than full vector diffraction wave propagation equations.
3. **Decoy intra-family background matching** retains slight low-frequency power spectral density variance ($\sim 8\%$).
