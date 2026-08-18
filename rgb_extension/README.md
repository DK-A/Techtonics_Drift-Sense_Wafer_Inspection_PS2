# RGB Optical Wafer Inspection Extension (Bonus Challenge)

## 1. Overview & Physical Formulation

This extension demonstrates the **cross-modal generalization** of our 5-Phase Localization Cascade from electron microscopy (SEM) to **3-Channel Visible-Light Optical Microscopy (RGB)**.

In semiconductor manufacturing, optical inspection tools (Brightfield/Darkfield wafer review stations, spectral ellipsometers, and defect scanners) operate in the visible light regime ($\lambda \in [400\text{ nm}, 700\text{ nm}]$).

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            PHYSICS DEGRADATION MODEL (RGB OPTICAL)                       │
├──────────────────────────┬───────────────────────────────┬───────────────────────────────┤
│ Physical Degradation     │ Mathematical Formulation      │ Semiconductor Physical Origin │
├──────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Thin-Film Interference   │ R(λ) = R0·[1 + V·cos(4πnd/λ)] │ SiO2 / Si3N4 oxide thickness  │
│ Optical Diffraction Blur │ PSF(r) = [2·J1(kr)/kr]^2      │ Numerical Aperture NA=0.85    │
│ Specular Reflection      │ I_glare = I0·exp(-r^2/2σ^2)   │ Highly reflective Cu/W metals │
│ Microscope Vignetting    │ V(r) = 1 - α·(r / R_max)^2    │ Lens periphery radial falloff │
│ Optical Sensor Noise     │ N ~ N(0, σ_sensor^2)          │ CMOS/CCD photon shot noise    │
└──────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## 2. 40-Pair RGB Optical Benchmark Dataset

The dataset consists of **40 high-resolution RGB image pairs** ($80$ images of size $1000 \times 1000 \times 3$) spanning all **8 required semiconductor pattern classes (P1–P8)** across 5 physical augmentation tiers:

1. **`NOMINAL_BRIGHTFIELD`**: Standard optical focus ($\sigma_{\text{blur}} = 1.0\text{ px}$), clean color contrast, nominal 10:1 scale.
2. **`THIN_FILM_DISPERSION`**: Multi-wavelength interference color shifts ($SiO_2$ thickness $d \in [120\text{ nm}, 300\text{ nm}]$).
3. **`DIFFRACTION_BLUR`**: Severe optical diffraction limit blur ($\sigma_{\text{blur}} = 2.2\text{ px}$) + optical astigmatism.
4. **`SPECULAR_GLARE`**: Localized specular reflection glare on copper metal lines + lens vignetting ($\alpha = 0.40$).
5. **`MIXED_STRESS`**: Off-center spatial placement + scale variations ($0.092\text{--}0.109$) + rotation ($\pm 1.8^\circ$) + stage drift ($\pm 9.5\text{ px}$).

---

## 3. Zero-Intervention Cross-Modal Generalization

The **exact same localization cascade (`localize.py`)** runs on 3-channel RGB optical micrographs without architectural changes:
1. **Luminance Projection**: Extracts ITU-R BT.601 perceptual luminance ($Y = 0.299R + 0.587G + 0.114B$) to capture invariant structural geometry.
2. **Local Contrast Normalization (CLAHE)**: Suppresses specular glares and vignetting gradients.
3. **2D Fourier Phase Correlation**: Resolves sub-pixel coordinates in the frequency domain, inherently invariant to optical diffraction blur.

---

## 4. Quickstart Execution Guide

```bash
# Step 1: Generate the 40-Pair RGB Optical Dataset
python generate_rgb_dataset.py

# Step 2: Run Evaluation & Generate Diagnostic Plots
python evaluate_rgb.py
```

---

## 5. Directory Structure

```text
rgb_extension/
├── README.md                 # Technical documentation & optical physics model
├── generate_rgb_dataset.py   # 40-pair physics-based RGB optical dataset generator
├── evaluate_rgb.py           # Benchmark evaluation script & plot generator
├── dataset/                  # 40 RGB Optical Pairs (80 images + manifest)
│   ├── reference/            # 100x High-Mag RGB Reference Micrographs (1000x1000x3)
│   ├── search/               # 10x Low-Mag RGB Search Field Micrographs (1000x1000x3)
│   └── manifest_rgb.csv      # Ground-truth coordinates & optical parameters
└── results/                  # Benchmark predictions & plots
    ├── predictions_rgb.csv   # Localized coordinates & runtimes
    ├── overall_rgb_metrics.csv
    └── plots/                # Error CDF & per-pattern bar charts
```
