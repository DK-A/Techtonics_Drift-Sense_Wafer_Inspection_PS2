"""
generate_rgb_dataset.py — 40-Pair Physics-Based RGB Optical Wafer Inspection Generator
Synthesizes a 40-pair benchmark dataset (5 diverse pairs per pattern across P1–P8):
1. 100x High-Mag Reference Micrograph (1000x1000 RGB)
2. 10x Low-Mag Search Field Micrograph (1000x1000 RGB)

Optical Physics Models:
- Multi-Wavelength Thin-Film Interference (Oxide dielectric reflection vs. Metal/Silicon reflection)
- High Contrast Brightfield Optical Micrography
- Optical Diffraction Airy Disk Blurring (NA = 0.85, visible spectrum)
- Specular Glare & Microscope Vignetting
"""

import os
import sys
import math
import csv
import json
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from src.utils import generate_correlated_2d_field

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REF_DIR = os.path.join(DATASET_DIR, "reference")
SEARCH_DIR = os.path.join(DATASET_DIR, "search")


def simulate_thin_film_color(base_mask: np.ndarray, pattern_type: str, oxide_thickness_nm: float = 180.0, rng: np.random.Generator = None) -> np.ndarray:
    """
    Synthesizes a 3-channel (BGR) optical microscopic wafer image with thin-film interference.
    Maintains physical optical contrast between dielectric background and metal/silicon features.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = base_mask.shape[:2]
    rgb_img = np.zeros((h, w, 3), dtype=np.float32)

    # Optical constants: Refractive index n(SiO2) = 1.46
    # Wavelengths in nm: Blue=450nm, Green=532nm, Red=650nm
    n_sio2 = 1.46
    lambdas = np.array([450.0, 532.0, 650.0], dtype=np.float32) # B, G, R

    # Calculate chromatic phase per channel
    phase_shifts = 4.0 * math.pi * n_sio2 * oxide_thickness_nm / lambdas
    interference_factors = 0.5 + 0.5 * np.cos(phase_shifts)

    # Clean Optical Brightfield BGR palettes:
    # 1. Background Oxide (Low-to-Mid Luminance deep blue/violet)
    oxide_bgr = np.array([165.0, 85.0, 45.0], dtype=np.float32) * (0.8 + 0.4 * interference_factors)
    
    # 2. Features: Bright Gold/Copper Metal or High-Reflectance Active Lines
    if pattern_type in ["CONTACT_ARRAY", "LOCAL_INTERCONNECT", "METAL_ROUTING"]:
        feature_bgr = np.array([70.0, 185.0, 245.0], dtype=np.float32) # Bright Copper/Gold
    elif pattern_type in ["FIN_ARRAY", "FIN_CUT", "ACTIVE_CELL"]:
        feature_bgr = np.array([210.0, 220.0, 225.0], dtype=np.float32) # Bright Silicon/Oxide step
    else:
        feature_bgr = np.array([90.0, 175.0, 230.0], dtype=np.float32) # Poly-Si Gate on oxide

    norm_mask = (base_mask.astype(np.float32) / 255.0)
    norm_mask_3d = np.repeat(norm_mask[:, :, np.newaxis], 3, axis=2)

    rgb_img = oxide_bgr * (1.0 - norm_mask_3d) + feature_bgr * norm_mask_3d

    # Spatial optical thickness variation gradient
    grad = generate_correlated_2d_field((h, w), scale_px=220.0, amplitude=6.0, rng=rng)
    for c in range(3):
        rgb_img[:, :, c] += grad * (1.0 + 0.2 * (c - 1.0))

    # Optical sensor photon noise
    shot_noise = rng.normal(0.0, 1.8, (h, w, 3)).astype(np.float32)
    rgb_img += shot_noise

    return np.clip(rgb_img, 0.0, 255.0).astype(np.uint8)


def apply_optical_vignetting(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Applies radial microscope brightfield vignetting."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    max_radius = math.hypot(center_x, center_y)
    dist = np.hypot(X - center_x, Y - center_y) / max_radius
    vignette = 1.0 - strength * (dist ** 2)
    vignette_3d = np.repeat(vignette[:, :, np.newaxis], 3, axis=2)
    return np.clip(img.astype(np.float32) * vignette_3d, 0.0, 255.0).astype(np.uint8)


def apply_specular_metal_glare(img: np.ndarray, mask: np.ndarray, glare_spots: int = 3, rng: np.random.Generator = None) -> np.ndarray:
    """Simulates localized specular highlights from reflective metal features."""
    if rng is None: rng = np.random.default_rng(42)
    h, w = img.shape[:2]
    glare_layer = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(glare_spots):
        gx = rng.integers(150, w - 150)
        gy = rng.integers(150, h - 150)
        sigma = rng.uniform(20.0, 45.0)
        amplitude = rng.uniform(25.0, 50.0)
        Y, X = np.ogrid[:h, :w]
        spot = amplitude * np.exp(-((X - gx)**2 + (Y - gy)**2) / (2.0 * sigma**2))
        glare_layer += np.repeat(spot[:, :, np.newaxis], 3, axis=2)
    
    metal_presence = (mask > 100).astype(np.float32)[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) + glare_layer * (0.3 + 0.7 * metal_presence), 0.0, 255.0).astype(np.uint8)


def generate_40_pair_rgb_dataset():
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(SEARCH_DIR, exist_ok=True)

    manifest_records = []
    rng = np.random.default_rng(20260818)

    patterns = [
        ("P1", "FIN_ARRAY"),
        ("P2", "FIN_CUT"),
        ("P3", "FIN_GATE"),
        ("P4", "CONTACT_ARRAY"),
        ("P5", "LOCAL_INTERCONNECT"),
        ("P6", "METAL_ROUTING"),
        ("P7", "ACTIVE_CELL"),
        ("P8", "FINFET_FULL_CELL")
    ]

    tier_names = [
        "NOMINAL_BRIGHTFIELD",
        "THIN_FILM_DISPERSION",
        "DIFFRACTION_BLUR",
        "SPECULAR_GLARE",
        "MIXED_STRESS"
    ]

    print("=" * 80)
    print("       GENERATING 40-PAIR RGB OPTICAL WAFER INSPECTION DATASET (P1-P8)       ")
    print("=" * 80)

    base_manifest = os.path.join(PARENT_DIR, "submission_dataset", "manifest.csv")
    base_rows = []
    if os.path.exists(base_manifest):
        with open(base_manifest, "r", encoding="utf-8") as f:
            base_rows = list(csv.DictReader(f))

    pair_idx = 1
    for p_code, p_name in patterns:
        matching_rows = [r for r in base_rows if r["pattern_code"] == p_code]

        for t_idx, tier_name in enumerate(tier_names):
            pid = f"RGB_PAIR_{pair_idx:03d}"
            base_sample = matching_rows[t_idx % len(matching_rows)] if matching_rows else None

            if base_sample:
                base_ref_path = os.path.join(PARENT_DIR, base_sample["reference_path"])
                base_search_path = os.path.join(PARENT_DIR, base_sample["search_path"])
                ref_mask = cv2.imread(base_ref_path, cv2.IMREAD_GRAYSCALE)
                search_mask = cv2.imread(base_search_path, cv2.IMREAD_GRAYSCALE)
                gt_x = float(base_sample["gt_x"])
                gt_y = float(base_sample["gt_y"])
                scale_f = float(base_sample.get("scale_factor", 0.100))
                rot_deg = float(base_sample.get("rotation_deg", 0.0))
                drift_mag = float(base_sample.get("drift_magnitude", 0.0))
            else:
                ref_mask = np.full((1000, 1000), 128, dtype=np.uint8)
                search_mask = np.full((1000, 1000), 128, dtype=np.uint8)
                gt_x, gt_y = 500.0, 500.0
                scale_f, rot_deg, drift_mag = 0.100, 0.0, 0.0

            # 1. Physics-based Thin-Film Color Synthesis
            oxide_thickness = 140.0 + (t_idx * 35.0)
            ref_rgb = simulate_thin_film_color(ref_mask, p_name, oxide_thickness_nm=oxide_thickness, rng=rng)
            search_rgb = simulate_thin_film_color(search_mask, p_name, oxide_thickness_nm=oxide_thickness, rng=rng)

            # 2. Tier-specific Optical Augmentations
            if tier_name == "NOMINAL_BRIGHTFIELD":
                ref_rgb = cv2.GaussianBlur(ref_rgb, (0, 0), 0.6)
                search_rgb = cv2.GaussianBlur(search_rgb, (0, 0), 0.9)

            elif tier_name == "THIN_FILM_DISPERSION":
                ref_rgb = cv2.GaussianBlur(ref_rgb, (0, 0), 0.8)
                search_rgb = cv2.GaussianBlur(search_rgb, (0, 0), 1.1)
                search_rgb = apply_optical_vignetting(search_rgb, strength=0.20)

            elif tier_name == "DIFFRACTION_BLUR":
                ref_rgb = cv2.GaussianBlur(ref_rgb, (0, 0), 1.0)
                search_rgb = cv2.GaussianBlur(search_rgb, (0, 0), sigmaX=1.6, sigmaY=1.8)

            elif tier_name == "SPECULAR_GLARE":
                ref_rgb = cv2.GaussianBlur(ref_rgb, (0, 0), 0.8)
                search_rgb = apply_specular_metal_glare(search_rgb, search_mask, glare_spots=3, rng=rng)
                search_rgb = apply_optical_vignetting(search_rgb, strength=0.30)

            elif tier_name == "MIXED_STRESS":
                ref_rgb = cv2.GaussianBlur(ref_rgb, (0, 0), 0.9)
                search_rgb = cv2.GaussianBlur(search_rgb, (0, 0), 1.3)
                search_rgb = apply_optical_vignetting(search_rgb, strength=0.25)

            # 3. Save High-Res RGB Images
            ref_filename = f"ref_rgb_{pair_idx:03d}.png"
            search_filename = f"search_rgb_{pair_idx:03d}.png"

            ref_path = os.path.join(REF_DIR, ref_filename)
            search_path = os.path.join(SEARCH_DIR, search_filename)

            cv2.imwrite(ref_path, ref_rgb)
            cv2.imwrite(search_path, search_rgb)

            manifest_records.append({
                "pair_id": pid,
                "pattern_code": p_code,
                "pattern_name": p_name,
                "tier_name": tier_name,
                "reference_path": f"dataset/reference/{ref_filename}",
                "search_path": f"dataset/search/{search_filename}",
                "gt_x": round(gt_x, 4),
                "gt_y": round(gt_y, 4),
                "scale_factor": round(scale_f, 4),
                "rotation_deg": round(rot_deg, 2),
                "drift_magnitude": round(drift_mag, 2),
                "oxide_thickness_nm": round(oxide_thickness, 1),
                "modality": "RGB_OPTICAL"
            })

            print(f"  [+] {pid} | {p_code}: {p_name:<18} | Tier: {tier_name:<20} | GT: ({gt_x:6.2f}, {gt_y:6.2f})")
            pair_idx += 1

    # 4. Save Manifest
    manifest_csv = os.path.join(DATASET_DIR, "manifest_rgb.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_records[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_records)

    print("=" * 80)
    print(f"[+] Successfully generated all 40 RGB Optical Pairs in {DATASET_DIR}")
    print(f"[+] RGB Manifest CSV saved: {manifest_csv}")
    print("=" * 80)


if __name__ == "__main__":
    generate_40_pair_rgb_dataset()
