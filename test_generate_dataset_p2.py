"""
test_generate_dataset_p2.py — 220-Pair Dataset Generator Using Official Applied Materials Phase 2 Pipeline
Renders 220 Pairs Total into test_phase2_dataset/:
  - Set A (Nominal SEM Grayscale): 80 Pairs
  - Set B (Degraded SEM Stress): 70 Pairs (Severity levels 1, 2, 3, 4)
  - Set C (Target-Absent Decoys): 50 Pairs (present=0)
  - Set D (Phase 2 RGB Optical Bonus): 20 Pairs (3-channel BGR images)

Uses CPU Multiprocessing for parallel rendering!
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count

# Insert reference generator path
ref_gen_dir = os.path.abspath("phase_2_reference_generator/generator")
sys.path.insert(0, ref_gen_dir)

from src.phase2_pipeline import (
    Phase2Params, generate_phase2_sample, to_optical_rgb,
    ZOOM_MIN, ZOOM_MAX, THETA_MIN, THETA_MAX
)

SEVERITY_LEVELS = {
    0: dict(dose_search=300.0, shear_amplitude_px=1.0, drift_jitter_px=0.30, detector_noise_sigma_search=4.0),
    1: dict(dose_search=150.0, shear_amplitude_px=1.5, drift_jitter_px=0.45, detector_noise_sigma_search=6.0, speckle_sigma=0.06),
    2: dict(dose_search=90.0, shear_amplitude_px=2.0, drift_jitter_px=0.65, detector_noise_sigma_search=8.0, charging_streak_prob=1.5, charging_streak_intensity=1.2, speckle_sigma=0.11, vignette_strength=0.10),
    3: dict(dose_search=55.0, shear_amplitude_px=2.5, drift_jitter_px=0.85, detector_noise_sigma_search=10.0, charging_streak_prob=3.0, charging_streak_intensity=2.0, speckle_sigma=0.19, salt_pepper_prob=0.005, astigmatism_ratio=1.35, vignette_strength=0.18, linewidth_bias_nm=-4.0),
    4: dict(dose_search=32.0, shear_amplitude_px=3.0, drift_jitter_px=1.05, detector_noise_sigma_search=14.0, charging_streak_prob=4.5, charging_streak_intensity=2.8, speckle_sigma=0.30, salt_pepper_prob=0.012, astigmatism_ratio=1.60, vignette_strength=0.30, gamma=1.25, barrel_distortion_k=0.005, linewidth_bias_nm=6.0),
}

ALL_ARCHITECTURES = [
    "dram_1x", "dram_dense", "dram_wide", "dram_loose", "dram_compact",
    "finfet_7nm", "finfet_10nm", "finfet_14nm", "finfet_22nm"
]

def render_single_pair(args_tuple):
    pair_idx, set_cat, arch, severity_lvl, output_dir, master_seed = args_tuple
    pair_id = f"p{pair_idx:03d}"
    seed = int(master_seed + pair_idx * 1007)
    rng = np.random.default_rng(seed)

    is_present = (set_cat != "SET_C")
    is_rgb = (set_cat == "SET_D")

    # Sample Zoom & Rotation angle
    if set_cat == "SET_A":
        z = float(rng.uniform(9.0, 11.0))
        theta = float(rng.uniform(-2.5, 2.5))
    elif set_cat in ["SET_B", "SET_C", "SET_D"]:
        z = float(rng.uniform(ZOOM_MIN, ZOOM_MAX))
        theta = float(rng.uniform(THETA_MIN, THETA_MAX))

    sev = SEVERITY_LEVELS[severity_lvl]
    params = Phase2Params(
        zoom=z,
        theta_deg=theta,
        present=is_present,
        **sev
    )

    result = generate_phase2_sample(arch, params, rng, min_margin=0.00)

    ref_img = result["reference_img"]
    search_img = result["search_img"]

    if is_rgb:
        ref_img = to_optical_rgb(ref_img, rng, blur_px=2.4)
        search_img = to_optical_rgb(search_img, rng, blur_px=1.6)

    set_folder = set_cat.lower()
    os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "search"), exist_ok=True)

    ref_rel = f"reference/{pair_id}.png"
    search_rel = f"search/{pair_id}.png"

    cv2.imwrite(os.path.join(output_dir, ref_rel), ref_img)
    cv2.imwrite(os.path.join(output_dir, search_rel), search_img)

    gt_info = result["gt"]
    gt_found = 1 if is_present else 0
    gt_x = float(gt_info["x"]) if is_present else 0.0
    gt_y = float(gt_info["y"]) if is_present else 0.0
    gt_theta = float(gt_info["theta"]) if is_present else 0.0
    gt_scale = float(gt_info["scale"]) if is_present else 0.0

    pairs_row = {
        "pair_id": pair_id,
        "search_path": search_rel,
        "reference_path": ref_rel
    }

    gt_row = {
        "pair_id": pair_id,
        "present": gt_found,
        "x": round(gt_x, 4),
        "y": round(gt_y, 4),
        "theta": round(gt_theta, 4),
        "scale": round(gt_scale, 6)
    }

    meta_row = {
        "pair_id": pair_id,
        "set_category": set_cat,
        "architecture": arch,
        "severity": severity_lvl,
        "zoom": round(z, 4),
        "theta": round(theta, 4),
        "present": gt_found,
        "search_path": search_rel,
        "reference_path": ref_rel
    }

    return pairs_row, gt_row, meta_row

def main():
    parser = argparse.ArgumentParser(description="Official Applied Materials Phase 2 Dataset Generator")
    parser.add_argument("--output-dir", "--output", "--out_dir", type=str, default="test_phase2_dataset", help="Output directory")
    parser.add_argument("--pairs", type=int, default=220, help="Total pairs to generate (e.g. 220 or 20)")
    parser.add_argument("--seed", type=int, default=20260827, help="Random seed")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="CPU cores for parallel rendering")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 100)
    print(f" OFFICIAL APPLIED MATERIALS PHASE 2 PIPELINE GENERATOR ({args.pairs} PAIRS, {args.cores} CORES)")
    print(f" Target Folder: {args.output_dir}/")
    print("=" * 100)

    # Dynamic Pair Allocation:
    total_pairs = args.pairs
    if total_pairs == 220:
        nA, nB, nC, nD = 70, 70, 60, 20
    elif total_pairs == 20:
        nA, nB, nC, nD = 7, 7, 4, 2
    else:
        nA = max(1, int(round(total_pairs * 0.32)))
        nB = max(1, int(round(total_pairs * 0.32)))
        nC = max(1, int(round(total_pairs * 0.27)))
        nD = max(0, total_pairs - (nA + nB + nC))

    tasks = []
    pair_idx = 1

    # Set A (Nominal)
    for i in range(nA):
        arch = ALL_ARCHITECTURES[i % len(ALL_ARCHITECTURES)]
        tasks.append((pair_idx, "SET_A", arch, 0, args.output_dir, args.seed))
        pair_idx += 1

    # Set B (Degraded, Severity 1, 2, 3, 4)
    for i in range(nB):
        arch = ALL_ARCHITECTURES[i % len(ALL_ARCHITECTURES)]
        sev = (i % 4) + 1 # Severity 1, 2, 3, 4
        tasks.append((pair_idx, "SET_B", arch, sev, args.output_dir, args.seed))
        pair_idx += 1

    # Set C (Target-Absent Decoys)
    for i in range(nC):
        arch = ALL_ARCHITECTURES[i % len(ALL_ARCHITECTURES)]
        tasks.append((pair_idx, "SET_C", arch, 0, args.output_dir, args.seed))
        pair_idx += 1

    # Set D (Phase 2 RGB Optical Bonus)
    for i in range(nD):
        arch = ALL_ARCHITECTURES[i % len(ALL_ARCHITECTURES)]
        tasks.append((pair_idx, "SET_D", arch, 0, args.output_dir, args.seed))
        pair_idx += 1

    t0 = time.perf_counter()
    with Pool(min(args.cores, cpu_count())) as pool:
        results = pool.map(render_single_pair, tasks)

    pairs_rows = [r[0] for r in results]
    gt_rows = [r[1] for r in results]
    meta_rows = [r[2] for r in results]

    with open(os.path.join(args.output_dir, "pairs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairs_rows)

    with open(os.path.join(args.output_dir, "ground_truth.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gt_rows)

    with open(os.path.join(args.output_dir, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    print("-" * 100)
    print(f" SUCCESS: Rendered 220 Official Phase 2 Pairs in {time.perf_counter() - t0:.2f}s!")
    print(f"  * Public File: {os.path.join(args.output_dir, 'pairs.csv')}")
    print(f"  * GT File    : {os.path.join(args.output_dir, 'ground_truth.csv')}")
    print(f"  * Metadata   : {os.path.join(args.output_dir, 'metadata.csv')}")
    print("=" * 100)

if __name__ == "__main__":
    main()
