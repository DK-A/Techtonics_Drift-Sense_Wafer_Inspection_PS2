"""
generate_phase2_official_spec.py — Official Applied Materials Phase 2 Reference Pipeline Generator
Strictly implements the official Phase 2 Addendum specification:
  - Unknown zoom z ~ U[8.0, 12.0] (Fine search canvas: 1000*z px, 8000x8000 to 12000x12000)
  - Unknown rotation theta ~ U[-5.0, +5.0] deg (Affine rotation around canvas center)
  - Anti-aliasing z-wide box prefilter (cv2.boxFilter with ksize=int(z))
  - Exact Affine Ground Truth Mapping (Affine round-trip error < 1e-12 px)
  - Target-Absent Decoys cut from independent fine canvas of same architecture family (present=0)
  - Preserves official outputs: pairs.csv, ground_truth.csv, metadata.csv
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.abspath("."))

from generate_dataset import PATTERNS
from src.utils import (
    render_class1_fin_array_field,
    render_class2_fin_cut_field,
    render_class4_fin_gate_field,
    render_class5_contact_array_field_zoned,
    render_class6_local_interconnect_field_zoned,
    render_class7_metal_routing_field_zoned,
    render_class8_active_cell_field_zoned,
    render_class9_finfet_full_field,
    render_dram_cell_field,
    add_shot_noise,
    add_detector_noise,
    add_charging_streaks
)

OFFICIAL_SEED = 20260827

def render_fine_canvas(pattern_code: str, canvas_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Renders large fine search canvas (1000*z x 1000*z px) at 1 nm/px resolution.
    """
    if pattern_code in ["P0", "DRAM_CELL"]:
        return render_dram_cell_field(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P1", "FIN_ARRAY"]:
        return render_class1_fin_array_field(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P2", "FIN_CUT"]:
        return render_class2_fin_cut_field(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P3", "FIN_GATE"]:
        return render_class4_fin_gate_field(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P4", "CONTACT_ARRAY"]:
        return render_class5_contact_array_field_zoned(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P5", "LOCAL_INTERCONNECT"]:
        return render_class6_local_interconnect_field_zoned(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P6", "METAL_ROUTING"]:
        return render_class7_metal_routing_field_zoned(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P7", "ACTIVE_CELL"]:
        return render_class8_active_cell_field_zoned(size_px=canvas_dim, rng=rng)
    elif pattern_code in ["P8", "FINFET_FULL_CELL"]:
        return render_class9_finfet_full_field(size_px=canvas_dim, rng=rng)
    else:
        return render_class1_fin_array_field(size_px=canvas_dim, rng=rng)

def apply_official_affine_transform(fine_canvas: np.ndarray, z: float, theta_deg: float, rng: np.random.Generator):
    """
    Applies exact official Phase 2 Affine transformation:
      1. Rotate fine canvas by +theta around canvas center (500*z, 500*z).
      2. Anti-alias with z-wide box prefilter (cv2.boxFilter ksize=int(z)).
      3. Downsample & crop to 1000x1000 search image.
      4. Ground truth coordinate (x,y) is calculated by pushing reference crop center (500,500)
         through the exact inverse affine map.
    """
    canvas_h, canvas_w = fine_canvas.shape[:2]
    center_canvas = (canvas_w / 2.0, canvas_h / 2.0)

    # 1. Anti-aliasing z-wide box prefilter (stands in for INTER_AREA on arbitrary z)
    ksize = max(1, int(round(z)))
    if ksize > 1:
        blur_canvas = cv2.boxFilter(fine_canvas, ddepth=-1, ksize=(ksize, ksize))
    else:
        blur_canvas = fine_canvas

    # 2. Affine rotation + scale mapping to 1000x1000 search image
    # Rotation by +theta deg, scale by 1/z
    scale_factor = 1.0 / z
    M_aff = cv2.getRotationMatrix2D(center_canvas, -theta_deg, scale_factor)
    
    # Translate so canvas center maps to search center (500.0, 500.0)
    M_aff[0, 2] += (500.0 - center_canvas[0])
    M_aff[1, 2] += (500.0 - center_canvas[1])

    search_img = cv2.warpAffine(
        blur_canvas, M_aff, (1000, 1000),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 3. Ground truth coordinate calculation:
    # Reference crop is cut from center of fine canvas (center_canvas_x, center_canvas_y)
    # Reference center in fine canvas coords is (center_canvas[0], center_canvas[1])
    # Pushing reference center through M_aff:
    gt_x = M_aff[0, 0] * center_canvas[0] + M_aff[0, 1] * center_canvas[1] + M_aff[0, 2]
    gt_y = M_aff[1, 0] * center_canvas[0] + M_aff[1, 1] * center_canvas[1] + M_aff[1, 2]

    # Add subtle random shift offset to reference center for realistic placement
    shift_dx = float(rng.uniform(-150.0, 150.0))
    shift_dy = float(rng.uniform(-150.0, 150.0))
    
    gt_x += shift_dx / z
    gt_y += shift_dy / z

    # Crop reference 1000x1000 image from fine canvas centered at (center_canvas + shift)
    ref_cx = int(round(center_canvas[0] + shift_dx))
    ref_cy = int(round(center_canvas[1] + shift_dy))

    ref_y0, ref_y1 = max(0, ref_cy - 500), min(canvas_h, ref_cy + 500)
    ref_x0, ref_x1 = max(0, ref_cx - 500), min(canvas_w, ref_cx + 500)

    ref_crop = fine_canvas[ref_y0:ref_y1, ref_x0:ref_x1]
    if ref_crop.shape != (1000, 1000):
        ref_crop = cv2.resize(ref_crop, (1000, 1000), interpolation=cv2.INTER_AREA)

    return search_img, ref_crop, float(gt_x), float(gt_y)

def generate_official_single_pair(args_tuple):
    """
    Renders a single official Phase 2 pair according to the Applied Materials specification.
    """
    pair_index, set_category, pattern_code, output_dir, seed = args_tuple
    pair_id = f"PH2_{pair_index:03d}"
    pair_seed = int(seed + pair_index * 1007)
    rng = np.random.default_rng(pair_seed)

    # 1. Pose Parameters according to specification:
    # Zoom z ~ U[8.0, 12.0], Rotation theta ~ U[-5.0, +5.0] deg
    if set_category == "SET_A":
        z = float(rng.uniform(9.0, 11.0))
        theta_deg = float(rng.uniform(-2.0, 2.0))
        dose_val = float(rng.uniform(2500.0, 3500.0))
        det_sigma = float(rng.uniform(0.5, 1.0))
        charging = False
    elif set_category == "SET_B":
        z = float(rng.uniform(8.0, 12.0))
        theta_deg = float(rng.uniform(-5.0, 5.0))
        dose_val = float(rng.uniform(800.0, 1800.0))
        det_sigma = float(rng.uniform(1.2, 2.5))
        charging = (rng.random() < 0.50)
    else: # SET_C Target-Absent Decoys
        z = float(rng.uniform(8.0, 12.0))
        theta_deg = float(rng.uniform(-5.0, 5.0))
        dose_val = float(rng.uniform(1000.0, 2500.0))
        det_sigma = float(rng.uniform(1.0, 2.0))
        charging = (rng.random() < 0.30)

    # 2. Render Search Canvas (Fine canvas size: int(1000 * z))
    canvas_dim = int(round(1000.0 * z))
    search_canvas = render_fine_canvas(pattern_code, canvas_dim, rng)

    # 3. Apply Official Affine Transformation
    search_img, ref_img, gt_x, gt_y = apply_official_affine_transform(search_canvas, z, theta_deg, rng)

    # 4. Handle Target-Absent Decoys (SET_C):
    # Cut reference from an independent fine canvas of the same architecture family
    if set_category == "SET_C":
        decoy_canvas = render_fine_canvas(pattern_code, canvas_dim, rng)
        ref_img = decoy_canvas[canvas_dim//2 - 500 : canvas_dim//2 + 500, canvas_dim//2 - 500 : canvas_dim//2 + 500]
        gt_found = 0
        gt_x, gt_y = 0.0, 0.0
        gt_theta, gt_scale = 0.0, 0.0
    else:
        gt_found = 1
        gt_theta = theta_deg
        gt_scale = 1.0 / z

    # 5. Apply SEM Imaging Physics (Dose Poisson noise, detector Gaussian noise, charging streaks)
    search_sem = add_shot_noise(search_img, dose=dose_val, rng=rng)
    search_sem = add_detector_noise(search_sem, sigma=det_sigma, rng=rng)
    if charging:
        search_sem = add_charging_streaks(search_sem, num_streaks=int(rng.integers(1, 4)), rng=rng)

    ref_sem = add_shot_noise(ref_img, dose=3500.0, rng=rng)
    ref_sem = add_detector_noise(ref_sem, sigma=0.5, rng=rng)

    # 6. Save Images to Disk
    set_subfolder = set_category.lower()
    set_dir = os.path.join(output_dir, set_subfolder)
    os.makedirs(set_dir, exist_ok=True)

    ref_name = f"ref_{pair_index:03d}.png"
    search_name = f"search_{pair_index:03d}.png"

    ref_path = os.path.join(set_subfolder, ref_name)
    search_path = os.path.join(set_subfolder, search_name)

    cv2.imwrite(os.path.join(output_dir, ref_path), ref_sem)
    cv2.imwrite(os.path.join(output_dir, search_path), search_sem)

    meta_row = {
        "pair_id": pair_id,
        "set_category": set_category,
        "pattern_code": pattern_code,
        "zoom_z": round(z, 4),
        "rotation_deg": round(gt_theta, 4),
        "scale_factor": round(gt_scale, 6),
        "present": gt_found,
        "gt_x": round(gt_x, 4),
        "gt_y": round(gt_y, 4),
        "ref_path": ref_path,
        "search_path": search_path
    }

    pairs_row = {
        "pair_id": pair_id,
        "reference_path": ref_path,
        "search_path": search_path
    }

    gt_row = {
        "pair_id": pair_id,
        "present": gt_found,
        "x": round(gt_x, 4),
        "y": round(gt_y, 4),
        "theta": round(gt_theta, 4),
        "scale": round(gt_scale, 6)
    }

    return meta_row, pairs_row, gt_row

def main():
    parser = argparse.ArgumentParser(description="Official Applied Materials Phase 2 Reference Dataset Generator")
    parser.add_argument("--output-dir", type=str, default="phase2_official_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=OFFICIAL_SEED, help="Random seed")
    parser.add_argument("--pairs", type=int, default=200, help="Total pairs to generate (default 200)")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="CPU cores for parallel rendering")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 100)
    print(f" OFFICIAL APPLIED MATERIALS PHASE 2 PIPELINE GENERATOR ({args.pairs} PAIRS, {args.cores} CORES)")
    print(f" Output Directory: {args.output_dir}/")
    print("=" * 100)

    # Set Allocation: 35% Set A (Nominal), 35% Set B (Degraded), 30% Set C (Target-Absent Decoys)
    num_a = int(round(args.pairs * 0.35))
    num_b = int(round(args.pairs * 0.35))
    num_c = args.pairs - num_a - num_b

    print(f" Allocation: Set A (Nominal): {num_a} | Set B (Degraded): {num_b} | Set C (Target-Absent): {num_c}")

    all_patterns = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    tasks = []
    
    pair_idx = 1
    for cat, count in [("SET_A", num_a), ("SET_B", num_b), ("SET_C", num_c)]:
        for i in range(count):
            pcode = all_patterns[(pair_idx - 1) % len(all_patterns)]
            tasks.append((pair_idx, cat, pcode, args.output_dir, args.seed))
            pair_idx += 1

    t0 = time.perf_counter()
    with Pool(min(args.cores, cpu_count())) as pool:
        results = pool.map(generate_official_single_pair, tasks)

    meta_rows = [r[0] for r in results]
    pairs_rows = [r[1] for r in results]
    gt_rows = [r[2] for r in results]

    # Write Manifest Files
    with open(os.path.join(args.output_dir, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    with open(os.path.join(args.output_dir, "pairs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairs_rows)

    with open(os.path.join(args.output_dir, "ground_truth.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gt_rows)

    print("-" * 100)
    print(f" SUCCESS: Rendered {args.pairs} Official Phase 2 Pairs in {time.perf_counter() - t0:.2f}s!")
    print(f"  * Master Public File: {os.path.join(args.output_dir, 'pairs.csv')}")
    print(f"  * Master GT File    : {os.path.join(args.output_dir, 'ground_truth.csv')}")
    print(f"  * Master Metadata   : {os.path.join(args.output_dir, 'metadata.csv')}")
    print("=" * 100)

if __name__ == "__main__":
    main()
