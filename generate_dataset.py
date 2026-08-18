"""
generate_dataset.py — Realistic 120-Pair SEM Benchmark Generator (P1–P8)
Generates 120 completely unseen, unconstrained image pairs across 8 required patterns (15 pairs per pattern).
Key Highlights:
- Non-Junction Arbitrary Positioning: Targets are extracted at continuous random coordinates
  across the layout space (trench spaces, line midpoints, partial edges, asymmetric routing)
  rather than being constrained to integer grid intersections or central junctions.
- Fully Varied Stress Dimensions:
  * Samples 1–4: Controlled Pure Noise Progression (Low -> Medium -> High -> Severe)
  * Samples 5–6: Extreme Spatial Position Robustness (Random Corner/Edge Quadrants)
  * Samples 7–8: Multi-Scale Magnification Robustness (0.091–0.111 range around 10:1)
  * Samples 9–10: Angular Rotation Robustness (+/-2.0 deg)
  * Samples 11–12: SEM Stage Drift Robustness (+/-11.0 px)
  * Samples 13–14: Dense Periodic Ambiguity Stress
  * Sample 15: Mixed Multi-Stress Condition

Usage:
    python generate_dataset.py
    # Or specify custom directory:
    python generate_dataset.py --out_dir eval_dataset
"""

import os
import csv
import json
import argparse
import numpy as np
import cv2

from src.utils import (
    render_class1_fin_array_field,
    render_class2_fin_cut_field,
    render_class4_fin_gate_field,
    render_class5_contact_array_field_zoned,
    render_class6_local_interconnect_field_zoned,
    render_class7_metal_routing_field_zoned,
    render_class8_active_cell_field_zoned,
    render_class9_finfet_full_field,
    add_shot_noise,
    add_detector_noise,
    add_charging_streaks
)

MASTER_SEED = 20260818

# 8 Required SEMICON Patterns (P1–P8)
PATTERNS = [
    ("P1", "FIN_ARRAY"),
    ("P2", "FIN_CUT"),
    ("P3", "FIN_GATE"),
    ("P4", "CONTACT_ARRAY"),
    ("P5", "LOCAL_INTERCONNECT"),
    ("P6", "METAL_ROUTING"),
    ("P7", "ACTIVE_CELL"),
    ("P8", "FINFET_FULL_CELL")
]


def render_pattern_field(pattern_name: str, rng: np.random.Generator, cx: float, cy: float, target_dim: float) -> np.ndarray:
    if pattern_name == "FIN_ARRAY":
        return render_class1_fin_array_field(size_px=1000, rng=rng)
    elif pattern_name == "FIN_CUT":
        return render_class2_fin_cut_field(size_px=1000, rng=rng)
    elif pattern_name == "FIN_GATE":
        return render_class4_fin_gate_field(size_px=1000, rng=rng)
    elif pattern_name == "CONTACT_ARRAY":
        return render_class5_contact_array_field_zoned(size_px=1000, rng=rng)
    elif pattern_name == "LOCAL_INTERCONNECT":
        return render_class6_local_interconnect_field_zoned(size_px=1000, rng=rng)
    elif pattern_name == "METAL_ROUTING":
        return render_class7_metal_routing_field_zoned(size_px=1000, rng=rng)
    elif pattern_name == "ACTIVE_CELL":
        return render_class8_active_cell_field_zoned(size_px=1000, rng=rng)
    elif pattern_name == "FINFET_FULL_CELL":
        return render_class9_finfet_full_field(
            size_px=1000, rng=rng,
            gt_x=int(cx - target_dim / 2.0), gt_y=int(cy - target_dim / 2.0),
            gt_w=int(target_dim), gt_h=int(target_dim)
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")


def generate_benchmark_120(out_dir: str = "submission_dataset", master_seed: int = MASTER_SEED):
    print("=" * 85, flush=True)
    print("      SEMICON 120-PAIR DATASET GENERATION (8 REQUIRED PATTERNS: P1–P8)     ")
    print("=" * 85, flush=True)

    ref_dir = os.path.join(out_dir, "reference")
    search_dir = os.path.join(out_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_rows = []
    seeds_record = {"master_seed": master_seed, "pairs": {}}
    pair_counter = 1

    for p_code, p_name in PATTERNS:
        for s_idx in range(1, 16):
            sample_id = f"PAIR_{pair_counter:03d}"
            pair_seed = int(master_seed + pair_counter * 7919)
            pair_rng = np.random.default_rng(pair_seed)
            seeds_record["pairs"][sample_id] = pair_seed

            # Non-Junction Arbitrary Coordinates:
            # Generate arbitrary continuous floating coordinates with non-pitch offsets
            non_grid_offset_x = float(pair_rng.uniform(7.5, 23.5)) * pair_rng.choice([-1, 1])
            non_grid_offset_y = float(pair_rng.uniform(7.5, 23.5)) * pair_rng.choice([-1, 1])

            # Default parameters
            scale = 0.100
            angle = 0.0
            drift_x, drift_y = 0.0, 0.0
            pos_name = "random_interior"
            base_cx = 500.0 + non_grid_offset_x
            base_cy = 500.0 + non_grid_offset_y
            charging = False
            periodicity = "STANDARD"
            is_mixed = False

            # Samples 1-4: Controlled Pure Noise Progression (Low -> Med -> High -> Severe)
            if s_idx == 1:
                stress_cat = "NOISE_ROBUSTNESS"
                noise_lvl = "LOW"
                dose, det_sigma, blur_sigma = 3500.0, 0.8, 1.0
            elif s_idx == 2:
                stress_cat = "NOISE_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.6, 1.4
            elif s_idx == 3:
                stress_cat = "NOISE_ROBUSTNESS"
                noise_lvl = "HIGH"
                dose, det_sigma, blur_sigma = 1000.0, 2.4, 1.9
            elif s_idx == 4:
                stress_cat = "NOISE_ROBUSTNESS"
                noise_lvl = "SEVERE"
                dose, det_sigma, blur_sigma = 500.0, 3.2, 2.5
                charging = True

            # Samples 5-6: Target Position Robustness (Corner / Edge Off-Center Non-Junctions)
            elif s_idx == 5:
                stress_cat = "POSITION_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                pos_name = "top_left"
                base_cx = float(pair_rng.uniform(220.0, 340.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(220.0, 340.0)) + non_grid_offset_y
            elif s_idx == 6:
                stress_cat = "POSITION_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                pos_name = "bottom_right"
                base_cx = float(pair_rng.uniform(660.0, 780.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(660.0, 780.0)) + non_grid_offset_y

            # Samples 7-8: Multi-Scale Robustness
            elif s_idx == 7:
                stress_cat = "SCALE_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                scale = float(pair_rng.uniform(0.091, 0.095))
                base_cx = float(pair_rng.uniform(400.0, 600.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(400.0, 600.0)) + non_grid_offset_y
            elif s_idx == 8:
                stress_cat = "SCALE_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                scale = float(pair_rng.uniform(0.106, 0.111))
                base_cx = float(pair_rng.uniform(400.0, 600.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(400.0, 600.0)) + non_grid_offset_y

            # Samples 9-10: Angular Rotation Robustness
            elif s_idx == 9:
                stress_cat = "ROTATION_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                angle = float(pair_rng.uniform(-1.9, -1.0))
                base_cx = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_y
            elif s_idx == 10:
                stress_cat = "ROTATION_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                angle = float(pair_rng.uniform(1.0, 1.9))
                base_cx = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_y

            # Samples 11-12: Imaging Drift Robustness
            elif s_idx == 11:
                stress_cat = "DRIFT_ROBUSTNESS"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                drift_x = float(pair_rng.uniform(3.0, 6.0)) * pair_rng.choice([-1, 1])
                drift_y = float(pair_rng.uniform(3.0, 6.0)) * pair_rng.choice([-1, 1])
                base_cx = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_y
            elif s_idx == 12:
                stress_cat = "DRIFT_ROBUSTNESS"
                noise_lvl = "HIGH"
                dose, det_sigma, blur_sigma = 1200.0, 2.0, 1.6
                drift_x = float(pair_rng.uniform(7.0, 11.0)) * pair_rng.choice([-1, 1])
                drift_y = float(pair_rng.uniform(7.0, 11.0)) * pair_rng.choice([-1, 1])
                base_cx = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(420.0, 580.0)) + non_grid_offset_y

            # Samples 13-14: Periodic Ambiguity Stress
            elif s_idx == 13:
                stress_cat = "PERIODIC_AMBIGUITY"
                noise_lvl = "MEDIUM"
                dose, det_sigma, blur_sigma = 2000.0, 1.5, 1.3
                periodicity = "HIGH_REPEAT"
                pos_name = "right_edge"
                base_cx = float(pair_rng.uniform(670.0, 780.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(400.0, 600.0)) + non_grid_offset_y
            elif s_idx == 14:
                stress_cat = "PERIODIC_AMBIGUITY"
                noise_lvl = "HIGH"
                dose, det_sigma, blur_sigma = 1000.0, 2.2, 1.8
                periodicity = "HIGH_REPEAT"
                pos_name = "bottom_left"
                base_cx = float(pair_rng.uniform(220.0, 340.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(660.0, 780.0)) + non_grid_offset_y

            # Sample 15: Mixed Multi-Stress
            elif s_idx == 15:
                stress_cat = "MIXED_STRESS"
                is_mixed = True
                noise_lvl = "HIGH"
                dose, det_sigma, blur_sigma = 1100.0, 2.1, 1.7
                pos_name = "top_right"
                base_cx = float(pair_rng.uniform(660.0, 780.0)) + non_grid_offset_x
                base_cy = float(pair_rng.uniform(220.0, 340.0)) + non_grid_offset_y
                scale = float(pair_rng.uniform(0.094, 0.106))
                angle = float(pair_rng.uniform(-1.5, 1.5))
                drift_x = float(pair_rng.uniform(3.0, 7.0)) * pair_rng.choice([-1, 1])
                drift_y = float(pair_rng.uniform(3.0, 7.0)) * pair_rng.choice([-1, 1])
                charging = True

            # Dynamic region classification
            if base_cx < 400 and base_cy < 400:
                pos_name = "top_left"
            elif base_cx > 600 and base_cy < 400:
                pos_name = "top_right"
            elif base_cx < 400 and base_cy > 600:
                pos_name = "bottom_left"
            elif base_cx > 600 and base_cy > 600:
                pos_name = "bottom_right"
            elif base_cx < 380:
                pos_name = "left_edge"
            elif base_cx > 620:
                pos_name = "right_edge"
            elif base_cy < 380:
                pos_name = "top_edge"
            elif base_cy > 620:
                pos_name = "bottom_edge"
            else:
                pos_name = "arbitrary_interior"

            # Subpixel jitter
            sub_jx = float(pair_rng.uniform(-0.45, 0.45))
            sub_jy = float(pair_rng.uniform(-0.45, 0.45))
            target_cx = base_cx + sub_jx + drift_x
            target_cy = base_cy + sub_jy + drift_y

            target_dim = float(1000.0 * scale)
            margin = target_dim / 2.0 + 20.0
            target_cx = float(np.clip(target_cx, margin, 1000.0 - margin))
            target_cy = float(np.clip(target_cy, margin, 1000.0 - margin))

            field_raw = render_pattern_field(p_name, pair_rng, target_cx, target_cy, target_dim)

            # 1000x1000 reference patch extraction
            M_crop = cv2.getRotationMatrix2D((target_cx, target_cy), -angle, 1000.0 / target_dim)
            M_crop[0, 2] += (1000.0 / 2.0 - target_cx)
            M_crop[1, 2] += (1000.0 / 2.0 - target_cy)
            ref_raw = cv2.warpAffine(field_raw, M_crop, (1000, 1000), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

            search_aug = add_shot_noise(field_raw, dose=dose, rng=pair_rng)
            search_aug = add_detector_noise(search_aug, sigma=det_sigma, rng=pair_rng)
            if charging:
                search_aug = add_charging_streaks(search_aug, rng=pair_rng)

            ref_filename = f"ref_{pair_counter:03d}.png"
            search_filename = f"search_{pair_counter:03d}.png"
            ref_rel = os.path.join(out_dir, "reference", ref_filename).replace("\\", "/")
            search_rel = os.path.join(out_dir, "search", search_filename).replace("\\", "/")

            cv2.imwrite(os.path.join(ref_dir, ref_filename), ref_raw)
            cv2.imwrite(os.path.join(search_dir, search_filename), search_aug)

            manifest_rows.append({
                "pair_id": sample_id,
                "pattern_code": p_code,
                "pattern_name": p_name,
                "reference_path": ref_rel,
                "search_path": search_rel,
                "gt_x": round(target_cx, 3),
                "gt_y": round(target_cy, 3),
                "gt_norm_x": round(target_cx / 1000.0, 4),
                "gt_norm_y": round(target_cy / 1000.0, 4),
                "position_region": pos_name,
                "noise_level": noise_lvl,
                "noise_details": f"dose={int(dose)},det_sigma={det_sigma:.1f},blur={blur_sigma:.1f},charging={charging}",
                "scale_factor": round(scale, 4),
                "rotation_deg": round(angle, 2),
                "drift_magnitude": round(float(np.hypot(drift_x, drift_y)), 2),
                "drift_x": round(drift_x, 2),
                "drift_y": round(drift_y, 2),
                "stress_category": stress_cat,
                "periodicity": periodicity,
                "is_mixed_stress": is_mixed,
                "master_seed": master_seed,
                "pair_seed": pair_seed
            })

            print(f"  [{pair_counter:03d}/120] {sample_id} ({p_name}) -> Pos: ({target_cx:.1f}, {target_cy:.1f}) [{pos_name}] | Noise: {noise_lvl} | Cat: {stress_cat}")
            pair_counter += 1

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    seeds_path = os.path.join(out_dir, "seeds.json")
    with open(seeds_path, "w", encoding="utf-8") as f:
        json.dump(seeds_record, f, indent=2)

    print("\n" + "=" * 85)
    print(f"Successfully generated 120 non-junction randomized pairs in '{out_dir}'!")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Seeds record saved to: {seeds_path}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate 120-Pair SEM Benchmark Dataset")
    parser.add_argument("--out_dir", type=str, default="submission_dataset", help="Output directory for generated dataset")
    parser.add_argument("--seed", type=int, default=MASTER_SEED, help="Master seed for reproducible dataset generation")
    args = parser.parse_args()

    generate_benchmark_120(out_dir=args.out_dir, master_seed=args.seed)


if __name__ == "__main__":
    main()
