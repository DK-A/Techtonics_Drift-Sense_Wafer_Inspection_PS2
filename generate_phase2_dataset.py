"""
generate_phase2_dataset.py — Ultra-Fast Direct-Rendering SEM & RGB Optical Generator
Generates Phase 2 submission dataset (220 Pairs Total) in ~15-20 seconds using CPU Multiprocessing!

Output Directory Structure:
  - phase2_dataset/set_a/ (70 Pairs: SEM Clean / Nominal, Grayscale)
  - phase2_dataset/set_b/ (70 Pairs: SEM Stress / Pose, Grayscale)
  - phase2_dataset/set_c/ (60 Pairs: Target-Absent Rejections, Grayscale)
  - phase2_dataset/set_d/ (20 Pairs: PHASE 2 RGB OPTICAL BONUS SET, 3-Channel Images)

Master manifests in phase2_dataset/ metadata.csv and pairs.csv combine all 220 pairs!
Globally unique sequential image filenames (ref_001.png ... ref_220.png) guarantee ZERO naming conflicts!
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

PHASE2_SEED = 20260830

def fast_render_pattern_field(pattern_name: str, rng: np.random.Generator, target_cx: float = 500.0, target_cy: float = 500.0, target_dim: float = 100.0) -> np.ndarray:
    """
    Renders 1000x1000 px pattern fields directly at target resolution.
    Uses native spatial zoned composition from src/utils.py to break periodic symmetry.
    """
    if pattern_name in ["DRAM_CELL", "DRAM_ARRAY", "P0"]:
        return render_class5_contact_array_field_zoned(size_px=1000, rng=rng)
    elif pattern_name == "FIN_ARRAY":
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
    elif pattern_name in ["FINFET_FULL_CELL", "P8"]:
        return render_class9_finfet_full_field(
            size_px=1000, rng=rng,
            gt_x=int(target_cx - target_dim / 2.0), gt_y=int(target_cy - target_dim / 2.0),
            gt_w=int(target_dim), gt_h=int(target_dim)
        )
    else:
        return render_class1_fin_array_field(size_px=1000, rng=rng)

def render_rgb_optical_analogue(gray_img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Converts 2D grayscale pattern into realistic 3-channel BGR Optical Microscope Analogue:
      - Thin-film interference oxide tinting (Si/SiO2 blue/cyan & Copper/Gold routing).
      - Optical chromatic aberration (slight red/blue channel offset).
      - Specular reflection & optical vignetting illumination gradient.
    """
    norm = gray_img.astype(np.float32) / 255.0

    b_chan = np.clip(norm * 210.0 + 35.0 + rng.uniform(-10, 10), 0, 255)
    g_chan = np.clip(norm * 180.0 + 25.0 + rng.uniform(-10, 10), 0, 255)
    r_chan = np.clip(norm * 140.0 + 15.0 + rng.uniform(-10, 10), 0, 255)

    dx = int(rng.choice([-1, 1]))
    dy = int(rng.choice([-1, 1]))
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    r_chan = cv2.warpAffine(r_chan, M_shift, (1000, 1000), borderMode=cv2.BORDER_REFLECT_101)

    rgb_img = cv2.merge([b_chan.astype(np.uint8), g_chan.astype(np.uint8), r_chan.astype(np.uint8)])

    y_idx, x_idx = np.indices((1000, 1000), dtype=np.float32)
    dist = np.hypot(x_idx - 500.0, y_idx - 500.0) / 707.1
    vignette = np.clip(1.05 - 0.25 * (dist ** 2), 0.70, 1.0)
    vignette_3ch = cv2.merge([vignette, vignette, vignette])

    return np.clip(rgb_img.astype(np.float32) * vignette_3ch, 0, 255).astype(np.uint8)

def generate_single_pair(args_tuple):
    t0_pair = time.perf_counter()
    pair_counter, sample_id, p_code, p_name, set_category, set_folder, n_idx, total_pairs, out_dir = args_tuple

    rng = np.random.default_rng(PHASE2_SEED + pair_counter * 10007)

    is_absent = (set_category == "SET_C")
    is_optical = (set_category == "SET_D")
    gt_found = 0 if is_absent else 1

    if set_category == "SET_A":
        scale = float(rng.uniform(1.0 / 11.0, 1.0 / 9.0))
        angle = float(rng.uniform(-2.0, 2.0))
        dose_val = float(rng.uniform(2500.0, 3500.0))
        det_sigma = float(rng.uniform(0.5, 1.0))
        charging = False
    elif set_category == "SET_B":
        scale = float(rng.uniform(1.0 / 12.0, 1.0 / 8.0))
        angle = float(rng.uniform(-5.0, 5.0))
        dose_val = float(rng.uniform(800.0, 1800.0))
        det_sigma = float(rng.uniform(1.2, 2.5))
        charging = (rng.random() < 0.50)
    elif set_category == "SET_D": # Phase 2 RGB Optical Bonus Set
        scale = float(rng.uniform(1.0 / 11.0, 1.0 / 9.0))
        angle = float(rng.uniform(-4.0, 4.0))
        dose_val = 3500.0
        det_sigma = 0.5
        charging = False
    else: # SET_C Target-Absent Decoys
        scale = float(rng.uniform(1.0 / 12.0, 1.0 / 8.0))
        angle = float(rng.uniform(-5.0, 5.0))
        dose_val = float(rng.uniform(1000.0, 2500.0))
        det_sigma = float(rng.uniform(1.0, 2.0))
        charging = (rng.random() < 0.30)

    target_dim = float(1000.0 * scale)
    margin = target_dim / 2.0 + 30.0

    if not is_absent:
        target_cx = float(rng.uniform(margin, 1000.0 - margin))
        target_cy = float(rng.uniform(margin, 1000.0 - margin))

        search_field = fast_render_pattern_field(p_name, rng, target_cx, target_cy, target_dim)

        M_crop = cv2.getRotationMatrix2D((target_cx, target_cy), -angle, 1000.0 / target_dim)
        M_crop[0, 2] += (1000.0 / 2.0 - target_cx)
        M_crop[1, 2] += (1000.0 / 2.0 - target_cy)
        ref_raw = cv2.warpAffine(search_field, M_crop, (1000, 1000), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

        search_aug = search_field.copy()
        rot_gt_x, rot_gt_y = target_cx, target_cy
    else:
        rot_gt_x, rot_gt_y = 0.0, 0.0
        target_cx, target_cy = 500.0, 500.0

        # Section 4: Decoy reference must come from SAME architecture family but carry different macro structure
        if p_code == "P0": # DRAM family decoy -> different DRAM mat preset
            decoy_p_name = "DRAM_CELL"
        else: # FinFET family decoy -> different FinFET layer (e.g. CONTACT_ARRAY or FIN_GATE)
            decoy_p_name = "CONTACT_ARRAY" if p_code != "P4" else "FIN_GATE"

        rng_decoy = np.random.default_rng(PHASE2_SEED + 7777 + pair_counter)
        ref_field = fast_render_pattern_field(decoy_p_name, rng_decoy, 500.0, 500.0, target_dim)
        M_crop = cv2.getRotationMatrix2D((500.0, 500.0), -angle, 1000.0 / target_dim)
        ref_raw = cv2.warpAffine(ref_field, M_crop, (1000, 1000), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

        rng_abs = np.random.default_rng(PHASE2_SEED + 9000 + pair_counter)
        search_aug = fast_render_pattern_field(p_name, rng_abs, 500.0, 500.0, target_dim)

    if is_optical:
        # Convert to 3-channel RGB Optical Microscope Analogue
        ref_raw = render_rgb_optical_analogue(ref_raw, rng)
        search_aug = render_rgb_optical_analogue(search_aug, rng)
    else:
        # SEM Noise & Degradation
        ref_raw = add_shot_noise(ref_raw, dose=3500.0, rng=rng)
        search_aug = add_shot_noise(search_aug, dose=dose_val, rng=rng)
        search_aug = add_detector_noise(search_aug, sigma=det_sigma, rng=rng)
        if charging:
            search_aug = add_charging_streaks(search_aug, rng=rng)

    ref_filename = f"ref_{pair_counter:03d}.png"
    search_filename = f"search_{pair_counter:03d}.png"

    set_ref_rel = os.path.join(out_dir, set_folder, "reference", ref_filename).replace("\\", "/")
    set_search_rel = os.path.join(out_dir, set_folder, "search", search_filename).replace("\\", "/")

    set_ref_abs = os.path.join(out_dir, set_folder, "reference", ref_filename)
    set_search_abs = os.path.join(out_dir, set_folder, "search", search_filename)

    os.makedirs(os.path.dirname(set_ref_abs), exist_ok=True)
    os.makedirs(os.path.dirname(set_search_abs), exist_ok=True)
    cv2.imwrite(set_ref_abs, ref_raw)
    cv2.imwrite(set_search_abs, search_aug)

    meta_entry = {
        "pair_id": sample_id,
        "set_category": set_category,
        "pattern_code": p_code,
        "pattern_name": p_name,
        "reference_path": set_ref_rel,
        "search_path": set_search_rel,
        "gt_found": gt_found,
        "gt_x": round(rot_gt_x, 3),
        "gt_y": round(rot_gt_y, 3),
        "gt_norm_x": round(rot_gt_x / 1000.0, 4),
        "gt_norm_y": round(rot_gt_y / 1000.0, 4),
        "scale_factor": round(scale, 4),
        "rotation_deg": round(angle, 2),
        "noise_dose": int(dose_val),
        "render_time_s": round(time.perf_counter() - t0_pair, 4)
    }

    pair_entry = {
        "pair_id": sample_id,
        "ref_path": set_ref_rel,
        "search_path": set_search_rel,
        "pattern_type": p_name
    }

    return meta_entry, pair_entry

PATTERNS_PHASE2 = [
    ("P0", "DRAM_CELL"),
    ("P1", "FIN_ARRAY"),
    ("P2", "FIN_CUT"),
    ("P3", "FIN_GATE"),
    ("P4", "CONTACT_ARRAY"),
    ("P5", "LOCAL_INTERCONNECT"),
    ("P6", "METAL_ROUTING"),
    ("P7", "ACTIVE_CELL"),
    ("P8", "FINFET_FULL_CELL")
]

def generate_phase2_dataset(out_dir="phase2_dataset", total_pairs=220, include_rgb=False, num_workers=1):
    os.makedirs(out_dir, exist_ok=True)

    set_dirs = {
        "SET_A": os.path.join(out_dir, "set_a"),
        "SET_B": os.path.join(out_dir, "set_b"),
        "SET_C": os.path.join(out_dir, "set_c"),
        "SET_D": os.path.join(out_dir, "set_d")
    }

    for s_key, s_path in set_dirs.items():
        os.makedirs(os.path.join(s_path, "reference"), exist_ok=True)
        os.makedirs(os.path.join(s_path, "search"), exist_ok=True)

    if total_pairs <= 20:
        if include_rgb:
            # Standalone RGB Optical Dataset: 20 RGB Optical Pairs (Set D: 15 Present, 5 Absent)
            set_alloc = {
                "P0": (0, 0, 0, 4), "P1": (0, 0, 0, 1), "P2": (0, 0, 0, 1), "P3": (0, 0, 0, 1),
                "P4": (0, 0, 0, 1), "P5": (0, 0, 0, 1), "P6": (0, 0, 0, 1), "P7": (0, 0, 0, 2), "P8": (0, 0, 0, 8)
            }
        else:
            # Core 20 SEM Grayscale Set: 4 DRAM Pairs (P0) + 16 FinFET Pairs (P1-P8)
            # 10 Set A, 6 Set B, 4 Set C = 20 Pairs Total
            set_alloc = {
                "P0": (2, 1, 1, 0), "P1": (1, 0, 0, 0), "P2": (1, 0, 0, 0), "P3": (1, 0, 0, 0),
                "P4": (0, 1, 0, 0), "P5": (0, 1, 0, 0), "P6": (1, 0, 1, 0), "P7": (1, 1, 0, 0), "P8": (3, 2, 2, 0)
            }
    else:
        if include_rgb:
            # Standalone RGB Master Benchmark (20 RGB Pairs)
            set_alloc = {
                "P0": (0, 0, 0, 4), "P1": (0, 0, 0, 1), "P2": (0, 0, 0, 1), "P3": (0, 0, 0, 1),
                "P4": (0, 0, 0, 1), "P5": (0, 0, 0, 1), "P6": (0, 0, 0, 1), "P7": (0, 0, 0, 2), "P8": (0, 0, 0, 8)
            }
        else:
            # Master SEM & RGB Optical Benchmark (220 Pairs Total: 70 Set A, 70 Set B, 60 Set C, 20 Set D)
            # 40 DRAM Pairs (P0) + 180 FinFET Pairs (P1-P8)
            set_alloc = {
                "P0": (14, 14, 12, 4), "P1": (4, 4, 3, 1), "P2": (4, 4, 3, 1), "P3": (4, 4, 3, 1),
                "P4": (4, 4, 3, 1), "P5": (4, 4, 3, 1), "P6": (5, 5, 3, 1), "P7": (5, 5, 3, 2), "P8": (26, 26, 27, 8)
            }

    sum_A = sum(v[0] for v in set_alloc.values())
    sum_B = sum(v[1] for v in set_alloc.values())
    sum_C = sum(v[2] for v in set_alloc.values())
    sum_D = sum(v[3] for v in set_alloc.values())
    total_alloc_pairs = sum_A + sum_B + sum_C + sum_D

    print("=" * 120)
    print(f" FAST PARALLEL BENCHMARK GENERATOR ({total_alloc_pairs} PAIRS, {cpu_count()} CPU CORES)")
    print(f" Output Directory: {out_dir}/ | set_a: {sum_A} | set_b: {sum_B} | set_c: {sum_C} | set_d (RGB Optical): {sum_D}")
    print("=" * 120)

    tasks = []
    pair_counter = 0

    for p_idx, (p_code, p_name) in enumerate(PATTERNS_PHASE2):
        nA, nB, nC, nD = set_alloc.get(p_code, (8, 8, 7, 2))
        total_p_pairs = nA + nB + nC + nD

        for i in range(1, total_p_pairs + 1):
            pair_counter += 1
            sample_id = f"PH2_{p_code}_{i:03d}"

            if i <= nA:
                set_category = "SET_A"
                set_folder = "set_a"
            elif i <= (nA + nB):
                set_category = "SET_B"
                set_folder = "set_b"
            elif i <= (nA + nB + nC):
                set_category = "SET_C"
                set_folder = "set_c"
            else:
                set_category = "SET_D"
                set_folder = "set_d"

            tasks.append((pair_counter, sample_id, p_code, p_name, set_category, set_folder, i, total_pairs, out_dir))

    t_start = time.perf_counter()

    master_meta_rows = []
    master_pairs_rows = []

    set_meta_rows = {"SET_A": [], "SET_B": [], "SET_C": [], "SET_D": []}
    set_pairs_rows = {"SET_A": [], "SET_B": [], "SET_C": [], "SET_D": []}

    pattern_times = {}

    from tqdm import tqdm
    if num_workers > 1:
        num_workers = min(num_workers, cpu_count())
        print(f"  Rendering {len(tasks)} dataset pairs in PARALLEL MULTIPROCESSING mode ({num_workers} CPU CORES)...", flush=True)
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(generate_single_pair, tasks), total=len(tasks), desc="Rendering Dataset Pairs", unit="pair"))
    else:
        print(f"  Rendering {len(tasks)} dataset pairs in NORMAL SEQUENTIAL execution mode...", flush=True)
        results = [generate_single_pair(t) for t in tqdm(tasks, desc="Rendering Dataset Pairs", unit="pair")]

    for meta_entry, pair_entry in results:
        p_code = meta_entry["pattern_code"]
        p_name = meta_entry["pattern_name"]
        if p_code not in pattern_times:
            pattern_times[p_code] = {"name": p_name, "times": []}
        pattern_times[p_code]["times"].append(meta_entry.get("render_time_s", 0.05))

        s_cat = meta_entry["set_category"]
        master_meta_rows.append(meta_entry)
        master_pairs_rows.append(pair_entry)
        set_meta_rows[s_cat].append(meta_entry)
        set_pairs_rows[s_cat].append(pair_entry)

    print("\n" + "=" * 120)
    print(" PATTERN-WISE RENDER LATENCY SUMMARY")
    print("=" * 120)
    print(f"{'Code':<6} | {'Pattern Name':<22} | {'Count':<6} | {'Total Time (s)':<15} | {'Avg Latency (s/pair)':<20}")
    print("-" * 120)
    for p_code, p_info in pattern_times.items():
        cnt = len(p_info["times"])
        tot_t = sum(p_info["times"])
        avg_t = tot_t / cnt if cnt > 0 else 0.0
        print(f"{p_code:<6} | {p_info['name']:<22} | {cnt:<6} | {tot_t:<15.2f} | {avg_t:<20.3f}")
    print("=" * 120)

    # Save Master Manifests
    master_meta_path = os.path.join(out_dir, "metadata.csv")
    with open(master_meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=master_meta_rows[0].keys())
        writer.writeheader()
        writer.writerows(master_meta_rows)

    master_pairs_path = os.path.join(out_dir, "pairs.csv")
    with open(master_pairs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=master_pairs_rows[0].keys())
        writer.writeheader()
        writer.writerows(master_pairs_rows)

    # Save Official Section 2.3 ground_truth.csv
    gt_rows = []
    for r in master_meta_rows:
        gt_rows.append({
            "pair_id": r["pair_id"],
            "present": r["gt_found"],
            "x": r["gt_x"],
            "y": r["gt_y"],
            "theta": r["rotation_deg"],
            "scale": r["scale_factor"]
        })

    master_gt_path = os.path.join(out_dir, "ground_truth.csv")
    with open(master_gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "present", "x", "y", "theta", "scale"])
        writer.writeheader()
        writer.writerows(gt_rows)

    # Save Set-Specific Manifests in set_a/, set_b/, set_c/, set_d/
    for s_cat, s_folder in [("SET_A", "set_a"), ("SET_B", "set_b"), ("SET_C", "set_c"), ("SET_D", "set_d")]:
        if set_meta_rows[s_cat]:
            s_meta_path = os.path.join(out_dir, s_folder, "metadata.csv")
            with open(s_meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=set_meta_rows[s_cat][0].keys())
                writer.writeheader()
                writer.writerows(set_meta_rows[s_cat])

            s_pairs_path = os.path.join(out_dir, s_folder, "pairs.csv")
            with open(s_pairs_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=set_pairs_rows[s_cat][0].keys())
                writer.writeheader()
                writer.writerows(set_pairs_rows[s_cat])

    t_total = time.perf_counter() - t_start
    print("-" * 120)
    print(f" SUCCESS: Generated Complete Benchmark ({len(tasks)} Pairs) in {t_total:.2f}s!")
    print(f"  * Master Manifest       : {master_meta_path}")
    print(f"  * Master Public File     : {master_pairs_path}")
    print(f"  * Set A (SEM Clean)      : {out_dir}/set_a/ ({sum_A} Pairs)")
    print(f"  * Set B (SEM Stress)     : {out_dir}/set_b/ ({sum_B} Pairs)")
    print(f"  * Set C (Target-Absent)  : {out_dir}/set_c/ ({sum_C} Pairs)")
    print(f"  * Set D (Phase 2 RGB)    : {out_dir}/set_d/ ({sum_D} Pairs RGB Optical Bonus)")
    print("=" * 120 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ultra-Fast Direct-Rendering Hackathon Dataset")
    parser.add_argument("--out_dir", type=str, default="phase2_dataset", help="Output directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Alias for --out_dir")
    parser.add_argument("--pairs", type=int, default=220, help="Total pairs to generate (220 for full set including Set D)")
    parser.add_argument("--cores", type=int, default=8, help="CPU cores to use for multiprocessing")
    parser.add_argument("--seed", type=int, default=PHASE2_SEED, help="Random seed")
    args = parser.parse_args()

    target_dir = args.output_dir if args.output_dir is not None else args.out_dir
    generate_phase2_dataset(out_dir=target_dir, total_pairs=args.pairs, num_workers=args.cores)
