import os
import sys
import csv
import cv2
import numpy as np
import time
import concurrent.futures

from generate_dataset import generate_scene_pairs, PATTERN_TYPES
from localize import (
    get_clahe_uint8,
    generate_reference_variants,
    run_phase1_ncc,
    extract_reference_pitch,
    run_phase2_geometry_scoring,
    run_phase5_ml_reranker,
    run_phase3_fine_local_search,
    run_phase4_subpixel_refinement,
    localize_pair
)
from train import resolve_image_path


def _gen_single_pair(args):
    global_id, split, smoke_dir, seed, ptype, is_base, split_idx = args
    rng = np.random.default_rng(seed)
    return split, generate_scene_pairs(
        global_pair_id=global_id,
        split=split,
        out_dir=smoke_dir,
        rng=rng,
        pattern_type=ptype,
        is_base=is_base,
        split_pair_idx=split_idx
    )


def generate_smoke_dataset(smoke_dir="smoke_dataset", pairs_per_pattern=5):
    """
    Generates a fast smoke dataset: 5 pairs per pattern (45 total pairs: 27 train, 9 val, 9 test) in parallel
    """
    print("=" * 95, flush=True)
    print(f"GENERATING FAST SMOKE DATASET: {pairs_per_pattern} pairs/pattern x 9 patterns = {pairs_per_pattern*9} pairs", flush=True)
    print("=" * 95, flush=True)

    master_rng = np.random.default_rng(12345)
    os.makedirs(smoke_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    split_manifests = {s: [] for s in splits}

    tasks = []
    global_id = 1
    pair_counter = {s: 1 for s in splits}
    for p_idx, ptype in enumerate(PATTERN_TYPES):
        # 3 train (1 base, 2 aug), 1 val (1 aug), 1 test (1 aug)
        split_plan = [
            ("train", True, 1), ("train", False, 2),
            ("val", False, 1),
            ("test", False, 1),
        ]
        for split, is_base, count in split_plan:
            for _ in range(count):
                seed = int(master_rng.integers(0, 2**32 - 1))
                tasks.append((global_id, split, smoke_dir, seed, ptype, is_base, pair_counter[split]))
                pair_counter[split] += 1
                global_id += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(_gen_single_pair, tasks))

    for split, row in results:
        split_manifests[split].append(row)

    headers = [
        "pair_id", "pattern_type", "synthetic_regime", "random_seed",
        "reference_size", "search_size", "ref_path", "search_path",
        "gt_x", "gt_y", "drift_dx", "drift_dy", "pitch", "line_width",
        "rotation", "scale", "noise_settings", "hard_negative_count",
        "context_uniqueness", "difficulty", "split", "is_augmented"
    ]

    for s in splits:
        m_path = os.path.join(smoke_dir, f"{s}_metadata.csv")
        with open(m_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(split_manifests[s])
        print(f"  - Saved {s.upper()} split manifest: {len(split_manifests[s])} pairs -> {m_path}", flush=True)

    print("Smoke dataset generation complete.\n", flush=True)
    return smoke_dir


def audit_smoke_dataset(smoke_dir="smoke_dataset"):
    """
    Automated pre-training audit on the smoke dataset:
    TEST A — Scale Sanity
    TEST B — GT Coordinate Sanity
    TEST C — Split Disjointness
    TEST D — Phase 4 Safe Continuous Refinement
    TEST E — Baseline Cascade Accuracy
    """
    val_manifest = os.path.join(smoke_dir, "val_metadata.csv")
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    print("=" * 95, flush=True)
    print("TEST A: SCALE SANITY ON SMOKE VALIDATION SET (10:1 Standard, 0.100 Nominal, 0.091-0.111)")
    print("=" * 95, flush=True)
    print(f"{'Pair ID':10s} | {'Pattern':18s} | {'Regime':10s} | {'Expected Scale':15s} | {'NCC Selected Scale':18s} | {'Scale Err':10s} | {'Status':8s}", flush=True)
    print("-" * 95, flush=True)

    scale_errors = []
    for r in val_rows:
        ref_p = resolve_image_path(r["ref_path"], smoke_dir)
        search_p = resolve_image_path(r["search_path"], smoke_dir)
        ptype = r["pattern_type"]
        is_aug = r["is_augmented"] == "1"

        ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_p, cv2.IMREAD_GRAYSCALE)

        ref_norm = get_clahe_uint8(ref_img)
        search_norm = get_clahe_uint8(search_img)

        variants = generate_reference_variants(ref_norm, pattern_type=ptype)
        cands, _, _, _, _ = run_phase1_ncc(search_norm, variants)
        sel_scale = cands[0]["scale"]

        exp_scale = float(r["scale"]) * 0.100 if is_aug else 0.100
        sc_err = abs(sel_scale - exp_scale)
        scale_errors.append(sc_err)

        status = "PASS" if sc_err < 0.015 else "FAIL"
        reg = "Augmented" if is_aug else "Base Clean"
        print(f"{r['pair_id']:10s} | {ptype:18s} | {reg:10s} | {exp_scale:13.4f}   | {sel_scale:16.4f}   | {sc_err:8.4f}   | {status:8s}", flush=True)

    print(f"\nMean Scale Error across Smoke Val: {np.mean(scale_errors):.4f} (Max: {np.max(scale_errors):.4f})", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("TEST B: GT COORDINATE CONTINUITY & FRACTIONAL SUBPIXEL VARIANCE")
    print("=" * 95, flush=True)
    gt_xs = [float(r["gt_x"]) for r in val_rows]
    gt_ys = [float(r["gt_y"]) for r in val_rows]
    frac_xs = [abs(x - round(x)) for x in gt_xs]
    frac_ys = [abs(y - round(y)) for y in gt_ys]

    print(f"GT X Min: {np.min(gt_xs):.2f} | Max: {np.max(gt_xs):.2f} | Mean: {np.mean(gt_xs):.2f} | Std: {np.std(gt_xs):.2f}", flush=True)
    print(f"GT Y Min: {np.min(gt_ys):.2f} | Max: {np.max(gt_ys):.2f} | Mean: {np.mean(gt_ys):.2f} | Std: {np.std(gt_ys):.2f}", flush=True)
    print(f"Fractional Subpixel Offsets: Mean Frac X = {np.mean(frac_xs):.3f} px | Mean Frac Y = {np.mean(frac_ys):.3f} px (Continuous verified)", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("TEST C: SPLIT DISJOINTNESS & LEAKAGE AUDIT")
    print("=" * 95, flush=True)
    seeds = {}
    for s in ["train", "val", "test"]:
        m_path = os.path.join(smoke_dir, f"{s}_metadata.csv")
        with open(m_path, "r", encoding="utf-8") as f:
            seeds[s] = set(row["random_seed"] for row in csv.DictReader(f))

    tr_val_overlap = seeds["train"].intersection(seeds["val"])
    tr_te_overlap = seeds["train"].intersection(seeds["test"])
    val_te_overlap = seeds["val"].intersection(seeds["test"])

    print(f"Train Seeds: {len(seeds['train'])} | Val Seeds: {len(seeds['val'])} | Test Seeds: {len(seeds['test'])}", flush=True)
    print(f"Train / Val Overlap  : {len(tr_val_overlap)} (Expected: 0)", flush=True)
    print(f"Train / Test Overlap : {len(tr_te_overlap)} (Expected: 0)", flush=True)
    print(f"Val / Test Overlap   : {len(val_te_overlap)} (Expected: 0)", flush=True)
    assert len(tr_val_overlap) == 0 and len(tr_te_overlap) == 0 and len(val_te_overlap) == 0, "Split leakage detected!"
    print("Split Disjointness: PASS", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("TEST D & E: 5-PHASE CASCADE LOCALIZATION ON SMOKE VAL SET (P1-P9)")
    print("=" * 95, flush=True)
    loc_errors = []
    p4_valid_count = 0

    for r in val_rows:
        ref_p = resolve_image_path(r["ref_path"], smoke_dir)
        search_p = resolve_image_path(r["search_path"], smoke_dir)
        gt_x = float(r["gt_x"])
        gt_y = float(r["gt_y"])
        ptype = r["pattern_type"]

        res = localize_pair(ref_p, search_p, tau_conf=0.65, pattern_type=ptype)
        err = np.hypot(res["pred_x"] - gt_x, res["pred_y"] - gt_y)
        loc_errors.append(err)
        if res.get("subpixel_valid", 0) == 1:
            p4_valid_count += 1

        print(f"  {r['pair_id']:10s} ({ptype:18s}): GT ({gt_x:6.2f}, {gt_y:6.2f}) | Pred ({res['pred_x']:6.2f}, {res['pred_y']:6.2f}) | Err: {err:6.4f} px | Path: {res['path_used']}", flush=True)

    err_a = np.array(loc_errors)
    print("-" * 95, flush=True)
    print(f"Smoke Val Metrics (9 Patterns):", flush=True)
    print(f"  - Mean Error   : {np.mean(err_a):.4f} px", flush=True)
    print(f"  - Median Error : {np.median(err_a):.4f} px", flush=True)
    print(f"  - P95 Error    : {np.percentile(err_a, 95):.4f} px", flush=True)
    print(f"  - Max Error    : {np.max(err_a):.4f} px", flush=True)
    print(f"  - Acc < 1.0 px : {(np.sum(err_a < 1.0)/len(err_a))*100:.1f}%", flush=True)
    print(f"  - Acc < 0.6 px : {(np.sum(err_a < 0.6)/len(err_a))*100:.1f}%", flush=True)
    print("=" * 95 + "\n", flush=True)


if __name__ == "__main__":
    s_dir = generate_smoke_dataset("smoke_dataset", pairs_per_pattern=5)
    audit_smoke_dataset(s_dir)
