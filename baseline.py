"""
baseline.py — Naive Brute-Force Baseline Matcher (Section 5.1)
Evaluates coarse grid of scale z in [8, 12] (0.5x steps) and rotation theta in [-5, 5] (1.0 deg steps).
Predicts presence by thresholding correlation peak at 0.55.
"""

import os
import sys
import csv
import numpy as np
import cv2

def run_naive_baseline(reference_path, search_path):
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if ref is None or search is None:
        return {"found": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 10.0, "score": 0.0}

    best_score = -1.0
    best_loc = (0.0, 0.0)
    best_scale = 10.0
    best_theta = 0.0

    sh, sw = search.shape[:2]

    # Coarse grid of scale and rotation per Section 5.1
    scale_steps = np.arange(8.0, 12.5, 0.5)
    angle_steps = np.arange(-5.0, 6.0, 1.0)

    for z in scale_steps:
        # Scale template by 1/z factor relative to 1nm/px reference
        target_size = max(16, int(round(1000.0 / z)))
        tpl_base = cv2.resize(ref, (target_size, target_size), interpolation=cv2.INTER_AREA)

        for angle in angle_steps:
            if abs(angle) > 1e-3:
                M = cv2.getRotationMatrix2D((target_size / 2.0, target_size / 2.0), -angle, 1.0)
                tpl = cv2.warpAffine(tpl_base, M, (target_size, target_size), flags=cv2.INTER_LINEAR)
            else:
                tpl = tpl_base

            th, tw = tpl.shape[:2]
            if th >= sh or tw >= sw:
                continue

            res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_loc = (max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0)
                best_scale = z
                best_theta = angle

    # Predict presence by thresholding peak at 0.55 per Section 5.1
    found = 1 if best_score >= 0.55 else 0

    return {
        "found": found,
        "x": round(best_loc[0], 3) if found else 0.0,
        "y": round(best_loc[1], 3) if found else 0.0,
        "theta": best_theta if found else 0.0,
        "scale": best_scale if found else 10.0,
        "score": round(best_score, 6)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Naive Brute-Force Baseline Matcher")
    parser.add_argument("pairs", nargs="?", default="", help="Path to input pairs.csv")
    parser.add_argument("output", nargs="?", default="", help="Path to output baseline_predictions.csv")
    parser.add_argument("--dir", type=str, default="", help="Directory containing pairs.csv")
    args = parser.parse_args()

    if args.dir:
        pairs_csv = os.path.join(args.dir, "pairs.csv")
        out_csv = os.path.join(args.dir, "baseline_predictions.csv")
    elif args.pairs:
        pairs_csv = args.pairs
        out_csv = args.output or "baseline_predictions.csv"
    else:
        pairs_csv = "phase2_dataset/pairs.csv"
        out_csv = "baseline_predictions.csv"

    base_dir = os.path.dirname(os.path.abspath(pairs_csv))

    with open(pairs_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Running Naive Brute-Force Baseline Matcher on {len(rows)} pairs...")
    results = []

    for idx, r in enumerate(rows, 1):
        pair_id = r.get("pair_id", f"PAIR_{idx:03d}")
        ref_path = r.get("reference_path") or r.get("ref_path") or ""
        search_path = r.get("search_path") or ""
        
        ref_p = ref_path if os.path.exists(ref_path) else os.path.join(base_dir, ref_path)
        search_p = search_path if os.path.exists(search_path) else os.path.join(base_dir, search_path)

        res = run_naive_baseline(ref_p, search_p)
        res["pair_id"] = pair_id
        results.append(res)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "x", "y", "theta", "scale", "found", "score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Baseline Matcher finished: {len(results)} predictions saved to {out_csv}")

if __name__ == "__main__":
    main()
