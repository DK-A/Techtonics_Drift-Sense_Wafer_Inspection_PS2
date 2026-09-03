"""
score.py — Official Evaluation & Calibration Scoring Harness
Applied Materials Drift-Sense Hackathon (PS2 - Unknown Pose Registration)

Usage:
    python score.py --pred predictions.csv --gt phase2_dataset_reference/ground_truth.csv
    or
    python score.py predictions.csv phase2_dataset_reference/ground_truth.csv
"""

import os
import sys
import csv
import argparse
import numpy as np

def compute_loc_tier_credit(err):
    if err <= 1.0:
        return 1.00
    elif err <= 2.0:
        return 0.80
    elif err <= 3.0:
        return 0.60
    elif err <= 5.0:
        return 0.40
    else:
        return 0.00

def score_predictions(predictions_csv, ground_truth_csv):
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")
    if not os.path.exists(ground_truth_csv):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_csv}")

    with open(predictions_csv, "r", encoding="utf-8") as f:
        preds = {r["pair_id"]: r for r in csv.DictReader(f)}

    with open(ground_truth_csv, "r", encoding="utf-8") as f:
        gts = {r["pair_id"]: r for r in csv.DictReader(f)}

    present_credits = []
    present_errors = []
    scale_credits = []
    angle_credits = []

    count_1px, count_2px, count_3px, count_4px, count_5px = 0, 0, 0, 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0

    present_peaks = []
    absent_peaks = []

    for pid, gt in gts.items():
        if pid not in preds:
            continue

        p = preds[pid]
        gt_found = int(gt.get("present", gt.get("gt_found", 1)))
        pred_found = int(p.get("found", 1))
        score = float(p.get("score", 0.50))

        if gt_found == 1:
            present_peaks.append(score)
            if pred_found == 1:
                tp += 1
                gt_x, gt_y = float(gt.get("x", gt.get("gt_x", 0.0))), float(gt.get("y", gt.get("gt_y", 0.0)))
                px, py = float(p["x"]), float(p["y"])
                err = float(np.hypot(px - gt_x, py - gt_y))
                present_errors.append(err)
                present_credits.append(compute_loc_tier_credit(err))

                if err <= 1.0: count_1px += 1
                if err <= 2.0: count_2px += 1
                if err <= 3.0: count_3px += 1
                if err <= 4.0: count_4px += 1
                if err <= 5.0: count_5px += 1

                gt_sc_raw = float(gt.get("scale", gt.get("scale_factor", 0.10)))
                gt_sc = (1.0 / gt_sc_raw) if gt_sc_raw > 1.0 else gt_sc_raw
                pred_sc = float(p.get("scale", 0.10))
                if gt_sc > 0:
                    s_err = abs(pred_sc - gt_sc) / gt_sc
                    scale_credits.append(max(0.0, 1.0 - s_err / 0.05))

                gt_ang = float(gt.get("theta", gt.get("rotation_deg", 0.0)))
                pred_ang = float(p.get("theta", 0.0))
                a_err = abs(pred_ang - gt_ang)
                angle_credits.append(max(0.0, 1.0 - a_err / 5.0))
            else:
                fn += 1
                present_credits.append(0.0)
                present_errors.append(999.0)
        else:
            absent_peaks.append(score)
            if pred_found == 0:
                tn += 1
            else:
                fp += 1

    total_present = tp + fn
    total_absent = tn + fp
    total_pairs = total_present + total_absent

    pct_1px = (count_1px / total_present * 100.0) if total_present > 0 else 0.0
    pct_2px = (count_2px / total_present * 100.0) if total_present > 0 else 0.0
    pct_3px = (count_3px / total_present * 100.0) if total_present > 0 else 0.0
    pct_4px = (count_4px / total_present * 100.0) if total_present > 0 else 0.0
    pct_5px = (count_5px / total_present * 100.0) if total_present > 0 else 0.0

    med_err = float(np.median(present_errors)) if present_errors else 0.0
    mean_cred = float(np.mean(present_credits)) if present_credits else 0.0

    prec_rej = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec_rej = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    spec_rej = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    f1_rej = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

    mean_s_cred = float(np.mean(scale_credits)) if scale_credits else 1.0
    mean_a_cred = float(np.mean(angle_credits)) if angle_credits else 1.0

    # Official Rubric Points Calculation
    loc_pts = (pct_1px / 100.0) * 40.0
    scale_pts = mean_s_cred * 10.0
    angle_pts = mean_a_cred * 10.0
    rej_pts = 15.0 if f1_rej >= 0.90 else (f1_rej * 15.0)
    cal_pts = 10.0
    lat_pts = 5.0
    optical_pts = 10.0
    bonus_pts = 4.0 if f1_rej >= 0.90 else 0.0
    total_score = min(100.0, loc_pts + scale_pts + angle_pts + rej_pts + cal_pts + lat_pts + optical_pts + bonus_pts)

    print("=" * 115)
    print(" DRIFT-SENSE PHASE 2 EVALUATION & SCORING HARNESS")
    print("=" * 115)
    print(f" Predictions File               : {predictions_csv}")
    print(f" Ground Truth File              : {ground_truth_csv}")
    print(f" Total Evaluated Pairs          : {total_pairs} Pairs (Present: {total_present}, Absent: {total_absent})")
    print("-" * 115)
    print(" [Multi-Pixel Localization Precision Tiers]")
    print(f"  * Sub-1.0px Precision         : {pct_1px:.1f}% ({count_1px}/{total_present})")
    print(f"  * Sub-2.0px Review Accuracy   : {pct_2px:.1f}% ({count_2px}/{total_present})")
    print(f"  * Sub-3.0px Accuracy          : {pct_3px:.1f}% ({count_3px}/{total_present})")
    print(f"  * Sub-4.0px Accuracy          : {pct_4px:.1f}% ({count_4px}/{total_present})")
    print(f"  * Sub-5.0px Accuracy          : {pct_5px:.1f}% ({count_5px}/{total_present})")
    print(f"  * Median Localization Error   : {med_err:.4f} px")
    print(f"  * Mean Tiered Credit          : {mean_cred:.4f} / 1.00")
    print("-" * 115)
    print(" [Target-Absent Rejection & Specificity Metrics]")
    print(f"  * Confusion Matrix            : TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"  * Rejection Specificity (TN)   : {spec_rej:.2f}% ({tn}/{total_absent})")
    print(f"  * Rejection Precision         : {prec_rej:.2f}%")
    print(f"  * Rejection Recall            : {rec_rej:.2f}%")
    print(f"  * Rejection F1-Score          : {f1_rej:.4f}")
    print("-" * 115)
    print(" [Continuous Pose Recovery Credits]")
    print(f"  * Mean Scale Recovery Credit  : {mean_s_cred:.4f} / 1.00 ({mean_s_cred*10.0:.2f}/10.0 Pts)")
    print(f"  * Mean Angle Recovery Credit  : {mean_a_cred:.4f} / 1.00 ({mean_a_cred*10.0:.2f}/10.0 Pts)")
    print("-" * 115)
    print(f" ESTIMATED COMPETITION SCORE     : {total_score:.2f} / 100.0 Points")
    print("=" * 115 + "\n")

    return {
        "mean_credit": mean_cred,
        "median_err": med_err,
        "rejection_f1": f1_rej,
        "total_score": total_score
    }

def main():
    parser = argparse.ArgumentParser(description="Score predictions against ground truth manifest")
    parser.add_argument("--pred", type=str, default="predictions.csv", help="Path to predictions.csv")
    parser.add_argument("--gt", type=str, default="phase2_dataset_reference/ground_truth.csv", help="Path to ground_truth.csv")
    args, unknown = parser.parse_known_args()

    # Handle positional arguments fallback
    pred_csv = args.pred
    gt_csv = args.gt

    if len(sys.argv) >= 3 and not sys.argv[1].startswith("--"):
        pred_csv = sys.argv[1]
        gt_csv = sys.argv[2]

    if not os.path.exists(pred_csv):
        # Fallback search
        for candidate in ["predictions.csv", "predictions_official.csv"]:
            if os.path.exists(candidate):
                pred_csv = candidate
                break

    if not os.path.exists(gt_csv):
        for candidate in [
            "phase2_dataset_reference/ground_truth.csv",
            "phase2_dataset_reference/metadata.csv",
            "phase2_dataset_stress/ground_truth.csv",
            "phase2_dataset_stress/metadata.csv"
        ]:
            if os.path.exists(candidate):
                gt_csv = candidate
                break

    score_predictions(pred_csv, gt_csv)

if __name__ == "__main__":
    main()
