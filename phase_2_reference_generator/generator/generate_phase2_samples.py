#!/usr/bin/env python3
"""
Build the 20-pair Drift-Sense Phase 2 sample set.

Pose values are hand-specified rather than sampled, so the 20 pairs
provably span the disclosed ranges: zoom hits both endpoints (8.0 and
12.0), theta hits both endpoints (-5.0 and +5.0) and zero, and 4 of 20
pairs contain no reference at all.

    python generate_phase2_samples.py --output-dir ./phase2_samples

Emits:
    pairs.csv          -- give this to participants (no ground truth)
    ground_truth.csv   -- withhold; scoring key
    manifest_jury.csv  -- every generation parameter, jury only
    reference/*.png, search/*.png
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from src.phase2_pipeline import (Phase2Params, generate_phase2_sample,
                                 to_optical_rgb, ZOOM_MIN, ZOOM_MAX,
                                 THETA_MIN, THETA_MAX)

# Degradation bundles for Set B. Level 0 is the Set A / nominal condition.
# Kept out of the participant-facing docs on purpose: the categories are
# disclosed, the parameters are not.
SEVERITY = {
    0: dict(dose_search=300.0, shear_amplitude_px=1.0, drift_jitter_px=0.30,
            detector_noise_sigma_search=4.0),
    1: dict(dose_search=150.0, shear_amplitude_px=1.5, drift_jitter_px=0.45,
            detector_noise_sigma_search=6.0, speckle_sigma=0.06),
    2: dict(dose_search=90.0, shear_amplitude_px=2.0, drift_jitter_px=0.65,
            detector_noise_sigma_search=8.0, charging_streak_prob=1.5,
            charging_streak_intensity=1.2, speckle_sigma=0.11,
            vignette_strength=0.10),
    3: dict(dose_search=55.0, shear_amplitude_px=2.5, drift_jitter_px=0.85,
            detector_noise_sigma_search=10.0, charging_streak_prob=3.0,
            charging_streak_intensity=2.0, speckle_sigma=0.19,
            salt_pepper_prob=0.005, astigmatism_ratio=1.35,
            vignette_strength=0.18, linewidth_bias_nm=-4.0),
    4: dict(dose_search=32.0, shear_amplitude_px=3.0, drift_jitter_px=1.05,
            detector_noise_sigma_search=14.0, charging_streak_prob=4.5,
            charging_streak_intensity=2.8, speckle_sigma=0.30,
            salt_pepper_prob=0.012, astigmatism_ratio=1.60,
            vignette_strength=0.30, gamma=1.25, barrel_distortion_k=0.005,
            linewidth_bias_nm=6.0),
}

# id, set, architecture, zoom, theta, present, severity
PLAN = [
    # --- Set A: nominal pose, reference present (8) ---
    ("p001", "A", "dram_1x",     8.00,  0.00, True,  0),
    ("p002", "A", "dram_1x",    10.00, -1.20, True,  0),
    ("p003", "A", "finfet_10nm", 12.00,  0.00, True,  0),
    ("p004", "A", "finfet_10nm",  9.15,  4.60, True,  0),
    ("p005", "A", "dram_dense",  11.30, -4.90, True,  0),
    ("p006", "A", "finfet_14nm",  8.60,  2.30, True,  0),
    ("p007", "A", "dram_wide",   10.75, -3.10, True,  0),
    ("p008", "A", "finfet_22nm", 11.90,  1.40, True,  0),
    # --- Set B: degraded, reference present, 4 severity levels (6) ---
    ("p009", "B", "dram_1x",      9.40,  3.70, True,  1),
    ("p010", "B", "finfet_10nm", 10.60, -2.60, True,  2),
    ("p011", "B", "dram_loose",   8.25,  4.90, True,  3),
    ("p012", "B", "finfet_22nm", 11.75, -4.40, True,  4),
    ("p013", "B", "dram_compact",12.00,  0.60, True,  2),
    ("p014", "B", "finfet_7nm",   8.00, -0.90, True,  3),
    # --- Set C: no true instance (4) ---
    ("p015", "C", "dram_1x",      9.80,  2.10, False, 0),
    ("p016", "C", "finfet_10nm", 11.20, -3.50, False, 1),
    ("p017", "C", "dram_dense",   8.45,  4.20, False, 2),
    ("p018", "C", "finfet_14nm", 12.00, -1.70, False, 0),
    # --- Set D: optical RGB analogue, bonus only (2) ---
    ("p019", "D", "dram_1x",     10.30,  1.90, True,  0),
    ("p020", "D", "finfet_10nm",  9.05, -4.00, True,  1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./phase2_samples")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    ref_dir = os.path.join(args.output_dir, "reference")
    srch_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(srch_dir, exist_ok=True)

    pairs_rows, gt_rows, jury_rows = [], [], []

    for idx, (pid, subset, arch, zoom, theta, present, sev) in enumerate(PLAN):
        assert ZOOM_MIN <= zoom <= ZOOM_MAX, pid
        assert THETA_MIN <= theta <= THETA_MAX, pid
        t0 = time.time()
        rng = np.random.default_rng(args.seed + idx * 7919)
        params = Phase2Params(zoom=zoom, theta_deg=theta, present=present,
                              boundary_bias=0.70, **SEVERITY[sev])
        s = generate_phase2_sample(arch, params, rng)

        ref, srch, gt, v = s["reference_img"], s["search_img"], s["gt"], s["verify"]

        if subset == "D":
            ref_out = to_optical_rgb(ref, rng, blur_px=2.4)
            srch_out = to_optical_rgb(srch, rng, blur_px=1.6)
            channels = 3
        else:
            ref_out, srch_out = ref, srch
            channels = 1

        rp = os.path.join("reference", f"{pid}.png")
        sp = os.path.join("search", f"{pid}.png")
        cv2.imwrite(os.path.join(args.output_dir, rp), ref_out)
        cv2.imwrite(os.path.join(args.output_dir, sp), srch_out)

        pairs_rows.append({"pair_id": pid, "search_path": sp, "reference_path": rp})
        gt_rows.append({
            "pair_id": pid, "present": gt["present"],
            "x": round(gt["x"], 3), "y": round(gt["y"], 3),
            "theta": round(gt["theta"], 3), "scale": round(gt["scale"], 4),
        })
        jury_rows.append({
            "pair_id": pid, "set": subset, "architecture": arch,
            "decoy_architecture": s["decoy_architecture"],
            "channels": channels, "severity": sev,
            "zoom": zoom, "theta": theta, "present": int(present),
            "gt_x": round(gt["x"], 3), "gt_y": round(gt["y"], 3),
            "canvas_size_px": s["canvas_size"],
            "crop_origin_x": s["crop_origin"][0], "crop_origin_y": s["crop_origin"][1],
            "verify_err_px": round(v["err_px"], 3) if v["err_px"] == v["err_px"] else "",
            "verify_peak": round(v["peak"], 4) if v["peak"] == v["peak"] else "",
            "verify_margin": round(v["margin"], 4) if v["margin"] == v["margin"] else "",
            "crop_attempts": v["attempts"],
            **{k: params.as_dict()[k] for k in (
                "dose_search", "shear_amplitude_px", "drift_jitter_px",
                "detector_noise_sigma_search", "charging_streak_prob",
                "charging_streak_intensity", "speckle_sigma", "salt_pepper_prob",
                "astigmatism_ratio", "vignette_strength", "gamma",
                "barrel_distortion_k", "linewidth_bias_nm")},
            "seed": args.seed + idx * 7919,
        })
        print(f"{pid} set{subset} {arch:13s} z={zoom:5.2f} th={theta:+5.2f} "
              f"present={int(present)} ch={channels} "
              f"verify_err={v['err_px']:.2f} margin={v['margin']:.3f} "
              f"tries={v['attempts']} [{time.time()-t0:.0f}s]")

    def dump(name, rows):
        path = os.path.join(args.output_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path

    dump("pairs.csv", pairs_rows)
    dump("ground_truth.csv", gt_rows)
    dump("manifest_jury.csv", jury_rows)
    print(f"\nWrote {len(PLAN)} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
