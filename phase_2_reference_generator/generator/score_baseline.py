#!/usr/bin/env python3
"""Run the naive ZNCC baseline over the sample set and report Phase 2 metrics.

This is the organizer calibration step: if the naive method scores too well
the set is too easy to separate a field; if it scores near zero the set is
noise. It also doubles as a worked example of the scoring rubric.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from baseline_zncc import search_pose

TIERS = [(1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40)]


def credit(err):
    for t, c in TIERS:
        if err <= t:
            return c
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./phase2_samples")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    pairs = list(csv.DictReader(open(os.path.join(args.dir, "pairs.csv"))))
    gt = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(args.dir, "ground_truth.csv")))}
    jury = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(args.dir, "manifest_jury.csv")))}

    rows = []
    print(f"{'id':5s} {'set':4s} {'pres':4s} {'peak':6s} {'err_px':8s} "
          f"{'z_hat/z':14s} {'th_hat/th':14s} {'credit':6s}")
    for p in pairs:
        pid = p["pair_id"]
        ref = cv2.imread(os.path.join(args.dir, p["reference_path"]), cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(args.dir, p["search_path"]), cv2.IMREAD_GRAYSCALE)
        r = search_pose(ref, srch)
        g = gt[pid]
        present = int(g["present"])
        pred_present = int(r["score"] >= args.threshold)
        if present:
            err = float(np.hypot(r["x"] - float(g["x"]), r["y"] - float(g["y"])))
            cr = credit(err) if pred_present else 0.0
            zs = f"{r['scale']:5.2f}/{float(g['scale']):5.2f}"
            ts = f"{r['theta']:+5.1f}/{float(g['theta']):+5.1f}"
        else:
            err, cr, zs, ts = float("nan"), float("nan"), "-", "-"
        print(f"{pid:5s} {jury[pid]['set']:4s} {present:<4d} {r['score']:6.3f} "
              f"{err:8.2f} {zs:14s} {ts:14s} "
              f"{'' if cr != cr else f'{cr:.2f}':6s}")
        rows.append(dict(pair_id=pid, subset=jury[pid]["set"], present=present,
                         pred_present=pred_present, peak=r["score"], err=err,
                         credit=cr, z_hat=r["scale"], th_hat=r["theta"],
                         z=float(g["scale"]) if present else None,
                         th=float(g["theta"]) if present else None))

    pres = [r for r in rows if r["present"]]
    absent = [r for r in rows if not r["present"]]
    print("\n--- calibration ---")
    print(f"present peaks : min={min(r['peak'] for r in pres):.3f} "
          f"max={max(r['peak'] for r in pres):.3f}")
    print(f"absent  peaks : min={min(r['peak'] for r in absent):.3f} "
          f"max={max(r['peak'] for r in absent):.3f}")
    gap = min(r['peak'] for r in pres) - max(r['peak'] for r in absent)
    print(f"separation gap: {gap:+.3f}  (positive = rejectable by threshold)")

    tp = sum(1 for r in rows if r["present"] and r["pred_present"])
    fp = sum(1 for r in rows if not r["present"] and r["pred_present"])
    fn = sum(1 for r in rows if r["present"] and not r["pred_present"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"rejection @ thr={args.threshold}: TP={tp} FP={fp} FN={fn} "
          f"precision={prec:.2f} recall={rec:.2f} F1={f1:.3f}")

    for label, subset in (("Set A", "A"), ("Set B", "B"), ("Set D", "D")):
        sub = [r for r in pres if r["subset"] == subset]
        if sub:
            print(f"{label}: mean credit={np.mean([r['credit'] for r in sub]):.3f} "
                  f"median err={np.median([r['err'] for r in sub]):.2f}px")

    zerr = [abs(r["z_hat"] - r["z"]) / r["z"] for r in pres]
    terr = [abs(r["th_hat"] - r["th"]) for r in pres]
    print(f"pose (coarse grid): scale within {np.max(zerr)*100:.1f}% worst, "
          f"median {np.median(zerr)*100:.1f}%; theta worst {np.max(terr):.2f} deg, "
          f"median {np.median(terr):.2f} deg")
    print(f"overall mean credit (present pairs): "
          f"{np.mean([r['credit'] for r in pres]):.3f}")


if __name__ == "__main__":
    main()
