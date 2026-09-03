#!/usr/bin/env python3
"""Contact sheet: every search image with its ground-truth mark, reference inset."""
import csv
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "./phase2_samples"
pairs = list(csv.DictReader(open(os.path.join(D, "pairs.csv"))))
gt = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(D, "ground_truth.csv")))}
jury = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(D, "manifest_jury.csv")))}

SETC = {"A": "#347DA2", "B": "#0F4880", "C": "#C75000", "D": "#4F840E"}

fig, axes = plt.subplots(5, 4, figsize=(17.5, 23.5))
for ax, p in zip(axes.ravel(), pairs):
    pid = p["pair_id"]; g = gt[pid]; j = jury[pid]
    srch = cv2.imread(os.path.join(D, p["search_path"]), cv2.IMREAD_COLOR)
    ref = cv2.imread(os.path.join(D, p["reference_path"]), cv2.IMREAD_COLOR)
    srch = cv2.cvtColor(srch, cv2.COLOR_BGR2RGB)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    present = int(g["present"])
    z = float(j["zoom"]); th = float(j["theta"])

    canvas = srch.copy()
    if present:
        x, y = float(g["x"]), float(g["y"])
        side = 1000.0 / z
        h = side / 2.0
        # GT box, rotated by theta about the centre
        t = np.deg2rad(th)
        c, s = np.cos(t), np.sin(t)
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        R = np.array([[c, s], [-s, c]])
        pts = (corners @ R.T + [x, y]).astype(np.int32)
        cv2.polylines(canvas, [pts], True, (255, 210, 0), 3)
        cv2.drawMarker(canvas, (int(round(x)), int(round(y))), (255, 60, 60),
                       cv2.MARKER_CROSS, 26, 3)
    else:
        cv2.rectangle(canvas, (6, 6), (993, 993), (230, 80, 0), 6)

    # Reference inset, bottom-right
    ins = cv2.resize(ref, (300, 300), interpolation=cv2.INTER_AREA)
    cv2.rectangle(ins, (0, 0), (299, 299), (255, 255, 255), 4)
    canvas[690:990, 690:990] = ins

    ax.imshow(canvas)
    tag = {"A": "A nominal", "B": f"B degraded L{j['severity']}",
           "C": "C ABSENT", "D": "D optical"}[j["set"]]
    ax.set_title(f"{pid}  |  {tag}\n{j['architecture']}   z={z:.2f}x   "
                 f"$\\theta$={th:+.2f}°   present={present}",
                 fontsize=11, color=SETC[j["set"]],
                 fontweight="bold", linespacing=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(SETC[j["set"]]); sp.set_linewidth(2.5)

fig.suptitle("Drift-Sense Phase 2 — 20 sample pairs\n"
             "search image with ground-truth pose box (yellow) and centre (red); "
             "reference inset bottom-right; orange frame = no true instance",
             fontsize=15, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
out = os.path.join(D, "contact_sheet.png")
fig.savefig(out, dpi=62, facecolor="white")
print("wrote", out)
