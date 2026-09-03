"""
contact_sheet.py — Visual QA Contact Sheet Generator (Section 2.6)
Renders a grid contact sheet showing all search images with ground-truth pose & reference insets.
"""

import os
import sys
import csv
import numpy as np
import cv2

def generate_contact_sheet(dataset_dir="phase2_dataset", output_png="contact_sheet.png"):
    pairs_csv = os.path.join(dataset_dir, "pairs.csv")
    gt_csv = os.path.join(dataset_dir, "ground_truth.csv")
    if not os.path.exists(gt_csv):
        gt_csv = os.path.join(dataset_dir, "metadata.csv")

    if not os.path.exists(pairs_csv) or not os.path.exists(gt_csv):
        print(f"[Error] Dataset files missing in {dataset_dir}")
        return

    with open(pairs_csv, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    with open(gt_csv, "r", encoding="utf-8") as f:
        gts = {r["pair_id"]: r for r in csv.DictReader(f)}

    # Render top 20 pairs in 4x5 grid
    top20 = pairs[:20]
    cell_h, cell_w = 300, 300
    rows_n, cols_n = 4, 5
    canvas = np.zeros((rows_n * cell_h, cols_n * cell_w, 3), dtype=np.uint8)

    for idx, p in enumerate(top20):
        r_idx = idx // cols_n
        c_idx = idx % cols_n

        pid = p["pair_id"]
        gt = gts.get(pid, {})

        sp = os.path.join(dataset_dir, p["search_path"]) if not os.path.isabs(p["search_path"]) else p["search_path"]
        rp = os.path.join(dataset_dir, p["reference_path"]) if not os.path.isabs(p["reference_path"]) else p["reference_path"]

        search_img = cv2.imread(sp, cv2.IMREAD_COLOR)
        ref_img = cv2.imread(rp, cv2.IMREAD_COLOR)

        if search_img is None:
            search_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        if ref_img is None:
            ref_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        cell_search = cv2.resize(search_img, (cell_w, cell_h))
        cell_ref = cv2.resize(ref_img, (75, 75))

        present = int(gt.get("present", gt.get("gt_found", 1)))
        if present == 1:
            gt_x = float(gt.get("x", gt.get("gt_x", 500.0)))
            gt_y = float(gt.get("y", gt.get("gt_y", 500.0)))
            cx = int(round(gt_x * (cell_w / 1000.0)))
            cy = int(round(gt_y * (cell_h / 1000.0)))

            # Draw green crosshair & bounding circle for GT
            cv2.circle(cell_search, (cx, cy), 12, (0, 255, 0), 2)
            cv2.drawMarker(cell_search, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            lbl = f"P{idx+1:02d}: PRESENT"
            color = (0, 255, 0)
        else:
            # Mark red ABSENT box
            cv2.rectangle(cell_search, (5, 5), (cell_w - 5, cell_h - 5), (0, 0, 255), 3)
            lbl = f"P{idx+1:02d}: ABSENT"
            color = (0, 0, 255)

        # Inset reference image in top-right corner
        cell_search[10:85, cell_w-85:cell_w-10] = cell_ref
        cv2.rectangle(cell_search, (cell_w-85, 10), (cell_w-10, 85), (255, 255, 255), 1)

        # Add text label
        cv2.putText(cell_search, lbl, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        canvas[r_idx*cell_h:(r_idx+1)*cell_h, c_idx*cell_w:(c_idx+1)*cell_w] = cell_search

    cv2.imwrite(output_png, canvas)
    print(f"Contact Sheet saved to {output_png} (Grid 4x5, {canvas.shape[1]}x{canvas.shape[0]} px)")

def main():
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else "phase2_dataset"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "contact_sheet.png"
    generate_contact_sheet(ds_dir, out_png)

if __name__ == "__main__":
    main()
