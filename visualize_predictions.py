"""
visualize_predictions.py — Visual Diagnostics & Failure Analysis
Renders publication-ready side-by-side localization diagnostic figures:
- Reference Image (100x magnification)
- Search Image (10x magnification) with Ground Truth (Green) vs Predicted (Magenta) bounding boxes
- Error vector and Phase escalation HUD metrics

Usage:
    python visualize_predictions.py --pred results/predictions_val.csv --gt dataset/val_metadata.csv --out_dir results/diagnostics --top_n 10
"""

import os
import argparse
import csv
import numpy as np
import cv2


def render_diagnostic_overlay(ref_img, search_img, gt_x, gt_y, pred_x, pred_y, box_size, metadata_text):
    """
    Renders high-contrast side-by-side diagnostic visualization figure.
    """
    sh, sw = search_img.shape[:2]
    rh, rw = ref_img.shape[:2]

    # Convert to 3-channel BGR
    search_bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR) if len(search_img.shape) == 2 else search_img.copy()
    ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape) == 2 else ref_img.copy()

    half_b = box_size / 2.0

    # 1. Draw Ground Truth Box & Crosshair (Bright Green)
    gx0, gy0 = int(round(gt_x - half_b)), int(round(gt_y - half_b))
    gx1, gy1 = int(round(gt_x + half_b)), int(round(gt_y + half_b))
    cv2.rectangle(search_bgr, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)
    cv2.drawMarker(search_bgr, (int(round(gt_x)), int(round(gt_y))), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(search_bgr, "GT", (gx0, max(15, gy0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 2. Draw Predicted Box & Crosshair (Bright Magenta)
    px0, py0 = int(round(pred_x - half_b)), int(round(pred_y - half_b))
    px1, py1 = int(round(pred_x + half_b)), int(round(pred_y + half_b))
    cv2.rectangle(search_bgr, (px0, py0), (px1, py1), (255, 0, 255), 2)
    cv2.drawMarker(search_bgr, (int(round(pred_x)), int(round(pred_y))), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(search_bgr, "PRED", (px0, max(15, py0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    # 3. Draw Error Vector (Red Line)
    err = np.hypot(pred_x - gt_x, pred_y - gt_y)
    if err > 0.5:
        cv2.line(search_bgr, (int(round(gt_x)), int(round(gt_y))), (int(round(pred_x)), int(round(pred_y))), (0, 0, 255), 2)

    # 4. Canvas Composition
    canvas_w = rw + sw + 40
    canvas_h = max(rh, sh) + 120
    canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

    # Place Ref and Search
    canvas[80:80+rh, 20:20+rw] = ref_bgr
    canvas[80:80+sh, 20+rw+20:20+rw+20+sw] = search_bgr

    # Text Headers
    cv2.putText(canvas, "REFERENCE (100x Magnification)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "SEARCH FIELD (10x Magnification)", (20+rw+20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # HUD Metadata
    hud_y = canvas_h - 20
    cv2.putText(canvas, metadata_text, (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Render Localization Diagnostics")
    parser.add_argument("--pred", required=True, help="Predictions CSV")
    parser.add_argument("--gt", required=True, help="Ground Truth CSV")
    parser.add_argument("--out_dir", default="results/diagnostics", help="Output directory")
    parser.add_argument("--top_n", type=int, default=12, help="Number of samples to visualize")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def load_dict(p):
        d = {}
        with open(p, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d[r["pair_id"]] = r
        return d

    preds = load_dict(args.pred)
    gts = load_dict(args.gt)

    rendered = 0
    for pid, p in preds.items():
        if pid not in gts or rendered >= args.top_n:
            continue

        g = gts[pid]
        ref_p = g["ref_path"]
        search_p = g["search_path"]

        if not os.path.exists(ref_p) or not os.path.exists(search_p):
            continue

        ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_p, cv2.IMREAD_GRAYSCALE)

        gx, gy = float(g["gt_x"]), float(g["gt_y"])
        px, py = float(p["pred_x"]), float(p["pred_y"])
        err = np.hypot(px - gx, py - gy)

        box_size = 256.0 if g.get("pattern_type") in ["FIN_ARRAY", "FIN_CUT", "GATE_POLY", "FIN_GATE", "CONTACT_ARRAY"] else 125.0

        meta = (
            f"Pair: {pid} | Class: {g.get('pattern_type')} | "
            f"Error: {err:.2f}px | Conf: {p.get('confidence', 1.0)} | "
            f"Path: {p.get('path_used')} | Scale: {p.get('scale_used')} | Runtime: {p.get('runtime_ms', 0)}ms"
        )

        diag_fig = render_diagnostic_overlay(ref_img, search_img, gx, gy, px, py, box_size, meta)
        out_path = os.path.join(args.out_dir, f"diag_{pid}.png")
        cv2.imwrite(out_path, diag_fig)
        rendered += 1
        print(f"Rendered diagnostic figure: {out_path} (Error: {err:.3f} px)")

    print(f"\nAll {rendered} diagnostic figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
