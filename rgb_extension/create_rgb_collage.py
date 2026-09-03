"""
create_rgb_collage.py — 8-Pattern RGB Optical Wafer Inspection Slide Collage
Generates a compact 1680x440 blue-black presentation banner showing RGB Reference vs. RGB Search pairs.
"""

import os
import csv
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")

def create_rgb_slide_collage():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    manifest_csv = os.path.join(DATASET_DIR, "manifest_rgb.csv")
    out_path = os.path.join(PLOTS_DIR, "rgb_8_patterns_slide_collage.png")

    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    with open(manifest_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pick 1 representative RGB pair per pattern
    patterns = [
        ("P1", "FIN_ARRAY", "RGB SAQP Grid", 1),
        ("P2", "FIN_CUT", "RGB EUV Mask", 6),
        ("P3", "FIN_GATE", "RGB Gate Matrix", 12),
        ("P4", "CONTACT_ARRAY", "RGB CA/CB Vias", 16),
        ("P5", "LOCAL_INTERCONNECT", "RGB M0 Jogs", 21),
        ("P6", "METAL_ROUTING", "RGB Cu Grid", 26),
        ("P7", "ACTIVE_CELL", "RGB STI Cell", 31),
        ("P8", "FINFET_FULL_CELL", "RGB 3D FinFET", 36),
    ]

    canvas_w, canvas_h = 1680, 440
    # Deep Blue-Black Canvas (BGR: 20, 12, 6)
    canvas = np.full((canvas_h, canvas_w, 3), (20, 12, 6), dtype=np.uint8)

    rows_n, cols_n = 2, 4
    margin_x, margin_y = 12, 12
    gap_x, gap_y = 10, 10

    total_w = canvas_w - 2 * margin_x - (cols_n - 1) * gap_x
    total_h = canvas_h - 2 * margin_y - (rows_n - 1) * gap_y
    card_w = int(total_w / cols_n)
    card_h = int(total_h / rows_n)

    for idx, (p_code, p_name, p_desc, pair_num) in enumerate(patterns):
        r = idx // cols_n
        c = idx % cols_n

        x0 = margin_x + c * (card_w + gap_x)
        y0 = margin_y + r * (card_h + gap_y)
        x1 = x0 + card_w
        y1 = y0 + card_h

        pair_id = f"RGB_PAIR_{pair_num:03d}"
        item = next((row for row in rows if row["pair_id"] == pair_id), rows[0])

        ref_path = os.path.join(BASE_DIR, item["reference_path"])
        search_path = os.path.join(BASE_DIR, item["search_path"])
        gt_x = float(item["gt_x"])
        gt_y = float(item["gt_y"])
        scale_val = float(item.get("scale_factor", 0.100))

        ref_img = cv2.imread(ref_path)
        search_img = cv2.imread(search_path)

        if ref_img is None: ref_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        if search_img is None: search_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        # Card container
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (36, 20, 12), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (65, 38, 22), 1)

        # Header bar
        card_header_h = 24
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + card_header_h), (46, 26, 16), -1)
        cv2.line(canvas, (x0, y0 + card_header_h), (x1, y0 + card_header_h), (65, 38, 22), 1)
        cv2.putText(canvas, f"{p_code}: {p_name} ({p_desc})", (x0 + 8, y0 + 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        box_size = 152
        img_y = y0 + card_header_h + 8
        h_spacing = (card_w - (2 * box_size)) // 3
        ref_x = x0 + h_spacing
        search_x = ref_x + box_size + h_spacing

        # 1. 100x RGB Reference
        ref_resized = cv2.resize(ref_img, (box_size, box_size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(ref_resized, (0, 0), (box_size - 1, box_size - 1), (255, 140, 0), 2)
        canvas[img_y:img_y + box_size, ref_x:ref_x + box_size] = ref_resized

        # Tag in corner
        cv2.rectangle(canvas, (ref_x + 2, img_y + 2), (ref_x + 72, img_y + 16), (20, 10, 5), -1)
        cv2.putText(canvas, "100x RGB", (ref_x + 5, img_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 170, 50), 1, cv2.LINE_AA)

        # 2. 10x RGB Search Field with GT box
        search_copy = search_img.copy()
        box_half = int(round(500.0 * scale_val))
        cv2.rectangle(search_copy, (int(round(gt_x - box_half)), int(round(gt_y - box_half))),
                      (int(round(gt_x + box_half)), int(round(gt_y + box_half))), (0, 255, 0), 3)
        cv2.drawMarker(search_copy, (int(round(gt_x)), int(round(gt_y))), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
        
        search_resized = cv2.resize(search_copy, (box_size, box_size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(search_resized, (0, 0), (box_size - 1, box_size - 1), (0, 220, 100), 2)
        canvas[img_y:img_y + box_size, search_x:search_x + box_size] = search_resized

        # Tag in corner
        cv2.rectangle(canvas, (search_x + 2, img_y + 2), (search_x + 82, img_y + 16), (20, 10, 5), -1)
        cv2.putText(canvas, "10x Search", (search_x + 5, img_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 130), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"[+] RGB 8-pattern slide collage saved: {out_path}")

if __name__ == "__main__":
    create_rgb_slide_collage()
