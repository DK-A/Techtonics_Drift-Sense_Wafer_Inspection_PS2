import os
import csv
import cv2
import numpy as np

def create_minimalist_pattern_collage():
    manifest_path = "submission_dataset/manifest.csv"
    out_path = "results/plots/all_8_patterns_slide_collage.png"
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Selected diverse, off-center, distributed targets across all 4 quadrants and stress scenarios
    patterns = [
        ("P1", "FIN_ARRAY", "SAQP Fin Grid", 5),          # Top-Left (248.89, 228.04)
        ("P2", "FIN_CUT", "EUV Cut Mask", 21),          # Bottom-Right (698.89, 702.37)
        ("P3", "FIN_GATE", "3D Gate Matrix", 37),       # Off-Center Upper (560.45, 412.30)
        ("P4", "CONTACT_ARRAY", "BEOL CA/CB Vias", 50), # Top-Left (280.11, 235.40)
        ("P5", "LOCAL_INTERCONNECT", "M0 Jogs", 66),    # Bottom-Right (720.50, 690.20)
        ("P6", "METAL_ROUTING", "M1/M2 Grid", 80),      # Top-Left (260.15, 245.90)
        ("P7", "ACTIVE_CELL", "Standard Cell", 104),    # Bottom-Left (251.51, 714.49)
        ("P8", "FINFET_FULL_CELL", "Full 3D FinFET", 120) # Top-Right (685.03, 274.85)
    ]

    # Clean Minimalist Half-Slide Canvas: 1680 x 440 px
    canvas_w, canvas_h = 1680, 440
    
    # Deep Blue-Black Canvas Background (BGR: 20, 12, 6)
    canvas = np.full((canvas_h, canvas_w, 3), (20, 12, 6), dtype=np.uint8)

    # 2 Rows x 4 Columns (Edge-to-edge compact layout)
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

        pair_id = f"PAIR_{pair_num:03d}"
        item = next((row for row in rows if row["pair_id"] == pair_id), None)
        if item is None:
            item = next((row for row in rows if row["pattern_code"] == p_code), rows[0])

        ref_path = item["reference_path"]
        search_path = item["search_path"]
        gt_x = float(item["gt_x"])
        gt_y = float(item["gt_y"])
        scale_val = float(item.get("scale_factor", 0.100))

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None: ref_img = np.zeros((1000, 1000), dtype=np.uint8)
        if search_img is None: search_img = np.zeros((1000, 1000), dtype=np.uint8)

        # Midnight Navy Blue Card Body (BGR: 36, 20, 12)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (36, 20, 12), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (65, 38, 22), 1)

        # Card Title Header (BGR: 46, 26, 16)
        card_header_h = 24
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + card_header_h), (46, 26, 16), -1)
        cv2.line(canvas, (x0, y0 + card_header_h), (x1, y0 + card_header_h), (65, 38, 22), 1)

        # Clean Pattern Title
        cv2.putText(canvas, f"{p_code}: {p_name} ({p_desc})", (x0 + 8, y0 + 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Image box sizing
        box_size = 152
        img_y = y0 + card_header_h + 8
        
        # Horizontal layout centered
        h_spacing = (card_w - (2 * box_size)) // 3
        ref_x = x0 + h_spacing
        search_x = ref_x + box_size + h_spacing

        # 1. 100x Reference Preview
        ref_resized = cv2.resize(ref_img, (box_size, box_size), interpolation=cv2.INTER_AREA)
        ref_bgr = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(ref_bgr, (0, 0), (box_size - 1, box_size - 1), (255, 140, 0), 2)
        canvas[img_y:img_y + box_size, ref_x:ref_x + box_size] = ref_bgr

        # Label in corner
        cv2.rectangle(canvas, (ref_x + 2, img_y + 2), (ref_x + 65, img_y + 16), (20, 10, 5), -1)
        cv2.putText(canvas, "100x Ref", (ref_x + 5, img_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 170, 50), 1, cv2.LINE_AA)

        # 2. 10x Search Preview with GT marker & bounding box
        search_bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        box_half = int(round(500.0 * scale_val))
        cv2.rectangle(search_bgr, (int(round(gt_x - box_half)), int(round(gt_y - box_half))),
                      (int(round(gt_x + box_half)), int(round(gt_y + box_half))), (0, 255, 0), 3)
        cv2.drawMarker(search_bgr, (int(round(gt_x)), int(round(gt_y))), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
        
        search_resized = cv2.resize(search_bgr, (box_size, box_size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(search_resized, (0, 0), (box_size - 1, box_size - 1), (0, 220, 100), 2)
        canvas[img_y:img_y + box_size, search_x:search_x + box_size] = search_resized

        # Label in corner
        cv2.rectangle(canvas, (search_x + 2, img_y + 2), (search_x + 76, img_y + 16), (20, 10, 5), -1)
        cv2.putText(canvas, "10x Search", (search_x + 5, img_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 130), 1, cv2.LINE_AA)

    # Save outputs
    cv2.imwrite(out_path, canvas)
    cv2.imwrite("results/plots/all_8_patterns_overview_collage.png", canvas)
    print(f"[+] Off-center random minimalist pattern collage saved: {out_path}")

if __name__ == "__main__":
    create_minimalist_pattern_collage()
