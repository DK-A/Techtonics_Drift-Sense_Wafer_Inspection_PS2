"""FinFET-style structure generator. Upstream: aayushraina21/drift-sense-synthetic-data."""
import cv2
import numpy as np

from src.structural_defects import maybe_collapse_gap

BACKGROUND = 40
FIN_VAL = 150
GATE_VAL = 170
CONTACT_VAL = 225
POSITION_JITTER_NM = 1.0
WIDTH_JITTER_FRACTION = 0.10


def _line_positions(size_px, pitch_nm, rng):
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, POSITION_JITTER_NM)
    return np.array(positions)


def _line_mask(size_px, positions, width_nm, collapse_threshold_nm, rng,
               width_jitter_fraction=WIDTH_JITTER_FRACTION, linewidth_bias_nm=0.0):
    mask = np.zeros(size_px, dtype=bool)
    biased_width_nm = max(width_nm + linewidth_bias_nm, 1.0)
    widths = biased_width_nm * (1.0 + rng.normal(0, width_jitter_fraction, size=len(positions)))
    widths = np.clip(widths, biased_width_nm * 0.5, biased_width_nm * 1.5)
    for i, center in enumerate(positions):
        half_w = widths[i] / 2.0
        lo = int(round(center - half_w)); hi = int(round(center + half_w))
        mask[max(lo, 0):min(hi, size_px)] = True
        if i + 1 < len(positions):
            next_center = positions[i + 1]; next_half_w = widths[i + 1] / 2.0
            gap_nm = (next_center - next_half_w) - (center + half_w)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                bridge_lo = int(round(center + half_w)); bridge_hi = int(round(next_center - next_half_w))
                mask[max(bridge_lo, 0):min(bridge_hi, size_px)] = True
    return mask


def generate_finfet_canvas(size_px, preset, collapse_threshold_nm, rng,
                           linewidth_bias_nm=0.0, corner_rounding_px=0.0, return_layers=False):
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    fin_positions = _line_positions(size_px, preset["fin_pitch_nm"], rng)
    gate_positions = _line_positions(size_px, preset["gate_pitch_nm"], rng)
    col_mask = _line_mask(size_px, fin_positions, preset["fin_width_nm"],
                          collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm)
    row_mask = _line_mask(size_px, gate_positions, preset["gate_length_nm"],
                          collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm)
    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], FIN_VAL)
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], GATE_VAL)
    half = max(1, int(round(max(preset["contact_size_nm"] + linewidth_bias_nm, 1.0) / 2.0)))
    for i, fin_x in enumerate(fin_positions):
        for j in range(len(gate_positions) - 1):
            if (i + j) % 2 == 0:
                mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
                x, y = int(round(fin_x)), int(round(mid_y))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, CONTACT_VAL, -1)
    if corner_rounding_px >= 0.5:
        k = max(1, int(round(corner_rounding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    return canvas
