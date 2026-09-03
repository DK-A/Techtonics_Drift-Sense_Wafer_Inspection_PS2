"""DRAM-style structure generator. Upstream: aayushraina21/drift-sense-synthetic-data."""
import cv2
import numpy as np

from src.structural_defects import maybe_collapse_gap

BACKGROUND = 40
WORD_LINE_VAL = 150
BIT_LINE_VAL = 170
CONTACT_VAL = 225
POSITION_JITTER_NM = 1.5
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


def generate_dram_canvas(size_px, preset, collapse_threshold_nm, rng,
                         linewidth_bias_nm=0.0, corner_rounding_px=0.0, return_layers=False):
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    word_positions = _line_positions(size_px, preset["word_line_pitch_nm"], rng)
    bit_positions = _line_positions(size_px, preset["bit_line_pitch_nm"], rng)
    row_mask = _line_mask(size_px, word_positions, preset["word_line_width_nm"],
                          collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm)
    col_mask = _line_mask(size_px, bit_positions, preset["bit_line_width_nm"],
                          collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm)
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], WORD_LINE_VAL)
    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], BIT_LINE_VAL)
    base_radius = max(preset["contact_diameter_nm"] + linewidth_bias_nm, 1.0) / 2.0
    for i, wl in enumerate(word_positions):
        for j, bl in enumerate(bit_positions):
            if (i + j) % 2 == 0:
                radius = max(1, int(round(base_radius * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)))))
                cv2.circle(canvas, (int(round(bl)), int(round(wl))), radius, CONTACT_VAL, -1)
    if corner_rounding_px >= 0.5:
        k = max(1, int(round(corner_rounding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    return canvas
