"""
localize.py — Cross-Magnification SEM Pattern Localization System
Implements the 5-Phase Cascade Pipeline:

Phase 1: Global Multi-Scale / Multi-Angle NCC + Confidence Calibration (Gap & Sharpness)
Phase 2: Pitch & Geometry Consistency Scoring (Autocorrelation / FFT & Lattice Disambiguation)
Phase 3: Fine Local Search (Narrow local crop with sub-degree and fine-scale sweep)
Phase 4: Sub-Pixel Refinement (2D Fourier Phase Correlation + Parabolic Peak Interpolation)
Phase 5: ML Embedding Re-Ranking (Deep metric cosine similarity for high-ambiguity repeats)
Tiebreak Rule: Feature-Consistent Combined Score (Gradient Coherence + Contrast + NCC)

Usage:
    # Single pair inference:
    python localize.py --reference path/to/ref.png --search path/to/search.png

    # Batch manifest inference:
    python localize.py --manifest path/to/metadata.csv --out_csv results/predictions.csv
"""

import os
import sys
import time
import argparse
import json
import csv
import numpy as np
import cv2


# =====================================================================
# PART 1: PREPROCESSING
# =====================================================================

def get_clahe_uint8(img: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to normalize SEM illumination and secondary-electron contrast.
    """
    if img is None:
        raise ValueError("Input image is None")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# =====================================================================
# PART 2: PHASE 1 — GLOBAL NCC MATCHING + CONFIDENCE CALIBRATION
# =====================================================================

# =====================================================================
# PART 2: PHASE 1 — GLOBAL NCC MATCHING & SINGLE CALIBRATED GATE
# =====================================================================

def generate_reference_variants(ref_img: np.ndarray, scales=None, angles=None, pattern_type: str = "GENERIC"):
    """
    Generates downscaled, rotated reference template variants.
    Scales: Unified 10:1 scale standard (0.100 nominal, 0.091–0.111 robustness range)
    Angles: 9 values in [-2.0 deg, +2.0 deg] with 0.5 deg step
    """
    if scales is None:
        scales = np.array([0.091, 0.095, 0.100, 0.105, 0.111])

    if angles is None:
        angles = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    h, w = ref_img.shape[:2]
    variants = []

    for s in scales:
        target_w = max(16, int(round(w * s)))
        target_h = max(16, int(round(h * s)))
        scaled = cv2.resize(ref_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        for a in angles:
            if abs(a) < 1e-4:
                rotated = scaled
            else:
                center = (target_w / 2.0, target_h / 2.0)
                M = cv2.getRotationMatrix2D(center, a, 1.0)
                rotated = cv2.warpAffine(
                    scaled, M, (target_w, target_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
            variants.append({
                "template": rotated,
                "scale": float(s),
                "angle": float(a),
                "width": target_w,
                "height": target_h
            })
    return variants


def compute_phase1_confidence(score: float, gap: float, sharpness: float) -> float:
    """
    Computes a single unified, calibrated confidence score for Phase 1.
    Combines peak score, peak-to-second gap, and peak sharpness (PSR).
    """
    s_norm = np.clip(score, 0.0, 1.0)
    g_norm = np.clip(gap / 0.15, 0.0, 1.0)
    p_norm = np.clip((sharpness - 1.0) / 0.5, 0.0, 1.0)
    return float(0.45 * s_norm + 0.35 * g_norm + 0.20 * p_norm)


def run_phase1_ncc(search_img_uint8: np.ndarray, ref_variants: list, nms_radius=12.0):
    """
    Evaluates template variants against search image with pyramid acceleration.
    Extracts Top-K candidate pool and computes the single unified confidence gate score.
    """
    sh, sw = search_img_uint8.shape[:2]
    all_peaks = []

    # Coarse pyramid search (2x downsampled)
    search_half = cv2.resize(search_img_uint8, (sw // 2, sh // 2), interpolation=cv2.INTER_AREA)

    variant_scores = []
    for var in ref_variants:
        tpl = var["template"]
        th, tw = tpl.shape[:2]
        tpl_half = cv2.resize(tpl, (max(8, tw // 2), max(8, th // 2)), interpolation=cv2.INTER_AREA)
        if tpl_half.shape[0] >= search_half.shape[0] or tpl_half.shape[1] >= search_half.shape[1]:
            continue

        res_half = cv2.matchTemplate(search_half, tpl_half, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res_half)
        variant_scores.append((max_val, var, max_loc, res_half))

    variant_scores.sort(key=lambda x: x[0], reverse=True)
    top_variants = variant_scores[:14]

    best_response_map = None
    best_variant = None

    for _, var, coarse_loc, _ in top_variants:
        tpl = var["template"]
        th, tw = tpl.shape[:2]
        res = cv2.matchTemplate(search_img_uint8, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best_response_map is None or max_val > np.max(best_response_map):
            best_response_map = res
            best_variant = var

        thresh = max(0.20, max_val - 0.25)
        locs = np.where(res >= thresh)
        for y, x in zip(locs[0], locs[1]):
            all_peaks.append({
                "x_tl": int(x),
                "y_tl": int(y),
                "cx": float(x + tw / 2.0),
                "cy": float(y + th / 2.0),
                "score": float(res[y, x]),
                "scale": var["scale"],
                "angle": var["angle"],
                "width": tw,
                "height": th,
                "res_map": res
            })

    all_peaks.sort(key=lambda p: p["score"], reverse=True)

    nms_pool = []
    for peak in all_peaks:
        is_suppressed = False
        for kept in nms_pool:
            dist = np.hypot(peak["cx"] - kept["cx"], peak["cy"] - kept["cy"])
            if dist < nms_radius:
                is_suppressed = True
                break
        if not is_suppressed:
            nms_pool.append(peak)
            if len(nms_pool) >= 15:
                break

    if not nms_pool:
        nms_pool = [{
            "x_tl": sw // 2 - 50, "y_tl": sh // 2 - 50,
            "cx": sw / 2.0, "cy": sh / 2.0,
            "score": 0.0, "scale": 0.10, "angle": 0.0,
            "width": 100, "height": 100, "res_map": None
        }]

    top1 = nms_pool[0]
    score_top1 = top1["score"]
    score_top2 = nms_pool[1]["score"] if len(nms_pool) > 1 else (score_top1 - 0.4)
    gap = float(score_top1 - score_top2)

    sharpness = 1.0
    if top1.get("res_map") is not None:
        rmap = top1["res_map"]
        px, py = top1["x_tl"], top1["y_tl"]
        ry0, ry1 = max(0, py - 2), min(rmap.shape[0], py + 3)
        rx0, rx1 = max(0, px - 2), min(rmap.shape[1], px + 3)
        nb = rmap[ry0:ry1, rx0:rx1]
        mean_nb = np.mean(nb)
        sharpness = float(score_top1 / (mean_nb + 1e-6))

    gate_conf = compute_phase1_confidence(score_top1, gap, sharpness)

    # Adaptive Candidate Count
    if gate_conf > 0.75:
        target_k = 5
    elif gate_conf > 0.55:
        target_k = 8
    else:
        target_k = 10

    nms_candidates = nms_pool[:target_k]
    return nms_candidates, gap, sharpness, gate_conf, target_k


# =====================================================================
# PART 3: PHASE 2 — PATTERN-SPECIFIC GEOMETRY DISAMBIGUATION
# =====================================================================

def extract_reference_pitch(ref_img: np.ndarray, pattern_type: str = "GENERIC"):
    """
    Computes 2D autocorrelation on the reference image with pattern-specific priors.
    """
    ref_f = ref_img.astype(np.float32)
    ref_f = (ref_f - np.mean(ref_f)) / (np.std(ref_f) + 1e-6)

    F = np.fft.fft2(ref_f)
    power = np.abs(F) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)
    h, w = autocorr.shape
    cy, cx = h // 2, w // 2

    # Mask central self-peak
    r_mask = 14
    autocorr[cy - r_mask:cy + r_mask + 1, cx - r_mask:cx + r_mask + 1] = 0.0

    x_slice = np.mean(autocorr[cy - 16:cy + 17, cx:], axis=0)
    y_slice = np.mean(autocorr[cy:, cx - 16:cx + 17], axis=1)

    peak_x = int(np.argmax(x_slice)) if len(x_slice) > 0 else 0
    peak_y = int(np.argmax(y_slice)) if len(y_slice) > 0 else 0

    pitch_x = float(peak_x) if peak_x > 8 else 0.0
    pitch_y = float(peak_y) if peak_y > 8 else 0.0

    max_ac = float(np.max(autocorr))
    is_periodic = (max_ac > 0.04 * np.sum(power) / (h * w)) and (pitch_x > 0 or pitch_y > 0)

    # Reference pitch at 100x scale mapped to search coordinates (~0.10x)
    pitch_x_search = max(6.0, pitch_x * 0.10) if pitch_x > 0 else 22.0
    pitch_y_search = max(6.0, pitch_y * 0.10) if pitch_y > 0 else 22.0

    return {
        "is_periodic": is_periodic,
        "pitch_x": pitch_x_search,
        "pitch_y": pitch_y_search,
        "strength": max_ac,
        "pattern_type": pattern_type
    }


def run_phase2_geometry_scoring(candidates: list, search_img: np.ndarray, ref_pitch_info: dict) -> tuple:
    """
    Candidate-independent geometry evaluation and periodic ambiguity detection.
    1. Evaluates each candidate independently based on its own local edge energy and structural texture.
       NEVER awards a candidate an artificial geometry score based on self-referential distance to candidates[0].
    2. Identifies genuine periodic ambiguity when top candidates exhibit small NCC score gaps
       and distinct spatial displacements (e.g. periodic multiples or quadrant ambiguity).
    3. When genuine ambiguity exists and classical geometry cannot decisively resolve it,
       returns is_resolved = False to escalate to Phase 5 Siamese ML re-ranking.
    """
    if len(candidates) <= 1:
        return candidates, True

    sh, sw = search_img.shape[:2]
    ptype = ref_pitch_info.get("pattern_type", "GENERIC") if ref_pitch_info else "GENERIC"
    px = ref_pitch_info.get("pitch_x", 40.0) if ref_pitch_info else 40.0
    py = ref_pitch_info.get("pitch_y", 40.0) if ref_pitch_info else 40.0

    crop_r = int(max(24, min(px, py) * 1.5))
    scored_candidates = []

    for cand in candidates:
        cx, cy = cand["cx"], cand["cy"]
        y0, y1 = max(0, int(cy - crop_r)), min(sh, int(cy + crop_r))
        x0, x1 = max(0, int(cx - crop_r)), min(sw, int(cx + crop_r))
        crop = search_img[y0:y1, x0:x1]

        # Candidate-independent local structural quality
        if crop.size > 100:
            c_std = float(np.std(crop))
            contrast_score = float(np.clip(c_std / 38.0, 0.4, 1.2))

            # Directional edge coherence
            sobel_x = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
            edge_x = float(np.mean(np.abs(sobel_x)))
            edge_y = float(np.mean(np.abs(sobel_y)))

            if ptype in ["FIN_ARRAY", "FIN_CUT"]:
                # Fin lines are horizontal -> dominant vertical gradient
                dir_ratio = edge_y / (edge_x + 1e-4)
                edge_score = float(np.clip(dir_ratio / 1.5, 0.5, 1.3))
            elif ptype in ["FIN_GATE", "CONTACT_ARRAY"]:
                # 2D cross-grid structure -> balanced gradients
                edge_score = 1.0
            else:
                edge_score = 1.0

            cand_geom = 0.60 * contrast_score + 0.40 * edge_score
        else:
            cand_geom = 1.0

        # Boundary clearance penalty (avoid border artifact candidates)
        dist_to_border = min(cx, cy, sw - cx, sh - cy)
        border_weight = 1.0 if dist_to_border >= 60.0 else max(0.6, dist_to_border / 60.0)

        # Candidate-independent combined score
        geom_score = float(np.clip(cand_geom * border_weight, 0.2, 1.2))
        combined_score = float(0.70 * cand["score"] + 0.30 * geom_score)

        c_copy = dict(cand)
        c_copy["geom_score"] = round(geom_score, 4)
        c_copy["combined_score"] = round(combined_score, 4)
        scored_candidates.append(c_copy)

    # Sort candidates by combined score
    scored_candidates.sort(key=lambda c: c["combined_score"], reverse=True)

    # Ambiguity Detection: Check if top-2 candidates represent genuine periodic / spatial ambiguity
    c1 = scored_candidates[0]
    c2 = scored_candidates[1] if len(scored_candidates) > 1 else None

    if c2 is not None:
        ncc_gap = abs(c1["score"] - c2["score"])
        combined_gap = abs(c1["combined_score"] - c2["combined_score"])
        spatial_dist = float(np.hypot(c1["cx"] - c2["cx"], c1["cy"] - c2["cy"]))

        # If candidates have close NCC scores (< 0.040) and are spatially distinct (> 12 px),
        # they form a genuine ambiguity pool that geometry alone cannot resolve.
        is_ambiguous = (ncc_gap <= 0.040 or combined_gap <= 0.035) and (spatial_dist >= 12.0)
        is_resolved = not is_ambiguous
    else:
        is_resolved = True

    return scored_candidates, is_resolved


# =====================================================================
# PART 4: PHASE 5 — SIAMESE METRIC RE-RANKER (NO CENTER BIAS)
# =====================================================================

_GLOBAL_EMBEDDER = None

def get_embedder_model(weights_path="model/phase5_reranker.pt"):
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is not None:
        return _GLOBAL_EMBEDDER

    try:
        from train import LightweightSEMEmbedder
        import torch
        model = LightweightSEMEmbedder(emb_dim=64)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()
            _GLOBAL_EMBEDDER = model
        else:
            _GLOBAL_EMBEDDER = None
    except Exception:
        _GLOBAL_EMBEDDER = None
    return _GLOBAL_EMBEDDER


def run_phase5_ml_reranker(candidates: list, search_img: np.ndarray, ref_img: np.ndarray, weights_path=None, embedder_model=None):
    """
    Phase 5: ML Embedding Metric Re-Ranking
    Canonicalizes candidate patches into the upright 128x128 reference frame and
    evaluates deep cosine embedding similarity to resolve periodic lattice ambiguity.
    Candidate Selection Logic:
    1. Highest cosine embedding similarity outranking ghosts.
    2. If candidates are effectively tied, falls back to the feature-consistent combined score
       (directional gradient coherence, local contrast, and NCC sharpness) without spatial center bias.
    """
    if len(candidates) <= 1:
        return candidates[0]

    model = embedder_model if embedder_model is not None else get_embedder_model(weights_path or "model/phase5_reranker.pt")
    sh, sw = search_img.shape[:2]

    scored_cands = []

    if model is not None:
        try:
            import torch
            crop_size = 128
            ref_crop = cv2.resize(ref_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            t_ref = torch.from_numpy(ref_crop.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)

            # Move tensor to same device as model
            param = next(model.parameters(), None)
            if param is not None:
                t_ref = t_ref.to(param.device)

            with torch.no_grad():
                emb_ref = model(t_ref)

            for c in candidates[:8]:
                cx, cy = float(c["cx"]), float(c["cy"])
                sc = float(c.get("scale", 0.100))
                ang = float(c.get("angle", 0.0))

                # Canonicalize search candidate patch into 128x128 upright reference frame
                target_dim = max(10.0, 1000.0 * sc)
                M_align = cv2.getRotationMatrix2D((cx, cy), -ang, crop_size / target_dim)
                M_align[0, 2] += (crop_size / 2.0 - cx)
                M_align[1, 2] += (crop_size / 2.0 - cy)
                s_crop = cv2.warpAffine(
                    search_img, M_align, (crop_size, crop_size),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
                )

                t_s = torch.from_numpy(s_crop.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
                if param is not None:
                    t_s = t_s.to(param.device)

                with torch.no_grad():
                    emb_s = model(t_s)
                    cos_sim = float(torch.sum(emb_ref * emb_s).item())

                c_copy = dict(c)
                c_copy["siamese_score"] = round(cos_sim, 4)
                scored_cands.append(c_copy)
        except Exception:
            scored_cands = [dict(c) for c in candidates]
    else:
        scored_cands = [dict(c) for c in candidates]

    # Sort by Siamese similarity descending
    scored_cands.sort(key=lambda c: c.get("siamese_score", 0.0), reverse=True)

    if len(scored_cands) >= 2 and "siamese_score" in scored_cands[0]:
        s1 = scored_cands[0]["siamese_score"]
        s2 = scored_cands[1]["siamese_score"]
        margin = s1 - s2

        # ML Margin Gating: If clear ML winner (margin >= 0.02 and s1 >= 0.35), accept top candidate
        if s1 >= 0.35 and margin >= 0.02:
            return scored_cands[0]

        # If candidates are tied within margin, use feature-consistent combined classical score (no spatial bias)
        tied_candidates = [c for c in scored_cands if (s1 - c.get("siamese_score", 0.0)) <= 0.02]
        best_candidate = max(
            tied_candidates,
            key=lambda c: float(c.get("combined_score", c.get("score", 0.0)))
        )
        return best_candidate

    # Fallback to classical top candidate
    return candidates[0]


# =====================================================================
# PART 5: PHASE 3 — ADAPTIVE FINE LOCAL SEARCH & QUALITY CHECK
# =====================================================================

def run_phase3_fine_local_search(candidate: dict, search_img: np.ndarray, ref_img: np.ndarray, uncertainty_level: str = "high"):
    """
    Adaptive Fine-Search Window with Post-Search Candidate-Quality Validation:
    - 160x160 (low uncertainty), 200x200 (medium), 240x240 (high).
    - Post-search check: Rejects fine shift if score collapses or shifts implausibly.
    """
    cx, cy = candidate["cx"], candidate["cy"]
    coarse_scale = candidate["scale"]
    coarse_angle = candidate["angle"]

    sh, sw = search_img.shape[:2]

    if uncertainty_level == "low":
        crop_size = 160
    elif uncertainty_level == "medium":
        crop_size = 200
    else:
        crop_size = 240

    half = crop_size // 2
    x0 = int(np.clip(cx - half, 0, max(0, sw - crop_size)))
    y0 = int(np.clip(cy - half, 0, max(0, sh - crop_size)))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    local_search = search_img[y0:y1, x0:x1]

    fine_scales = np.linspace(coarse_scale - 0.005, coarse_scale + 0.005, 5)
    fine_angles = np.array([
        coarse_angle - 0.50, coarse_angle - 0.25, coarse_angle,
        coarse_angle + 0.25, coarse_angle + 0.50
    ])

    best_score = -1.0
    best_loc = (0, 0)
    best_scale = coarse_scale
    best_angle = coarse_angle
    best_tpl = None

    rw, rh = ref_img.shape[1], ref_img.shape[0]

    for fs in fine_scales:
        tw = max(16, int(round(rw * fs)))
        th = max(16, int(round(rh * fs)))
        if th >= crop_size or tw >= crop_size:
            continue
        scaled_ref = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)

        for fa in fine_angles:
            if abs(fa) < 1e-4:
                rot_tpl = scaled_ref
            else:
                center = (tw / 2.0, th / 2.0)
                M = cv2.getRotationMatrix2D(center, fa, 1.0)
                rot_tpl = cv2.warpAffine(
                    scaled_ref, M, (tw, th),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )

            res = cv2.matchTemplate(local_search, rot_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_scale = fs
                best_angle = fa
                best_tpl = rot_tpl

    if best_tpl is not None:
        th, tw = best_tpl.shape[:2]
        proposed_cx = x0 + best_loc[0] + tw / 2.0
        proposed_cy = y0 + best_loc[1] + th / 2.0

        # Post-Search Candidate-Quality Validation Check
        shift_dist = np.hypot(proposed_cx - cx, proposed_cy - cy)
        if best_score >= max(0.20, candidate["score"] - 0.15) and shift_dist <= (crop_size * 0.45):
            fine_cx, fine_cy = proposed_cx, proposed_cy
        else:
            fine_cx, fine_cy = cx, cy
            best_tpl = cv2.resize(ref_img, (int(round(rw * coarse_scale)), int(round(rh * coarse_scale))), interpolation=cv2.INTER_AREA)
    else:
        fine_cx, fine_cy = cx, cy

    return {
        "cx": float(fine_cx),
        "cy": float(fine_cy),
        "scale": float(best_scale),
        "angle": float(best_angle),
        "score": float(best_score if best_score > 0 else candidate["score"]),
        "template": best_tpl,
        "window_size": crop_size
    }


# =====================================================================
# PART 6: PHASE 4 — SAFE CONTINUOUS SUB-PIXEL REFINEMENT
# =====================================================================

def safe_subpixel_refinement(search_img: np.ndarray, tpl: np.ndarray, cx: float, cy: float):
    """
    Safe continuous sub-pixel refinement:
    1. Extracts floating-point subpixel patch directly centered at (cx, cy).
    2. Fourier Phase Correlation computes continuous shift.
    3. Refinement Guardrail: Only accepts shift if stability bounds (|shift| <= 0.45 px, response >= 0.18)
       are met; otherwise returns (0, 0, False) to prevent coordinate degradation.
    """
    th, tw = tpl.shape[:2]
    sh, sw = search_img.shape[:2]

    if th < 8 or tw < 8 or cx < tw / 2.0 or cx > (sw - tw / 2.0) or cy < th / 2.0 or cy > (sh - th / 2.0):
        return 0.0, 0.0, False

    try:
        search_crop = cv2.getRectSubPix(search_img, (tw, th), (float(cx), float(cy)))
        if search_crop.shape[:2] != (th, tw):
            return 0.0, 0.0, False

        tpl_f = tpl.astype(np.float32)
        crop_f = search_crop.astype(np.float32)
        (shift_x, shift_y), response = cv2.phaseCorrelate(crop_f, tpl_f)

        if response >= 0.18 and abs(shift_x) <= 0.45 and abs(shift_y) <= 0.45:
            return float(-shift_x), float(-shift_y), True
    except Exception:
        pass

    return 0.0, 0.0, False


def run_phase4_subpixel_refinement(p3_result: dict, search_img: np.ndarray):
    """
    Applies safe continuous sub-pixel refinement to the candidate position.
    """
    cx, cy = p3_result["cx"], p3_result["cy"]
    tpl = p3_result.get("template")
    if tpl is None:
        return cx, cy, False

    dx, dy, is_valid = safe_subpixel_refinement(search_img, tpl, cx, cy)
    return float(cx + dx), float(cy + dy), is_valid


# =====================================================================
# MASTER PIPELINE CONTROLLER
# =====================================================================

def localize_pair(
    reference_path: str,
    search_path: str,
    tau_conf: float = 0.65,
    pattern_type: str = "GENERIC",
    weights_path: str = None,
    embedder_model = None
) -> dict:
    """
    Executes the complete 5-phase cascade pipeline on a single image pair.
    """
    t_start = time.perf_counter()

    ref_raw = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if ref_raw is None:
        raise FileNotFoundError(f"Could not open reference image: {reference_path}")
    if search_raw is None:
        raise FileNotFoundError(f"Could not open search image: {search_path}")

    # 1. Preprocessing
    ref_norm = get_clahe_uint8(ref_raw)
    search_norm = get_clahe_uint8(search_raw)

    # 2. Phase 1: Global Multi-Scale NCC with Single Calibrated Gate Score
    ref_variants = generate_reference_variants(ref_norm, pattern_type=pattern_type)
    candidates, gap, sharpness, gate_conf, cand_count = run_phase1_ncc(search_norm, ref_variants, nms_radius=12.0)

    selected_candidate = candidates[0]
    top_score = selected_candidate["score"]

    # CASCADE DECISION ENGINE:
    # Check for genuine periodic ambiguity (competing candidate peaks within small score gap and distinct spatial locations)
    c1_c2_dist = float(np.hypot(candidates[0]["cx"] - candidates[1]["cx"], candidates[0]["cy"] - candidates[1]["cy"])) if len(candidates) > 1 else 0.0
    is_periodic_ambiguous = (gap < 0.075) and (len(candidates) > 1) and (c1_c2_dist >= 15.0)

    # Single Calibrated Confidence Gate (requires both high gate_conf AND absence of periodic ambiguity):
    if gate_conf >= tau_conf and not is_periodic_ambiguous:
        # High confidence direct path -> bypass Phase 2 and Phase 5
        path_used = "ncc_direct"
        uncertainty_level = "low"
    else:
        # Phase 2: Pattern-Specific Geometry Disambiguation
        ref_pitch_info = extract_reference_pitch(ref_raw, pattern_type=pattern_type)
        scored_candidates, is_resolved = run_phase2_geometry_scoring(candidates, search_raw, ref_pitch_info)

        if is_resolved:
            selected_candidate = scored_candidates[0]
            path_used = "geometry_verified"
            uncertainty_level = "medium"
        else:
            # Phase 5: Siamese Metric Embedding Re-Ranking (No Center Bias)
            selected_candidate = run_phase5_ml_reranker(scored_candidates, search_raw, ref_raw, weights_path=weights_path, embedder_model=embedder_model)
            path_used = "ml_reranked"
            uncertainty_level = "high"

    # Phase 3: Adaptive Fine Local Search with Post-Search Quality Validation
    p3_res = run_phase3_fine_local_search(selected_candidate, search_norm, ref_norm, uncertainty_level=uncertainty_level)
    window_size = p3_res["window_size"]

    # Phase 4: Safe Sub-Pixel Refinement
    final_x, final_y, subpixel_valid = run_phase4_subpixel_refinement(p3_res, search_norm)
    scale_used = p3_res["scale"]
    angle_used = p3_res["angle"]
    fine_score = p3_res["score"]

    total_runtime_ms = (time.perf_counter() - t_start) * 1000.0

    # Explicit Confidence State
    final_confidence_val = float(np.clip(
        0.50 * fine_score + 0.30 * min(1.0, gap / 0.15) + 0.20 * min(1.0, (sharpness - 1.0) / 0.5),
        0.05, 0.99
    ))

    if final_confidence_val > 0.80 and path_used == "ncc_direct":
        conf_state = "HIGH"
    elif final_confidence_val > 0.60 or path_used == "geometry_verified":
        conf_state = "MEDIUM"
    elif final_confidence_val > 0.40:
        conf_state = "LOW"
    else:
        conf_state = "AMBIGUOUS"

    return {
        "pred_x": round(float(final_x), 3),
        "pred_y": round(float(final_y), 3),
        "confidence": round(final_confidence_val, 4),
        "confidence_state": conf_state,
        "scale_used": round(float(scale_used), 4),
        "angle_used": round(float(angle_used), 2),
        "path_used": path_used,
        "window_size": window_size,
        "subpixel_valid": int(subpixel_valid),
        "candidate_count": cand_count,
        "runtime_ms": round(float(total_runtime_ms), 1),
        "stage_scores": {
            "p1_top1_score": round(float(candidates[0]["score"]), 4),
            "p1_gap": round(float(gap), 4),
            "p1_sharpness": round(float(sharpness), 4),
            "gate_conf": round(float(gate_conf), 4),
            "fine_score": round(float(fine_score), 4)
        }
    }



# =====================================================================
# BATCH EXECUTION & CLI
# =====================================================================

def run_batch_manifest(manifest_csv: str, output_csv: str, weights_path: str = None):
    """
    Processes a manifest CSV listing reference and search image pairs.
    """
    results = []
    base_dir = os.path.dirname(os.path.abspath(manifest_csv))

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} pairs through 5-Phase Cascade Pipeline...")
    runtimes = []
    errors = []

    for idx, row in enumerate(rows):
        pair_id = row.get("pair_id", f"PAIR_{idx+1:06d}")
        pattern_type = row.get("pattern_name") or row.get("pattern_type") or row.get("pattern", "GENERIC")
        split = row.get("split", "train")

        ref_path = row.get("ref_path") or row.get("reference_path") or ""
        search_path = row.get("search_path") or ""

        # Robust path resolution
        if ref_path and os.path.exists(ref_path):
            pass
        elif ref_path and os.path.exists(os.path.join(base_dir, ref_path)):
            ref_path = os.path.join(base_dir, ref_path)
        elif ref_path and os.path.exists(os.path.join(base_dir, "..", ref_path)):
            ref_path = os.path.abspath(os.path.join(base_dir, "..", ref_path))
        else:
            p_num = int(pair_id.split("_")[1]) if "_" in pair_id else idx + 1
            if os.path.exists(os.path.join(base_dir, "reference", f"ref_{p_num:03d}.png")):
                ref_path = os.path.join(base_dir, "reference", f"ref_{p_num:03d}.png")
            elif os.path.exists(os.path.join(base_dir, split, "ref", f"ref_{p_num:06d}.png")):
                ref_path = os.path.join(base_dir, split, "ref", f"ref_{p_num:06d}.png")

        if search_path and os.path.exists(search_path):
            pass
        elif search_path and os.path.exists(os.path.join(base_dir, search_path)):
            search_path = os.path.join(base_dir, search_path)
        elif search_path and os.path.exists(os.path.join(base_dir, "..", search_path)):
            search_path = os.path.abspath(os.path.join(base_dir, "..", search_path))
        else:
            p_num = int(pair_id.split("_")[1]) if "_" in pair_id else idx + 1
            if os.path.exists(os.path.join(base_dir, "search", f"search_{p_num:03d}.png")):
                search_path = os.path.join(base_dir, "search", f"search_{p_num:03d}.png")
            elif os.path.exists(os.path.join(base_dir, split, "search", f"search_{p_num:06d}.png")):
                search_path = os.path.join(base_dir, split, "search", f"search_{p_num:06d}.png")

        res = localize_pair(ref_path, search_path, pattern_type=pattern_type, weights_path=weights_path)
        runtimes.append(res["runtime_ms"])

        gt_x = float(row["gt_x"]) if "gt_x" in row and row["gt_x"] != "" else None
        gt_y = float(row["gt_y"]) if "gt_y" in row and row["gt_y"] != "" else None
        err_px = None
        if gt_x is not None and gt_y is not None:
            err_px = round(float(np.sqrt((res["pred_x"] - gt_x)**2 + (res["pred_y"] - gt_y)**2)), 3)
            errors.append(err_px)

        res_row = {
            "pair_id": pair_id,
            "pattern_type": pattern_type,
            "pred_x": res["pred_x"],
            "pred_y": res["pred_y"],
            "gt_x": gt_x if gt_x is not None else "",
            "gt_y": gt_y if gt_y is not None else "",
            "error_px": err_px if err_px is not None else "",
            "confidence": res["confidence"],
            "scale_used": res["scale_used"],
            "angle_used": res["angle_used"],
            "path_used": res["path_used"],
            "runtime_ms": res["runtime_ms"]
        }
        results.append(res_row)
        if gt_x is not None and gt_y is not None:
            print(f"  [{idx+1:03d}/{len(rows)}] {pair_id} ({pattern_type:<18}) -> Pred: ({res['pred_x']:6.2f}, {res['pred_y']:6.2f}) | GT: ({gt_x:6.2f}, {gt_y:6.2f}) | Err: {err_px:6.3f} px | {res['path_used']} ({res['runtime_ms']:5.1f}ms)")
        else:
            print(f"  [{idx+1:03d}/{len(rows)}] {pair_id} ({pattern_type:<18}) -> Pred: ({res['pred_x']:6.2f}, {res['pred_y']:6.2f}) | Path: {res['path_used']} ({res['runtime_ms']:5.1f}ms)")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = list(results[0].keys()) if results else [
        "pair_id", "pattern_type", "pred_x", "pred_y", "gt_x", "gt_y", "error_px",
        "confidence", "scale_used", "angle_used", "path_used", "runtime_ms"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved predictions to {output_csv}")
    print(f"Mean Runtime: {np.mean(runtimes):.1f} ms | Median Runtime: {np.median(runtimes):.1f} ms")
    if errors:
        print(f"Mean Error: {np.mean(errors):.3f} px | Median Error: {np.median(errors):.3f} px")
        print(f"Pass Rate (<5px): {np.mean([e < 5.0 for e in errors])*100.0:.1f}%")
        print(f"Pass Rate (<1px): {np.mean([e < 1.0 for e in errors])*100.0:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Cross-Magnification SEM Pattern Localization")
    parser.add_argument("--reference", type=str, help="Path to 1000x1000 reference image")
    parser.add_argument("--search", type=str, help="Path to 1000x1000 search image")
    parser.add_argument("--pattern", type=str, default="GENERIC", help="Pattern type name (optional)")
    parser.add_argument("--gt-x", "--gt_x", type=float, default=None, help="Optional ground truth X coordinate for error calculation")
    parser.add_argument("--gt-y", "--gt_y", type=float, default=None, help="Optional ground truth Y coordinate for error calculation")
    parser.add_argument("--manifest", type=str, help="Path to manifest CSV for batch processing")
    parser.add_argument("--out_csv", type=str, default="results/predictions.csv", help="Output predictions CSV path")
    parser.add_argument("--weights", type=str, default=None, help="Path to optional ML weights")
    parser.add_argument("--tau_conf", type=float, default=0.65, help="Phase 1 confidence threshold")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    args = parser.parse_args()

    if args.manifest:
        run_batch_manifest(args.manifest, args.out_csv, weights_path=args.weights)
    elif args.reference and args.search:
        res = localize_pair(
            args.reference,
            args.search,
            tau_conf=args.tau_conf,
            pattern_type=args.pattern,
            weights_path=args.weights
        )
        gt_x = args.gt_x
        gt_y = args.gt_y
        if (gt_x is None or gt_y is None) and os.path.exists("submission_dataset/manifest.csv"):
            ref_base = os.path.basename(args.reference)
            search_base = os.path.basename(args.search)
            try:
                with open("submission_dataset/manifest.csv", "r", encoding="utf-8") as mf:
                    reader = csv.DictReader(mf)
                    for row in reader:
                        if os.path.basename(row.get("reference_path", "")) == ref_base or os.path.basename(row.get("search_path", "")) == search_base:
                            gt_x = float(row["gt_x"])
                            gt_y = float(row["gt_y"])
                            break
            except Exception:
                pass

        if args.json:
            out = dict(res)
            if gt_x is not None and gt_y is not None:
                out["gt_x"] = gt_x
                out["gt_y"] = gt_y
                out["error_px"] = float(np.hypot(res['pred_x'] - gt_x, res['pred_y'] - gt_y))
            print(json.dumps(out, indent=2))
        else:
            print("==================================================")
            print("SEM PATTERN LOCALIZATION RESULT")
            print("==================================================")
            print(f"Predicted Center: ({res['pred_x']:.3f}, {res['pred_y']:.3f}) px")
            print(f"Confidence:       {res['confidence']:.4f}")
            print(f"Scale Used:       {res['scale_used']:.4f} (~{1.0/res['scale_used']:.1f}:1)")
            print(f"Angle Used:       {res['angle_used']:+.2f} deg")
            print(f"Cascade Path:     {res['path_used']}")
            print(f"Runtime:          {res['runtime_ms']:.1f} ms")
            if gt_x is not None and gt_y is not None:
                err = float(np.hypot(res['pred_x'] - gt_x, res['pred_y'] - gt_y))
                print(f"Ground Truth:     ({gt_x:.3f}, {gt_y:.3f}) px")
                print(f"Localization Err: {err:.4f} px ({'PASS <5px' if err < 5.0 else 'FAIL'})")
            print("==================================================")
            print(f"x={res['pred_x']:.4f} y={res['pred_y']:.4f}")
    elif os.path.exists("submission_dataset/manifest.csv"):
        print("No arguments provided — running batch inference on submission_dataset/manifest.csv...")
        run_batch_manifest("submission_dataset/manifest.csv", args.out_csv, weights_path=args.weights)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
