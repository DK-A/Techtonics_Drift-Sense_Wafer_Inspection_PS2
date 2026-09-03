"""
Phase-2 extension of the Drift-Sense sample pipeline.

Upstream (Phase 1, aayushraina21/drift-sense-synthetic-data) fixes three
things that Phase 2 removes:

  1. Zoom ratio is exactly 10x, because reference is 1 nm/px and search is
     10 nm/px. Here the search pixel size is z nm/px with z ~ U[8, 12], so
     the fine canvas is 1000*z px rather than a fixed 10000.
  2. Rotation appears only as an imaging artifact. Here the search raster is
     rotated by theta ~ U[-5, +5] degrees relative to the pattern, and theta
     is part of the ground truth.
  3. The reference crop always comes from the same canvas as the search
     image. Here an "absent" pair draws the reference from an independently
     generated canvas of the same architecture, so no true instance exists.

Geometry is a single affine: canvas -> search. Rotation about the canvas
centre by +theta (CCW), then isotropic scale 1/z, then a translation that
centres the 1000x1000 output. Ground truth is the reference crop's centre
pushed through that same affine, so localization, theta and scale are all
consistent by construction rather than bookkept separately.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from src import sem_imaging
from src.presets import get_preset
from src.patterns.zones import generate_zone_canvas

REFERENCE_SIZE_PX = 1000
SEARCH_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1.0

# Disclosed Phase 2 ranges. Participants may hard-code these.
ZOOM_MIN, ZOOM_MAX = 8.0, 12.0
THETA_MIN, THETA_MAX = -5.0, 5.0

# Keep the reference crop centre away from the search-image border so the
# true instance is never clipped.
GT_MARGIN_PX = 90


@dataclass
class Phase2Params:
    """Upstream GenerationParams plus the Phase 2 pose fields."""
    # --- pose (new in Phase 2) ---
    zoom: float = 10.0
    theta_deg: float = 0.0
    present: bool = True

    # --- upstream imaging / structure knobs ---
    beam_spot_size_nm: float = 5.0
    collapse_threshold_nm: float = 10.0
    dose_reference: float = 2000.0
    dose_search: float = 200.0
    shear_amplitude_px: float = 1.5
    drift_jitter_px: float = 0.5
    detector_noise_sigma_ref: float = 2.0
    detector_noise_sigma_search: float = 5.0
    astigmatism_ratio: float = 1.0
    vignette_strength: float = 0.0
    gamma: float = 1.0
    barrel_distortion_k: float = 0.0
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    speckle_sigma: float = 0.0
    salt_pepper_prob: float = 0.0
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    boundary_bias: float = 0.35
    linewidth_bias_nm: float = 0.0
    corner_rounding_px: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def canvas_size_for(zoom: float, theta_deg: float) -> int:
    """Fine-canvas side length in px (1 nm/px) needed so that the rotated
    search field of view is fully covered with no empty border.

    The search FOV is 1000*zoom canvas px on a side. Rotated by theta, that
    square's axis-aligned bounding box has side 1000*zoom*(|cos|+|sin|).
    """
    fov = SEARCH_SIZE_PX * zoom
    t = np.deg2rad(abs(theta_deg))
    need = fov * (np.cos(t) + np.sin(t))
    return int(np.ceil(need)) + 8


def canvas_to_search_affine(canvas_size: int, zoom: float, theta_deg: float) -> np.ndarray:
    """2x3 affine mapping fine-canvas coords -> search-image coords.

    Rotate by +theta_deg (CCW, y-down image convention) about the canvas
    centre, scale by 1/zoom, then translate so the canvas centre lands at
    the search-image centre.
    """
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    k = 1.0 / zoom
    # CCW rotation in a y-down raster is [[c, s], [-s, c]].
    R = np.array([[c, s], [-s, c]], dtype=np.float64) * k
    cc = (canvas_size - 1) / 2.0
    sc = (SEARCH_SIZE_PX - 1) / 2.0
    t_vec = np.array([sc, sc]) - R @ np.array([cc, cc])
    return np.hstack([R, t_vec.reshape(2, 1)])


def apply_affine_pt(M: np.ndarray, x: float, y: float) -> tuple:
    v = M @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


def invert_affine(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M)


def drift_row_shift(h: int, shear_amplitude_px: float, jitter_std_px: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Per-row horizontal shift used by the raster-drift warp.

    Returned rather than applied internally so the ground-truth point can be
    pushed through the identical shift. Matches upstream apply_raster_drift.
    """
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = (rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0
              else np.zeros(h))
    return (shear + jitter).astype(np.float32)


def apply_drift_with_shift(img: np.ndarray, row_shift: np.ndarray) -> np.ndarray:
    h, w = img.shape
    map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def drift_forward_pt(x: float, y: float, row_shift: np.ndarray) -> tuple:
    """Where a feature at input (x, y) ends up after the drift warp.

    The warp is a backward map: output(x, y) samples input(x + shift[y], y),
    so a feature moves by -shift[y].
    """
    yi = int(np.clip(round(y), 0, len(row_shift) - 1))
    return x - float(row_shift[yi]), y


def barrel_forward_pt(x: float, y: float, k: float, size: int) -> tuple:
    """Forward barrel map for a point (square image, so isotropic).

    apply_barrel_distortion is a backward map with
    r_in = r_out * (1 + k*r_out^2) in normalized radius. Given the feature's
    input radius r_in, solve that cubic for r_out by Newton iteration.
    """
    if k == 0.0:
        return x, y
    c = (size - 1) / 2.0
    nx, ny = (x - c) / c, (y - c) / c
    r_in = float(np.hypot(nx, ny))
    if r_in < 1e-9:
        return x, y
    r = r_in
    for _ in range(40):
        f = r * (1.0 + k * r * r) - r_in
        df = 1.0 + 3.0 * k * r * r
        step = f / df
        r -= step
        if abs(step) < 1e-12:
            break
    ratio = r / r_in
    return c + nx * ratio * c, c + ny * ratio * c


def render_search(canvas: np.ndarray, params: Phase2Params,
                  rng: np.random.Generator) -> np.ndarray:
    """Blur with the shared beam PSF, anti-alias for the z-fold downscale,
    warp canvas -> 1000x1000 search raster, then apply search-side noise.

    Upstream used INTER_AREA for the integer 10x downsample. warpAffine has
    no INTER_AREA, so the z-wide box prefilter below stands in for it; the
    beam PSF alone is not a sufficient low-pass at z=12.
    """
    z = params.zoom
    blurred = sem_imaging.gaussian_psf_blur(
        canvas, params.beam_spot_size_nm, PIXEL_SIZE_REF_NM, params.astigmatism_ratio)
    box = max(2, int(round(z)))
    blurred = cv2.blur(blurred, (box, box))

    M = canvas_to_search_affine(canvas.shape[0], z, params.theta_deg)
    img = cv2.warpAffine(blurred, M, (SEARCH_SIZE_PX, SEARCH_SIZE_PX),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    del blurred

    # Geometric distortions come after the pose affine, so they displace the
    # true feature relative to the affine-derived coordinate. Capture the
    # exact state used here and hand it back, so the ground-truth point can
    # be pushed through the same two maps instead of carrying ~1-2 px of
    # unexplained label error at high severity.
    row_shift = drift_row_shift(SEARCH_SIZE_PX, params.shear_amplitude_px,
                                params.drift_jitter_px, rng)
    img = apply_drift_with_shift(img, row_shift)
    img = sem_imaging.apply_barrel_distortion(img, params.barrel_distortion_k)
    geom = {"row_shift": row_shift, "barrel_k": params.barrel_distortion_k}
    img = sem_imaging.add_shot_noise(img, params.dose_search, rng)
    img = sem_imaging.add_detector_noise(img, params.detector_noise_sigma_search, rng)
    img = sem_imaging.add_speckle_noise(img, params.speckle_sigma, rng)
    img = sem_imaging.add_salt_and_pepper_noise(img, params.salt_pepper_prob, rng)
    img = sem_imaging.apply_vignette(img, params.vignette_strength)
    img = sem_imaging.apply_gamma(img, params.gamma)
    img = sem_imaging.add_charging_streaks(img, params.charging_streak_prob,
                                          params.charging_streak_intensity, rng)
    return img, geom


def _pick_crop_origin(canvas_size: int, M: np.ndarray, params: Phase2Params,
                      rng: np.random.Generator, strip_rects: list) -> tuple:
    """Choose the reference crop origin in canvas coords.

    Sample the target centre in *search* coords first, inside a margin, then
    pull it back through the inverse affine. That guarantees the true
    instance sits well inside the search image regardless of zoom or theta.
    """
    Minv = invert_affine(M)
    lo, hi = GT_MARGIN_PX, SEARCH_SIZE_PX - GT_MARGIN_PX

    # Boundary bias: prefer a mat/strip edge, which is a harder match than
    # deep inside a uniform periodic field.
    if strip_rects and rng.random() < params.boundary_bias:
        sx, sy, sw, sh = strip_rects[int(rng.integers(0, len(strip_rects)))]
        gx, gy = apply_affine_pt(M, sx + sw / 2.0, sy + sh / 2.0)
        gx += rng.uniform(-40, 40)
        gy += rng.uniform(-40, 40)
        gx = float(np.clip(gx, lo, hi))
        gy = float(np.clip(gy, lo, hi))
    else:
        gx = float(rng.uniform(lo, hi))
        gy = float(rng.uniform(lo, hi))

    cx, cy = apply_affine_pt(Minv, gx, gy)
    x0 = int(round(cx - REFERENCE_SIZE_PX / 2.0))
    y0 = int(round(cy - REFERENCE_SIZE_PX / 2.0))
    max_off = canvas_size - REFERENCE_SIZE_PX
    x0 = int(np.clip(x0, 0, max_off))
    y0 = int(np.clip(y0, 0, max_off))
    return x0, y0


def _template_from_reference(ref: np.ndarray, zoom: float, theta_deg: float) -> np.ndarray:
    """What the reference should look like inside the search raster."""
    k = max(2, int(round(zoom)))
    r = cv2.blur(ref, (k, k))
    out = int(round(ref.shape[0] / zoom))
    M = cv2.getRotationMatrix2D(((ref.shape[1] - 1) / 2, (ref.shape[0] - 1) / 2),
                                theta_deg, 1.0 / zoom)
    M[0, 2] += (out - 1) / 2 - (ref.shape[1] - 1) / 2
    M[1, 2] += (out - 1) / 2 - (ref.shape[0] - 1) / 2
    return cv2.warpAffine(r, M, (out, out), flags=cv2.INTER_LINEAR)


def verify_gt_unique(ref: np.ndarray, search: np.ndarray, gt: dict,
                     tol_px: float = 3.0) -> dict:
    """Is the labelled location the one a correct matcher would actually find?

    On a periodic array a uniform-interior crop can correlate better
    somewhere else than at its true origin, which would make the label
    unreproducible no matter how good the algorithm. This checks the global
    correlation peak at the *known* pose lands on the label, and reports the
    margin over the best competing peak outside a local exclusion window.
    """
    tpl = _template_from_reference(ref, gt["scale"], gt["theta"])
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    half = (tpl.shape[0] - 1) / 2.0
    px, py = loc[0] + half, loc[1] + half
    err = float(np.hypot(px - gt["x"], py - gt["y"]))

    # Second-best peak outside an exclusion disc around the winner.
    masked = res.copy()
    r_excl = int(max(tpl.shape[0] * 0.6, 12))
    x0 = max(loc[0] - r_excl, 0); y0 = max(loc[1] - r_excl, 0)
    masked[y0:loc[1] + r_excl, x0:loc[0] + r_excl] = -1.0
    _, second, _, _ = cv2.minMaxLoc(masked)

    return {"ok": err <= tol_px, "err_px": err, "peak": float(peak),
            "second_peak": float(second), "margin": float(peak - second)}


def build_canvas(architecture: str, canvas_size: int, params: Phase2Params,
                 rng: np.random.Generator) -> dict:
    preset = get_preset(architecture)
    return generate_zone_canvas(
        canvas_size, preset["kind"], params.collapse_threshold_nm, rng,
        mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px)


def generate_phase2_sample(architecture: str, params: Phase2Params,
                           rng: np.random.Generator,
                           decoy_architecture: str | None = None,
                           max_crop_attempts: int = 14,
                           good_margin: float = 0.12,
                           min_margin: float = 0.02) -> dict:
    """One Phase 2 pair.

    present=True  -> reference crop is cut from the same canvas the search
                     image is rendered from; ground truth pose is real.
    present=False -> search image comes from canvas A, reference crop from an
                     independently generated canvas B of the same
                     architecture kind. Same look, no true instance.
    """
    canvas_size = canvas_size_for(params.zoom, params.theta_deg)
    M = canvas_to_search_affine(canvas_size, params.zoom, params.theta_deg)

    zone = build_canvas(architecture, canvas_size, params, rng)
    canvas = zone["canvas"]

    # The search image does not depend on which crop becomes the reference,
    # so render it once and reuse it across crop attempts.
    search_img, geom = render_search(canvas, params, rng)

    if params.present:
        attempts = []
        chosen = None
        for attempt in range(max_crop_attempts):
            x0, y0 = _pick_crop_origin(canvas_size, M, params, rng, zone["strip_rects"])
            crop_try = canvas[y0:y0 + REFERENCE_SIZE_PX,
                              x0:x0 + REFERENCE_SIZE_PX].copy()
            gt_cx, gt_cy = apply_affine_pt(
                M, x0 + (REFERENCE_SIZE_PX - 1) / 2.0,
                y0 + (REFERENCE_SIZE_PX - 1) / 2.0)
            gt_cx, gt_cy = drift_forward_pt(gt_cx, gt_cy, geom["row_shift"])
            gt_cx, gt_cy = barrel_forward_pt(gt_cx, gt_cy, geom["barrel_k"],
                                             SEARCH_SIZE_PX)
            gt_try = {"present": 1, "x": gt_cx, "y": gt_cy,
                      "theta": params.theta_deg, "scale": params.zoom}
            ref_try = sem_imaging.image_reference(
                crop_try, pixel_size_nm=PIXEL_SIZE_REF_NM,
                spot_size_nm=params.beam_spot_size_nm,
                dose=params.dose_reference, rng=rng,
                detector_noise_sigma=params.detector_noise_sigma_ref,
                drift_jitter_px=params.drift_jitter_px * 0.2,
                astigmatism_ratio=params.astigmatism_ratio,
                vignette_strength=params.vignette_strength * 0.5,
                gamma=params.gamma,
                barrel_distortion_k=params.barrel_distortion_k * 0.3)
            v = verify_gt_unique(ref_try, search_img, gt_try)
            attempts.append((v, crop_try, gt_try, (x0, y0), ref_try))
            # Comfortable margin -- stop early.
            if v["ok"] and v["margin"] >= good_margin:
                chosen = (crop_try, gt_try, (x0, y0), ref_try, v)
                break
        if chosen is None:
            # Heavy degradation genuinely compresses the correlation margin,
            # so a hard threshold would reject pairs that are merely hard
            # rather than mislabelled. Take the on-label candidate with the
            # widest margin and require only that it clear the floor.
            on_label = [a for a in attempts if a[0]["ok"]]
            if not on_label:
                raise RuntimeError(
                    f"no crop landed on its label after {max_crop_attempts} "
                    f"attempts; best err "
                    f"{min(a[0]['err_px'] for a in attempts):.2f}px")
            best = max(on_label, key=lambda a: a[0]["margin"])
            if best[0]["margin"] < min_margin:
                raise RuntimeError(
                    f"best verifiable margin {best[0]['margin']:.3f} below "
                    f"floor {min_margin}")
            v, crop_try, gt_try, org, ref_try = best
            chosen = (crop_try, gt_try, org, ref_try, v)
        crop, gt, crop_origin, reference_img, verify = chosen
        verify["attempts"] = len(attempts)
        decoy_used = ""
    else:
        # Decoy canvas: same architecture family, independent structure.
        #
        # A uniform periodic crop is the wrong decoy: it matches *somewhere*
        # in any periodic search image, so raw correlation would score an
        # absent pair HIGHER than a true match and reward rejecting
        # confident matches. The reference therefore has to carry
        # large-scale structure the search image does not contain. Two
        # levers: force the crop onto a mat/strip junction, and give the
        # decoy a different zone geometry so its junction spacing cannot be
        # found in the search canvas.
        decoy_arch = decoy_architecture or architecture
        decoy_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        decoy_params = Phase2Params(**{**params.as_dict(),
                                       "mat_size_nm": params.mat_size_nm * 0.55,
                                       "strip_width_nm": params.strip_width_nm * 2.1})
        decoy_size = REFERENCE_SIZE_PX + 1600
        decoy_zone = build_canvas(decoy_arch, decoy_size, decoy_params, decoy_rng)
        dc = decoy_zone["canvas"]
        max_off = decoy_size - REFERENCE_SIZE_PX
        strips = decoy_zone.get("strip_rects") or []
        if strips:
            sx, sy, sw, sh = strips[int(decoy_rng.integers(0, len(strips)))]
            dx = int(np.clip(round(sx + sw / 2.0 - REFERENCE_SIZE_PX / 2.0), 0, max_off))
            dy = int(np.clip(round(sy + sh / 2.0 - REFERENCE_SIZE_PX / 2.0), 0, max_off))
        else:
            dx = int(decoy_rng.integers(0, max_off + 1))
            dy = int(decoy_rng.integers(0, max_off + 1))
        crop = dc[dy:dy + REFERENCE_SIZE_PX, dx:dx + REFERENCE_SIZE_PX].copy()
        del dc, decoy_zone
        gt = {"present": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0}
        crop_origin = (dx, dy)
        decoy_used = decoy_arch
        reference_img = sem_imaging.image_reference(
            crop, pixel_size_nm=PIXEL_SIZE_REF_NM,
            spot_size_nm=params.beam_spot_size_nm,
            dose=params.dose_reference, rng=rng,
            detector_noise_sigma=params.detector_noise_sigma_ref,
            drift_jitter_px=params.drift_jitter_px * 0.2,
            astigmatism_ratio=params.astigmatism_ratio,
            vignette_strength=params.vignette_strength * 0.5,
            gamma=params.gamma,
            barrel_distortion_k=params.barrel_distortion_k * 0.3)
        verify = {"ok": True, "err_px": float("nan"), "peak": float("nan"),
                  "second_peak": float("nan"), "margin": float("nan"), "attempts": 1}

    del canvas, zone

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt": gt,
        "verify": verify,
        "architecture": architecture,
        "decoy_architecture": decoy_used,
        "canvas_size": canvas_size,
        "crop_origin": crop_origin,
        "params": params.as_dict(),
    }


def to_optical_rgb(gray: np.ndarray, rng: np.random.Generator,
                   blur_px: float = 2.2) -> np.ndarray:
    """Crude optical-microscope analogue of an SEM frame, for Set D.

    Optical resolution is far worse than SEM and the response is
    wavelength-dependent, so: soften, then give each channel its own gain
    and a sub-pixel lateral offset (chromatic aberration). Returned BGR for
    cv2.imwrite.
    """
    k = max(3, int(2 * round(3 * blur_px) + 1))
    base = cv2.GaussianBlur(gray, (k, k), blur_px).astype(np.float64)
    gains = (0.78, 1.0, 1.18)          # B, G, R -- blue collects least
    shifts = (-0.7, 0.0, 0.6)          # px of chromatic lateral shift
    chans = []
    for g, sh in zip(gains, shifts):
        M = np.float32([[1, 0, sh], [0, 1, sh * 0.5]])
        c = cv2.warpAffine(base, M, base.shape[::-1], flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        c = c * g + rng.normal(0, 2.0, size=c.shape)
        chans.append(np.clip(c, 0, 255).astype(np.uint8))
    return cv2.merge(chans)
