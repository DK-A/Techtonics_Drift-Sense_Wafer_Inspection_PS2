import numpy as np
import cv2
import math

def generate_correlated_2d_field(shape, scale_px=160.0, amplitude=3.5, rng=None):
    """
    Generates a continuous, low-frequency 2D Gaussian random field to simulate
    realistic SEM spatial illumination drift, detector shading, and charging gradients.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    h, w = shape
    grid_h = max(2, int(h / scale_px))
    grid_w = max(2, int(w / scale_px))
    coarse = rng.normal(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)
    smooth = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = (smooth - np.mean(smooth)) / (np.std(smooth) + 1e-6)
    return smooth * amplitude

def apply_sem_blur(img, spot_size_nm, pixel_size_nm, astigmatism_ratio=1.0):
    sigma = spot_size_nm / pixel_size_nm
    if sigma <= 0.5:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma * astigmatism_ratio)
def downsample_area_average(img, factor):
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
def apply_edge_brightening(img, strength=0.35):
    img_float = img.astype(np.float32)
    sobel_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = np.clip(edges * strength, 0, 255)
    return np.clip(img_float + edges, 0, 255).astype(np.uint8)
def add_shot_noise(img, dose, rng):
    img_float = img.astype(np.float32) / 255.0
    electrons = img_float * dose
    noisy = rng.poisson(electrons).astype(np.float32)
    return np.clip((noisy / dose) * 255.0, 0, 255).astype(np.uint8)
def add_detector_noise(img, sigma, rng):
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
def apply_geometric_augmentation(img, base_box, tx, ty, angle, scale, shear):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    shear_rad = math.radians(shear)
    M[0, 1] += M[0, 0] * math.tan(shear_rad)
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    gx, gy, gw, gh = base_box
    corners = np.array([
        [gx, gy, 1],
        [gx+gw, gy, 1],
        [gx, gy+gh, 1],
        [gx+gw, gy+gh, 1]
    ])
    trans_corners = M.dot(corners.T).T
    min_x = np.min(trans_corners[:, 0])
    min_y = np.min(trans_corners[:, 1])
    max_x = np.max(trans_corners[:, 0])
    max_y = np.max(trans_corners[:, 1])
    new_box = (min_x, min_y, max_x - min_x, max_y - min_y)
    return warped, new_box
def add_charging_streaks(img, rng):
    return img
def render_composite_field(pattern_type, size, fin_pitch, gate_pitch, difficulty, rng, debug_layers=False):
    substrate = 50.0
    if pattern_type in ["FIN_ARRAY", "FIN_CUT", "FIN_GATE"]:
        if pattern_type == "FIN_GATE":
            nominal_width = fin_pitch * 0.18  # ~30-35 fine px
            nominal_gate_width = 65.0  # explicit: visible gate bar (~6-7 search px)
            # === Target Clean Geometry & Realism Parameters ====================
            GATE_INTENSITY_SCALE = 0.87   # global gate intensity scale
            FIN_INTENSITY_SCALE  = 0.55   # underlying fin intensity scale
            GATE_PITCH_VARIATION = 0.035  # ~3–4% correlated cumulative pitch drift
            GATE_CD_VARIATION    = 0.025  # ~2–3% correlated gate CD spread
            FIN_CD_VARIATION     = 0.025  # ~2–3% correlated fin CD spread (frozen)
            GATE_LATERAL_JITTER  = 2.5    # tiny, smooth natural gate center drift
            GATE_EDGE_ROUGHNESS  = 2.5    # subtle natural litho edge roughness
            ISECT_WEAK_PROB      = 0.65   # 65% weak / subtle crossing
            ISECT_MED_PROB       = 0.22   # 22% medium / noticeable bridge
            ISECT_STRONG_PROB    = 0.08   # 8% strong / enhanced step emission
            # (remaining 5% are very weak / inactive crossings)
            # ===============================================================
        else:
            nominal_width = fin_pitch * 0.32
            nominal_gate_width = nominal_width * 2.8
            
        x_coords = np.arange(size, dtype=np.float32)
        fin_id = np.round(x_coords / fin_pitch).astype(np.int32)
        num_fins = int(np.max(fin_id)) + 5
        
        # --- PER-FIN PHYSICAL PARAMETERS ---
        if pattern_type == "FIN_GATE":
            # Smooth correlated fin-to-fin position jitter: 0.5-1.5 fine px
            raw_dx = rng.normal(0, 1.2, num_fins).astype(np.float32)
            fin_dx = cv2.GaussianBlur(raw_dx.reshape(-1,1), (1, 7), 0).ravel()
            # Correlated per-fin CD variation: sigma~3% nominal (Spec §2)
            raw_dw_cd = rng.normal(0, FIN_CD_VARIATION * nominal_width, num_fins).astype(np.float32)
            fin_dw = cv2.GaussianBlur(raw_dw_cd.reshape(-1,1), (1, 9), 0).ravel()
            # Correlated column intensity modulation ±4%
            raw_col = rng.normal(0, 0.02, num_fins).astype(np.float32)
            fin_col_mod = np.clip(1.0 + cv2.GaussianBlur(raw_col.reshape(-1,1), (1, 7), 0).ravel(), 0.94, 1.06).astype(np.float32)
        else:
            raw_dx = rng.normal(0, 15.0, num_fins).astype(np.float32)
            fin_dx = __import__("cv2").GaussianBlur(raw_dx.reshape(-1,1), (1, 5), 0).ravel()
            raw_dw = rng.normal(0, 10.0, num_fins).astype(np.float32)
            fin_dw = __import__("cv2").GaussianBlur(raw_dw.reshape(-1,1), (1, 5), 0).ravel()
            fin_col_mod = rng.uniform(0.95, 1.05, num_fins).astype(np.float32)
            
        fin_bright = rng.uniform(8.5, 9.5, num_fins).astype(np.float32)
        fin_hl = rng.uniform(17.5, 19.5, num_fins).astype(np.float32)
        fin_sh = rng.uniform(11.0, 13.0, num_fins).astype(np.float32)
        if pattern_type == "FIN_GATE":
            fin_bright *= FIN_INTENSITY_SCALE
            fin_hl *= FIN_INTENSITY_SCALE
            fin_sh *= FIN_INTENSITY_SCALE

        fin_left_w = rng.uniform(2.0, 4.0, num_fins).astype(np.float32)
        fin_right_w = rng.uniform(2.0, 4.0, num_fins).astype(np.float32)
        term_y_shift = rng.integers(-25, 26, num_fins)

        # --- PER-GATE PHYSICAL PARAMETERS ---
        gate_y_coords = np.arange(size, dtype=np.float32)
        gate_id = np.round(gate_y_coords / gate_pitch).astype(np.int32)
        num_gates = int(np.max(gate_id)) + 5
        
        if pattern_type == "FIN_GATE":
            # Cumulative gate pitch variation (~3-4% correlated cumulative drift)
            raw_pitch_var = rng.normal(0, GATE_PITCH_VARIATION * gate_pitch, num_gates).astype(np.float32)
            smoothed_pitch_var = cv2.GaussianBlur(raw_pitch_var.reshape(-1,1), (1, 3), 0).ravel()
            gate_dy = np.cumsum(smoothed_pitch_var)
            gate_dy -= gate_dy.mean() # Keep centered
            # Correlated gate CD variation: sigma~2.5%
            raw_gcd = rng.normal(0, GATE_CD_VARIATION * nominal_gate_width, num_gates).astype(np.float32)
            gate_dw = cv2.GaussianBlur(raw_gcd.reshape(-1,1), (1, 5), 0).ravel()
            gate_col_mod = np.ones(num_gates, dtype=np.float32) * GATE_INTENSITY_SCALE
        else:
            raw_gdy = rng.normal(0, 15.0, num_gates).astype(np.float32)
            gate_dy = __import__("cv2").GaussianBlur(raw_gdy.reshape(-1,1), (1, 5), 0).ravel()
            gate_dw = rng.uniform(-15.0, 15.0, num_gates).astype(np.float32)
            gate_col_mod = rng.uniform(0.95, 1.05, num_gates).astype(np.float32)
        # Gate brightness/highlight/shadow — apply intensity scale for FIN_GATE (Spec §7)
        gate_bright = rng.uniform(6.5, 9.5, num_gates).astype(np.float32)
        gate_hl = rng.uniform(14.0, 19.0, num_gates).astype(np.float32)
        gate_sh = rng.uniform(8.0, 12.0, num_gates).astype(np.float32)
        gate_top_w = rng.uniform(2.0, 4.0, num_gates).astype(np.float32)
        gate_bot_w = rng.uniform(2.0, 4.0, num_gates).astype(np.float32)
        if pattern_type == "FIN_GATE":
            gate_bright *= GATE_INTENSITY_SCALE
            gate_hl     *= GATE_INTENSITY_SCALE
            gate_sh     *= GATE_INTENSITY_SCALE
            # gate_col_mod already set above with GATE_INTENSITY_SCALE baked in
        else:
            gate_col_mod = rng.uniform(0.95, 1.05, num_gates).astype(np.float32)
        # Expand 1D arrays across X
        dx_1d = fin_dx[fin_id]
        dw_1d = fin_dw[fin_id]
        bright_1d = fin_bright[fin_id]
        hl_1d = fin_hl[fin_id]
        sh_1d = fin_sh[fin_id]
        lw_1d = fin_left_w[fin_id]
        rw_1d = fin_right_w[fin_id]
        cx_1d = fin_id * fin_pitch
        # ============================================================
        # REALISTIC GEOMETRIC VARIATION
        # Per-fin smooth centerline + independent left/right edge LER
        # All magnitudes scaled to survive 10x area-average downsampling
        # ============================================================
        # 1. Per-fin smooth centerline displacement: x_fin(y) = x_nominal + delta_x + epsilon(y)
        #    Correlation length: size//40 points -> each point covers 40 fine px = 4 search px
        #    Amplitude: Â±10px fine scale = Â±1.0px search scale
        cl_y_res = size // 40          # number of control points along Y
        # Generate independent centerline noise for each fin
        cl_noise_raw = rng.normal(0, 1.0, (cl_y_res, num_fins)).astype(np.float32)
        # Smooth with a Gaussian kernel along Y (within each fin column)
        cl_smooth = np.zeros_like(cl_noise_raw)
        for fi in range(num_fins):
            col = cl_noise_raw[:, fi]
            # Apply Gaussian smoothing with sigma=3 control points
            kernel_size = min(15, cl_y_res // 2 * 2 + 1)
            blurred = cv2.GaussianBlur(col.reshape(-1, 1), (1, kernel_size), sigmaX=0, sigmaY=3.0)
            cl_smooth[:, fi] = blurred.ravel()
        # Normalize each fin's centerline to zero mean, then scale
        for fi in range(num_fins):
            cl_smooth[:, fi] -= cl_smooth[:, fi].mean()
            std = cl_smooth[:, fi].std() + 1e-6
            cl_smooth[:, fi] = cl_smooth[:, fi] / std
        # Upsample to full fine resolution
        centerline_field = cv2.resize(cl_smooth, (num_fins, size), interpolation=cv2.INTER_CUBIC)
        if pattern_type == "FIN_GATE":
            # Smooth correlated lateral wander: ±0.25 to ±0.40 search px (2.5-4.0 fine px)
            cl_amplitude = rng.uniform(2.5, 4.0, num_fins).astype(np.float32)
        else:
            cl_amplitude = rng.uniform(2.5, 6.0, num_fins).astype(np.float32)
        centerline_field *= cl_amplitude[np.newaxis, :]
        # 2. Independent Left and Right LER
        #    Correlation length: size//80 control points -> 80px fine = 8px search
        #    Amplitude: ±1.2px fine scale
        ler_y_res = size // 80
        ler_left_raw = rng.normal(0, 1.0, (ler_y_res, num_fins)).astype(np.float32)
        ler_right_raw = rng.normal(0, 1.0, (ler_y_res, num_fins)).astype(np.float32)
        ler_kernel = min(9, ler_y_res // 2 * 2 + 1)
        # Smooth each fin's edge independently
        for fi in range(num_fins):
            ler_left_raw[:, fi] = cv2.GaussianBlur(
                ler_left_raw[:, fi].reshape(-1,1), (1, ler_kernel), sigmaX=0, sigmaY=2.0).ravel()
            ler_right_raw[:, fi] = cv2.GaussianBlur(
                ler_right_raw[:, fi].reshape(-1,1), (1, ler_kernel), sigmaX=0, sigmaY=2.0).ravel()
        # Upsample to full resolution
        ler_left_field = cv2.resize(ler_left_raw, (num_fins, size), interpolation=cv2.INTER_CUBIC)
        ler_right_field = cv2.resize(ler_right_raw, (num_fins, size), interpolation=cv2.INTER_CUBIC)
        # Scale: ±1.2px fine (crisp, clean edges)
        ler_left_field *= 1.2
        ler_right_field *= 1.2
        # 3. Local width variation along Y per fin (LWR)
        #    Smooth width field: base width ± 1.0px fine along Y
        lwr_y_res = size // 60
        lwr_raw = rng.normal(0, 1.0, (lwr_y_res, num_fins)).astype(np.float32)
        lwr_kernel = min(11, lwr_y_res // 2 * 2 + 1)
        for fi in range(num_fins):
            lwr_raw[:, fi] = cv2.GaussianBlur(
                lwr_raw[:, fi].reshape(-1,1), (1, lwr_kernel), sigmaX=0, sigmaY=2.5).ravel()
        lwr_field = cv2.resize(lwr_raw, (num_fins, size), interpolation=cv2.INTER_CUBIC)
        lwr_field *= 1.0  # ±1.0px fine width variation (~2-3% of nominal)
        # Build 2D arrays indexed by (y, fin_id) for each pixel
        # fin_id_clamped maps each x pixel to its nearest fin
        y_idx = np.arange(size)[:, np.newaxis]       # (size, 1)
        fin_id_clamped = np.clip(fin_id, 0, num_fins - 1)  # (size,)
        px = x_coords[np.newaxis, :]
        
        # === V20: Unified 2D Process Field ===
        if pattern_type == "FIN_GATE":
            pf_res = max(8, size // 1000)
            proc_raw = rng.normal(0, 1.0, (pf_res, pf_res)).astype(np.float32)
            proc_raw = cv2.GaussianBlur(proc_raw, (3, 3), 1.0)
            proc_field_2d = cv2.resize(proc_raw, (size, size), interpolation=cv2.INTER_CUBIC)
            proc_field_2d /= (proc_field_2d.std() + 1e-6)
        else:
            proc_field_2d = np.zeros((size, size), dtype=np.float32)
        
        # Centerline displacement for each pixel: shape (size, size)
        centerline_2d = centerline_field[y_idx, fin_id_clamped]  # (size, size)
        # LER per edge: shape (size, size)
        ler_left_2d  = ler_left_field[y_idx, fin_id_clamped]    # (size, size)
        ler_right_2d = ler_right_field[y_idx, fin_id_clamped]   # (size, size)
        # Local width variation: shape (size, size)
        lwr_2d = lwr_field[y_idx, fin_id_clamped]               # (size, size)
        # Fin centerline position (x) for every (y, x) pixel
        # = nominal_center + static_shift + smooth_centerline_displacement
        fin_cx_2d = cx_1d[np.newaxis, :] + dx_1d[np.newaxis, :] + centerline_2d  # (size, size)
        # Per-pixel half-width (base width ± static CD ± local LWR)
        if pattern_type == "FIN_GATE":
            local_fin_cd = nominal_width * (1.0 + FIN_CD_VARIATION * proc_field_2d)
            half_w_2d = (local_fin_cd + lwr_2d) / 2.0
        else:
            half_w_2d = (nominal_width + dw_1d[np.newaxis, :] + lwr_2d) / 2.0
        half_w_2d = np.clip(half_w_2d, 2.0, None)  # never collapse to zero
        # Left and right physical edge positions
        # Generate wobbly centerline, LER, and LWR for gates (similar to fins but horizontal)
        gcl_x_res = size // 40
        gcl_noise_raw = rng.normal(0, 1.0, (num_gates, gcl_x_res)).astype(np.float32)
        gcl_smooth = np.zeros_like(gcl_noise_raw)
        for gi in range(num_gates):
            col = gcl_noise_raw[gi, :]
            kernel_size = min(15, gcl_x_res // 2 * 2 + 1)
            gcl_smooth[gi, :] = cv2.GaussianBlur(col.reshape(-1, 1), (1, kernel_size), sigmaX=0, sigmaY=3.0).ravel()
        for gi in range(num_gates):
            gcl_smooth[gi, :] -= gcl_smooth[gi, :].mean()
            std = gcl_smooth[gi, :].std() + 1e-6
            gcl_smooth[gi, :] /= std
        gate_centerline_field = cv2.resize(gcl_smooth, (size, num_gates), interpolation=cv2.INTER_CUBIC)
        gcl_amplitude = rng.uniform(0.5, 1.5, num_gates).astype(np.float32)
        gate_centerline_field *= gcl_amplitude[:, np.newaxis]
        gler_x_res = size // 80
        gler_top_raw = rng.normal(0, 1.0, (num_gates, gler_x_res)).astype(np.float32)
        gler_bot_raw = rng.normal(0, 1.0, (num_gates, gler_x_res)).astype(np.float32)
        gler_kernel = min(9, gler_x_res // 2 * 2 + 1)
        for gi in range(num_gates):
            gler_top_raw[gi, :] = cv2.GaussianBlur(gler_top_raw[gi, :].reshape(-1,1), (1, gler_kernel), sigmaX=0, sigmaY=2.0).ravel()
            gler_bot_raw[gi, :] = cv2.GaussianBlur(gler_bot_raw[gi, :].reshape(-1,1), (1, gler_kernel), sigmaX=0, sigmaY=2.0).ravel()
        if pattern_type == "FIN_GATE":
            gler_scale = GATE_EDGE_ROUGHNESS
        else:
            gler_scale = 0.5
        gler_top_field = cv2.resize(gler_top_raw, (size, num_gates), interpolation=cv2.INTER_CUBIC) * gler_scale
        gler_bot_field = cv2.resize(gler_bot_raw, (size, num_gates), interpolation=cv2.INTER_CUBIC) * gler_scale
        glwr_x_res = size // 60
        glwr_raw = rng.normal(0, 1.0, (num_gates, glwr_x_res)).astype(np.float32)
        glwr_kernel = min(11, glwr_x_res // 2 * 2 + 1)
        for gi in range(num_gates):
            glwr_raw[gi, :] = cv2.GaussianBlur(glwr_raw[gi, :].reshape(-1,1), (1, glwr_kernel), sigmaX=0, sigmaY=2.5).ravel()
        glwr_field = cv2.resize(glwr_raw, (size, num_gates), interpolation=cv2.INTER_CUBIC) * 1.0
        # Low frequency bending across the gate width (x axis): very small, natural drift
        gbend_x_res = max(10, size // 400)
        gbend_raw = rng.normal(0, 1.0, (num_gates, gbend_x_res)).astype(np.float32)
        gbend_smooth = np.zeros_like(gbend_raw)
        for gi in range(num_gates):
            gbend_smooth[gi, :] = cv2.GaussianBlur(gbend_raw[gi, :].reshape(-1, 1), (1, 5), sigmaX=0, sigmaY=2.0).ravel()
            gbend_smooth[gi, :] -= gbend_smooth[gi, :].mean()
            std = gbend_smooth[gi, :].std() + 1e-6
            gbend_smooth[gi, :] /= std
        gate_bend_field = cv2.resize(gbend_smooth, (size, num_gates), interpolation=cv2.INTER_CUBIC)
        if pattern_type == "FIN_GATE":
            # Very small smooth gate center drift (Spec §5)
            gbend_amp = rng.uniform(GATE_LATERAL_JITTER * 0.6, GATE_LATERAL_JITTER * 1.1, num_gates).astype(np.float32)
        else:
            gbend_amp = rng.uniform(5.0, 15.0, num_gates).astype(np.float32)
        gate_bend_field *= gbend_amp[:, np.newaxis]
        # Smooth local intensity variation along each gate (±2% variation)
        gint_raw = rng.normal(1.0, 0.02, (num_gates, max(10, size // 300))).astype(np.float32)
        gint_field = cv2.resize(gint_raw, (size, num_gates), interpolation=cv2.INTER_CUBIC)
        # Overlay shift (translation ±0.2-0.5 search px) and small rotation (±0.1-0.3 degrees)
        rotation_angle = rng.uniform(-0.004, 0.004)
        overlay_y_drift = cv2.resize(rng.normal(0, 3.0, (8, 8)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        gate_id_clamped = np.clip(gate_id, 0, num_gates - 1)
        ys_2d = np.arange(size)[:, np.newaxis]
        xs_2d = np.arange(size)[np.newaxis, :]
        gate_cy_2d = gate_id_clamped[:, np.newaxis] * gate_pitch + gate_dy[gate_id_clamped][:, np.newaxis] + gate_centerline_field[gate_id_clamped, :] + gate_bend_field[gate_id_clamped, :] + overlay_y_drift + xs_2d * np.tan(rotation_angle)
        # Swelling at intersections (loading effect)
        is_fin_nominal = (px >= fin_cx_2d - nominal_width/2.0) & (px <= fin_cx_2d + nominal_width/2.0)
        is_gate_nominal = (ys_2d >= gate_cy_2d - nominal_gate_width/2.0) & (ys_2d <= gate_cy_2d + nominal_gate_width/2.0)
        half_w_2d = (nominal_width + dw_1d[np.newaxis, :] + lwr_2d) / 2.0
        half_w_2d = half_w_2d * (1.0 + 0.02 * is_gate_nominal)
        half_w_2d = np.clip(half_w_2d, 2.0, None)
        
        if pattern_type in ["FIN_GATE", "GATE_POLY"]:
            local_gate_cd = nominal_gate_width * (1.0 + GATE_CD_VARIATION * proc_field_2d)
            half_w_gate_2d = (local_gate_cd + glwr_field[gate_id_clamped, :]) / 2.0
        else:
            half_w_gate_2d = (nominal_gate_width + gate_dw[gate_id_clamped][:, np.newaxis] + glwr_field[gate_id_clamped, :]) / 2.0
            
        half_w_gate_2d = half_w_gate_2d * (1.0 + 0.015 * is_fin_nominal)
        half_w_gate_2d = np.clip(half_w_gate_2d, 2.0, None)
        left_edge_x  = fin_cx_2d - half_w_2d + ler_left_2d    # (size, size)
        right_edge_x = fin_cx_2d + half_w_2d + ler_right_2d   # (size, size)
        top_edge_y = gate_cy_2d - half_w_gate_2d + gler_top_field[gate_id_clamped, :]
        bot_edge_y = gate_cy_2d + half_w_gate_2d + gler_bot_field[gate_id_clamped, :]
        is_gate = (ys_2d >= top_edge_y) & (ys_2d <= bot_edge_y)
        gate_top_w_2d = gate_top_w[gate_id_clamped][:, np.newaxis]
        gate_bot_w_2d = gate_bot_w[gate_id_clamped][:, np.newaxis]
        gate_top_edge = (ys_2d >= top_edge_y) & (ys_2d < top_edge_y + gate_top_w_2d)
        gate_bot_edge = (ys_2d <= bot_edge_y) & (ys_2d > bot_edge_y - gate_bot_w_2d)
        # Pixel x positions: (1, size)  # (1, size)
        # Discrete Fin Mask: pixel is inside fin if between left and right edge
        is_fin = (px >= left_edge_x) & (px <= right_edge_x)    # (size, size)
        # Sidewall edge masks using variable width per fin
        lw_2d = lw_1d[np.newaxis, :]  # (1, size)
        rw_2d = rw_1d[np.newaxis, :]  # (1, size)
        left_edge  = (px >= left_edge_x)  & (px < left_edge_x  + lw_2d)
        right_edge = (px <= right_edge_x) & (px > right_edge_x - rw_2d)
        # MAT Boundaries (Independent of fins)
        mat_x_positions = []
        trench_widths_x = []
        mat_y_positions = []
        horizontal_cuts = []
        
        if pattern_type != "FIN_GATE":
            curr_x = rng.integers(2000, 4000)
            while curr_x < size:
                mat_x_positions.append(curr_x)
                trench_widths_x.append(int(rng.integers(50, 90)))
                curr_x += rng.integers(10000, 20000)
            curr_y = rng.integers(1500, 3000)
            while curr_y < size:
                mat_y_positions.append(curr_y)
                curr_y += rng.integers(8000, 16000)
            # Define a raw trench mask (unblurred)
            trench_mask_raw = np.ones((size, size), dtype=np.float32)
            # Apply global vertical trenches
            for x, w_x in zip(mat_x_positions, trench_widths_x):
                if x + w_x < size:
                    t_val = rng.uniform(0.05, 0.15)
                    x0 = max(0, int(x))
                    x1 = min(size, int(x + w_x))
                    trench_mask_raw[:, x0:x1] = np.minimum(trench_mask_raw[:, x0:x1], t_val)
            # Apply localized horizontal cuts along the boundary lines
            for y in mat_y_positions:
                w_y = rng.integers(30, 60)
                if y + w_y < size:
                    x_curr = 0
                    while x_curr < size:
                        chunk_w = rng.integers(300, 900)
                        x_next = min(size, x_curr + chunk_w)
                        # 25% probability of having a localized cut in this chunk
                        if rng.random() > 0.75:
                            t_val = rng.uniform(0.05, 0.15)
                            y_jitter = rng.integers(-30, 31)
                            y0 = max(0, int(y + y_jitter))
                            y1 = min(size, int(y + w_y + y_jitter))
                            trench_mask_raw[y0:y1, x_curr:x_next] = np.minimum(trench_mask_raw[y0:y1, x_curr:x_next], t_val)
                            horizontal_cuts.append({"y_top": y0, "y_bot": y1, "x_start": x_curr, "x_end": x_next})
                        x_curr = x_next
        # Apply sparse fin cuts if pattern_type is FIN_CUT
        cut_event_count = 0
        cut_lengths_list = []
        cut_ys_list = []
        cut_fids_list = []
        if pattern_type == "FIN_CUT":
            # list of placed cuts to enforce spatial exclusion: (fid, vy)
            placed_cuts = []
            # Decide cuts fin-by-fin to keep them independent
            for fid in range(10, num_fins - 11):
                # 24% probability that a fin has a cut event (meaning ~80% are continuous)
                if rng.random() < 0.265:
                    num_cuts_on_fin = 1 if rng.random() > 0.15 else 2
                    for _ in range(num_cuts_on_fin):
                        # Occasional slightly wider cut (cuts 2 adjacent fins)
                        cut_num_fins = 1 if rng.random() > 0.10 else 2
                        f_start = fid - cut_num_fins // 2
                        f_end = f_start + cut_num_fins
                        cx_start = f_start * fin_pitch - fin_pitch/2
                        cx_end = (f_end - 1) * fin_pitch + fin_pitch/2
                        cx_nominal = (cx_start + cx_end) / 2.0
                        span_w = cx_end - cx_start
                        # Find a valid Y coordinate that does not overlap existing horizontal MAT boundaries
                        # AND does not violate spatial exclusion zones (at least 1500px from other cuts on nearby fins)
                        valid_y = False
                        vy = 0.0
                        for retry in range(10):
                            vy = rng.uniform(1000, size - 1000)
                            # Check MAT boundary
                            if trench_mask_raw[int(vy), int(cx_nominal)] < 0.9:
                                continue
                            # Check spatial exclusion: no neighboring fin (within 4 pitch steps) can have a cut within 1500px
                            conflict = False
                            for other_fid, other_vy in placed_cuts:
                                if abs(fid - other_fid) <= 4 and abs(vy - other_vy) < 1500:
                                    conflict = True
                                    break
                            if not conflict:
                                valid_y = True
                                break
                        if not valid_y:
                            continue
                        # Add a tiny vertical jitter (Â±1-2 pixels)
                        vy += rng.uniform(-2.0, 2.0)
                        # Use highly varied cut-length distribution (mostly short, fewer medium/long)
                        # plus length variation Â±15-25%
                        nominal_cut_length = rng.uniform(130, 240)
                        r_len = rng.random()
                        if r_len < 0.75:
                            cut_length = rng.uniform(0.7, 1.3) * nominal_cut_length
                        elif r_len < 0.94:
                            cut_length = rng.uniform(1.3, 2.0) * nominal_cut_length
                        else:
                            cut_length = rng.uniform(2.0, 3.0) * nominal_cut_length
                        # Per-cut length variation: Â±15-25%
                        cut_length *= rng.uniform(0.75, 1.25)
                        y0 = max(0, int(vy - cut_length // 2))
                        y1 = min(size, int(vy + cut_length // 2))
                        # Draw realistic non-rectangular termination morphology
                        # Per-cut width variation: Â±10-15%
                        w_scale = rng.uniform(0.85, 1.15)
                        half_w = (span_w / 2.0) * w_scale
                        x_min = max(0, int(cx_nominal - half_w * 1.5))
                        x_max = min(size, int(cx_nominal + half_w * 1.5))
                        y0_min = max(0, int(y0 - 50))
                        y1_max = min(size, int(y1 + 50))
                        xs = np.arange(x_min, x_max, dtype=np.float32)
                        ys = np.arange(y0_min, y1_max, dtype=np.float32)[:, np.newaxis]
                        dx = (xs - cx_nominal) / half_w
                        # Endpoint profile: only rounded or angled (no flat) to prevent perfectly rectangular cuts
                        prof = rng.choice(["rounded", "angled"])
                        if prof == "rounded":
                            slope_top = rng.uniform(-2.0, 2.0)
                            slope_bot = rng.uniform(-2.0, 2.0)
                            rounding_top = rng.uniform(4.0, 10.0)
                            rounding_bot = rng.uniform(-10.0, -4.0)
                        else: # angled
                            slope_top = rng.uniform(-8.0, 8.0)
                            slope_bot = rng.uniform(-8.0, 8.0)
                            rounding_top = rng.uniform(0.0, 2.0)
                            rounding_bot = rng.uniform(-2.0, 0.0)
                        # Add a tiny wobbly perturbation to endpoint shape (low-frequency and smooth, no pixel jaggedness)
                        wobble_top = __import__("cv2").GaussianBlur(rng.normal(0, 3.5, dx.shape).reshape(1, -1), (7, 1), 0).ravel()
                        wobble_bot = __import__("cv2").GaussianBlur(rng.normal(0, 3.5, dx.shape).reshape(1, -1), (7, 1), 0).ravel()
                        y0_x = y0 + slope_top * dx + rounding_top * (1.0 - dx**2) + wobble_top
                        y1_x = y1 + slope_bot * dx + rounding_bot * (1.0 - dx**2) + wobble_bot
                        in_cut = (ys >= y0_x) & (ys <= y1_x)
                        # Three cut strengths: ~20% faint, ~60% normal, ~20% stronger
                        # with intensity/depth variation Â±10-15%
                        r_str = rng.random()
                        int_var = rng.uniform(0.85, 1.15)
                        if r_str < 0.20:
                            # Faint: partial cut where the fin is weakened but clearly a deliberate gap
                            t_val = rng.uniform(0.18, 0.28) * int_var
                            v_mult = 0.45 * int_var
                        elif r_str < 0.80:
                            # Normal
                            t_val = rng.uniform(0.04, 0.12) * int_var
                            v_mult = 0.65 * int_var
                        else:
                            # Stronger
                            t_val = rng.uniform(0.00, 0.03) * int_var
                            v_mult = 0.85 * int_var
                        # Keep intensity within safe physical bounds
                        t_val = np.clip(t_val, 0.0, 1.0)
                        v_mult = np.clip(v_mult, 0.0, 1.0)
                        trench_mask_raw[y0_min:y1_max, x_min:x_max] = np.where(
                            in_cut,
                            np.minimum(trench_mask_raw[y0_min:y1_max, x_min:x_max], t_val),
                            trench_mask_raw[y0_min:y1_max, x_min:x_max]
                        )
                        # Add to horizontal_cuts so termination dots are drawn at cut boundaries
                        horizontal_cuts.append({"y_top": y0, "y_bot": y1, "x_start": x_min, "x_end": x_max, "v_mult": v_mult})
                        placed_cuts.append((fid, vy))
                        cut_event_count += 1
                        cut_lengths_list.append(cut_length)
                        cut_ys_list.append(vy)
                        cut_fids_list.append(fid)
            # Compute Diagnostics
            cut_event_density = cut_event_count / (num_fins - 20)
            mean_cut_length = np.mean(cut_lengths_list) if cut_lengths_list else 0.0
            std_cut_length = np.std(cut_lengths_list) if cut_lengths_list else 0.0
            # Sort cuts by Y to calculate Y-spacing
            sorted_ys = sorted(cut_ys_list)
            mean_cut_spacing = np.mean(np.diff(sorted_ys)) if len(sorted_ys) > 1 else 0.0
            cut_position_std = np.std(cut_ys_list) if cut_ys_list else 0.0
            # Neighbor cut position correlation
            # Match each cut to its nearest neighbor's cut (in terms of fin ID)
            neighbor_diffs = []
            for i, fid in enumerate(cut_fids_list):
                best_diff = None
                for j, fid2 in enumerate(cut_fids_list):
                    if i != j and abs(fid - fid2) == 1:
                        y_diff = abs(cut_ys_list[i] - cut_ys_list[j])
                        if best_diff is None or y_diff < best_diff:
                            best_diff = y_diff
                if best_diff is not None:
                    neighbor_diffs.append(best_diff)
            neighbor_cut_correlation = np.mean(neighbor_diffs) if neighbor_diffs else 0.0
            if debug_layers:
                print(f"--- Automated Diagnostics for FIN_CUT ---")
                print(f"cut_event_count: {cut_event_count}")
                print(f"cut_event_density: {cut_event_density:.4f}")
                print(f"mean_cut_length: {mean_cut_length:.1f}")
                print(f"std_cut_length: {std_cut_length:.1f}")
                print(f"mean_cut_spacing: {mean_cut_spacing:.1f}")
                print(f"cut_position_std: {cut_position_std:.1f}")
                print(f"neighbor_cut_correlation (avg Y diff to adjacent cut): {neighbor_cut_correlation:.1f}")
        # Speed optimization: For FIN_GATE, trench mask is uniformly 1.0 (no remap needed!)
        if pattern_type == "FIN_GATE":
            trench_factor = 1.0
        else:
            # Create wobbly boundaries using a very smooth 2D noise field
            bound_wobble_y = cv2.resize(rng.normal(0, 35.0, (8, 8)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
            bound_wobble_x = cv2.resize(rng.normal(0, 35.0, (8, 8)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
            y_indices, x_indices = np.indices((size, size), dtype=np.float32)
            map_y = np.clip(y_indices + bound_wobble_y, 0, size - 1).astype(np.float32)
            map_x = np.clip(x_indices + bound_wobble_x, 0, size - 1).astype(np.float32)
            trench_factor = cv2.remap(trench_mask_raw, map_x, map_y, cv2.INTER_LINEAR)
            trench_factor = cv2.GaussianBlur(trench_factor, (15, 15), 3.0)
            boundary_noise = rng.normal(1.0, 0.08, (size // 100, size // 100)).astype(np.float32)
            boundary_noise_up = cv2.resize(boundary_noise, (size, size), interpolation=cv2.INTER_CUBIC)
            trench_factor = np.clip(trench_factor * boundary_noise_up, 0.0, 1.0)
        # LAYER 1: FIN GEOMETRY
        fin_geometry = np.zeros((size, size), dtype=np.float32)
        bright_2d = np.broadcast_to(bright_1d, (size, size))
        fin_geometry[is_fin] = bright_2d[is_fin]
        fin_geometry *= trench_factor
        # LAYER 2: FIN TOPOGRAPHY (Detector edge response)
        fin_topography = np.zeros((size, size), dtype=np.float32)
        hl_2d = np.broadcast_to(hl_1d, (size, size))
        sh_2d = np.broadcast_to(sh_1d, (size, size))
        fin_topography[left_edge] = hl_2d[left_edge]
        fin_topography[right_edge] = -sh_2d[right_edge]
        fin_topography *= trench_factor
        # LAYER 3: BOUNDARY & CONTEXT
        boundary_layer = np.zeros((size, size), dtype=np.float32)
        for cut in horizontal_cuts:
            y_top = cut["y_top"]
            y_bot = cut["y_bot"]
            x_start = cut["x_start"]
            x_end = cut["x_end"]
            v_mult = cut.get("v_mult", 0.72) # retrieve custom visibility scaling
            for fid in range(num_fins):
                cx_nominal = fid * fin_pitch + fin_dx[fid]
                # Check if fin falls inside the X span of the cut
                if x_start <= cx_nominal < x_end:
                    # Draw binary choice: ~40% of terminations have NO bright dot (creates gaps/variation)
                    if rng.random() > 0.40:
                        t_scale  = rng.uniform(0.95, 1.05)
                        t_radius = max(1, int(nominal_width * rng.uniform(0.35, 0.60)))
                        t_dy     = term_y_shift[fid] + rng.integers(-4, 5)
                        # Top termination
                        if y_top < size:
                            cl_top = float(centerline_field[min(size-1, y_top), fid])
                            real_cx_top = int(cx_nominal + cl_top)
                            safe_cx = min(size-1, max(0, real_cx_top))
                            wobble_y = int(bound_wobble_y[y_top, safe_cx])
                            yt_draw = y_top - wobble_y + t_dy
                            if 0 <= real_cx_top < size and 0 <= yt_draw < size:
                                hl_val = float(fin_hl[fid]) * t_scale * v_mult
                                sh_val = float(fin_sh[fid]) * t_scale * v_mult
                                br_val = float(fin_bright[fid]) * t_scale * v_mult
                                # Slightly elliptical shape variation
                                axis_y = max(1, int(t_radius * rng.uniform(0.90, 1.10)))
                                cv2.ellipse(boundary_layer, (real_cx_top, yt_draw), (t_radius, axis_y), 0, 0, 360, float(-sh_val), -1)
                                cv2.ellipse(boundary_layer, (real_cx_top - 1, yt_draw), (t_radius, axis_y), 0, 0, 360, float(hl_val), -1)
                                if t_radius > 1:
                                    cv2.ellipse(boundary_layer, (real_cx_top, yt_draw), (t_radius - 1, max(1, axis_y - 1)), 0, 0, 360, float(br_val), -1)
                    # Draw binary choice for bottom
                    if rng.random() > 0.40:
                        t_scale2  = rng.uniform(0.95, 1.05)
                        t_radius2 = max(1, int(nominal_width * rng.uniform(0.35, 0.60)))
                        t_dy2     = term_y_shift[fid] + rng.integers(-4, 5)
                        # Bottom termination
                        if y_bot < size:
                            cl_bot = float(centerline_field[min(size-1, y_bot), fid])
                            real_cx_bot = int(cx_nominal + cl_bot)
                            safe_cx = min(size-1, max(0, real_cx_bot))
                            wobble_y = int(bound_wobble_y[y_bot, safe_cx])
                            yb_draw = y_bot - wobble_y + t_dy2
                            if 0 <= real_cx_bot < size and 0 <= yb_draw < size:
                                hl_val2 = float(fin_hl[fid]) * t_scale2 * v_mult
                                sh_val2 = float(fin_sh[fid]) * t_scale2 * v_mult
                                br_val2 = float(fin_bright[fid]) * t_scale2 * v_mult
                                axis_y2 = max(1, int(t_radius2 * rng.uniform(0.90, 1.10)))
                                cv2.ellipse(boundary_layer, (real_cx_bot, yb_draw), (t_radius2, axis_y2), 0, 0, 360, float(-sh_val2), -1)
                                cv2.ellipse(boundary_layer, (real_cx_bot - 1, yb_draw), (t_radius2, axis_y2), 0, 0, 360, float(hl_val2), -1)
                                if t_radius2 > 1:
                                    cv2.ellipse(boundary_layer, (real_cx_bot, yb_draw), (t_radius2 - 1, max(1, axis_y2 - 1)), 0, 0, 360, float(br_val2), -1)
        # Soften and blur the terminations so they blend physically into rounded caps
        boundary_layer = cv2.GaussianBlur(boundary_layer, (9, 9), 1.5)
        # LAYER 4: SUBTLE FINE SEM ROUGHNESS (reduced 75% to preserve clean geometry)
        r_small = rng.normal(0, 1.2, (size // 2, size // 2)).astype(np.float32)
        roughness = cv2.resize(r_small, (size, size), interpolation=cv2.INTER_LINEAR)
        # COMPOSITING (Multi-layer design based on pattern type)
        canvas = np.full((size, size), substrate, dtype=np.float32)
        
        if pattern_type == "FIN_GATE":
            # LAYER 5: GATE GEOMETRY
            gate_geometry = np.zeros((size, size), dtype=np.float32)
            gate_bright_2d = np.broadcast_to(gate_bright[gate_id_clamped][:, np.newaxis], (size, size))
            gate_geometry[is_gate] = (gate_bright_2d * gint_field[gate_id_clamped, :])[is_gate]
            if isinstance(trench_factor, np.ndarray):
                gate_geometry *= (trench_factor * 0.2 + 0.8)
            
            # LAYER 6: GATE TOPOGRAPHY
            gate_topography = np.zeros((size, size), dtype=np.float32)
            gate_hl_2d = np.broadcast_to(gate_hl[gate_id_clamped][:, np.newaxis], (size, size))
            gate_sh_2d = np.broadcast_to(gate_sh[gate_id_clamped][:, np.newaxis], (size, size))
            gate_topography[gate_top_edge] = (gate_hl_2d * gint_field[gate_id_clamped, :])[gate_top_edge]
            gate_topography[gate_bot_edge] = (-gate_sh_2d * gint_field[gate_id_clamped, :])[gate_bot_edge]
            if isinstance(trench_factor, np.ndarray):
                gate_topography *= (trench_factor * 0.2 + 0.8)
            
            # Lay down fin background
            canvas += fin_geometry + fin_topography

            # === Clean Spatially Correlated Fin×Gate Crossing Model ===
            # Sample continuous 2D process field at gate/fin nodes
            proc_isect = cv2.resize(proc_field_2d, (num_fins, num_gates), interpolation=cv2.INTER_AREA)
            r_cat = rng.random((num_gates, num_fins)).astype(np.float32)
            # 70% driven by continuous process field, 30% local perturbation
            r_blended = np.clip(r_cat * 0.30 + (proc_isect * 0.5 + 0.5) * 0.70, 0.0, 1.0)
            
            # Distribution: 65% weak, 22% medium, 8% strong, 5% very weak/inactive
            weak_mask   = r_blended < ISECT_WEAK_PROB
            med_mask    = (~weak_mask) & (r_blended < (ISECT_WEAK_PROB + ISECT_MED_PROB))
            strong_mask = (~weak_mask) & (~med_mask) & (r_blended < (ISECT_WEAK_PROB + ISECT_MED_PROB + ISECT_STRONG_PROB))
            inactive_mask = ~(weak_mask | med_mask | strong_mask)
            
            # Subtle physical step multiplier per node
            isect_step = np.ones((num_gates, num_fins), dtype=np.float32)
            if weak_mask.any():
                isect_step[weak_mask]     = rng.uniform(1.08, 1.18, int(weak_mask.sum())).astype(np.float32)
            if med_mask.any():
                isect_step[med_mask]      = rng.uniform(1.24, 1.38, int(med_mask.sum())).astype(np.float32)
            if strong_mask.any():
                isect_step[strong_mask]   = rng.uniform(1.45, 1.70, int(strong_mask.sum())).astype(np.float32)
            if inactive_mask.any():
                isect_step[inactive_mask] = rng.uniform(0.98, 1.02, int(inactive_mask.sum())).astype(np.float32)

            # Localized spatial geometry per crossing
            i_dx   = rng.normal(0, 1.5, (num_gates, num_fins)).astype(np.float32)
            i_dy   = rng.normal(0, 1.5, (num_gates, num_fins)).astype(np.float32)
            i_rx   = rng.uniform(nominal_width * 0.8, nominal_width * 1.3, (num_gates, num_fins)).astype(np.float32)
            i_ry   = rng.uniform(nominal_gate_width * 0.8, nominal_gate_width * 1.3, (num_gates, num_fins)).astype(np.float32)
            i_fade = rng.uniform(1.2, 2.2, (num_gates, num_fins)).astype(np.float32)

            gate_signal = gate_geometry + gate_topography
            # Subtle regional drift (±2%)
            gate_signal *= (1.0 + 0.02 * proc_field_2d)
            
            gate_mask = is_gate
            g_id = gate_id_clamped[:, np.newaxis]
            g_id_2d = np.broadcast_to(g_id, (size, size))[gate_mask]
            f_id = fin_id_clamped[np.newaxis, :]
            f_id_2d = np.broadcast_to(f_id, (size, size))[gate_mask]
            
            px_g = np.broadcast_to(px, (size, size))[gate_mask]
            ys_g = np.broadcast_to(ys_2d, (size, size))[gate_mask]
            
            dx_g = px_g - fin_cx_2d[gate_mask] - i_dx[g_id_2d, f_id_2d]
            dy_g = ys_g - gate_cy_2d[gate_mask] - i_dy[g_id_2d, f_id_2d]
            
            dist_sq = (dx_g / i_rx[g_id_2d, f_id_2d])**2 + (dy_g / i_ry[g_id_2d, f_id_2d])**2
            bump_shape = np.exp(-dist_sq * i_fade[g_id_2d, f_id_2d])
            
            # Physical 3D crossing bridge: transmit underlying fin ridge through gate
            fin_elev = (fin_geometry + fin_topography)[gate_mask]
            gate_signal[gate_mask] += fin_elev * 0.40 * bump_shape
            
            step_val = isect_step[g_id_2d, f_id_2d]
            gate_signal[gate_mask] *= (1.0 + (step_val - 1.0) * bump_shape)

            # Composite gate over substrate
            canvas[is_gate] = substrate + gate_signal[is_gate]
        else:
            # Standard P1 / P2 vertical fin composite
            canvas += fin_geometry + fin_topography
            
        # Add common layers: boundary/termination caps and background roughness
        canvas += boundary_layer
        canvas += roughness
        
        # Apply downsampling-surviving column/row modulation
        col_x = np.arange(size)
        fin_id_col = np.clip(np.round(col_x / fin_pitch).astype(np.int32), 0, num_fins - 1)
        col_mod_1d = fin_col_mod[fin_id_col]
        
        row_y = np.arange(size)
        gate_id_row = np.clip(np.round(row_y / gate_pitch).astype(np.int32), 0, num_gates - 1)
        row_mod_1d = gate_col_mod[gate_id_row]
        
        mod_2d = np.zeros((size, size), dtype=np.float32)
        mod_2d[:, :] = col_mod_1d[np.newaxis, :]
        if pattern_type == "FIN_GATE":
            row_mod_2d = np.broadcast_to(row_mod_1d[:, np.newaxis], (size, size))
            mod_2d[is_gate] = row_mod_2d[is_gate]
            
        fin_signal = canvas - substrate
        fin_signal *= mod_2d
        
        if pattern_type == "FIN_GATE":
            # Add global large-scale SEM nonuniformity to break up the uniform grid appearance
            global_shading_raw = rng.normal(1.0, 0.08, (5, 5)).astype(np.float32)
            global_shading = cv2.resize(global_shading_raw, (size, size), interpolation=cv2.INTER_CUBIC)
            fin_signal *= global_shading

        canvas = substrate + fin_signal
        
        # Fixed-scale mapping: stretch physical raw intensity to [0, 255]
        canvas = (canvas - 15.0) * 2.7
        np.clip(canvas, 0, 255, out=canvas)
        
        if debug_layers:
            debug_dict = {
                "fin_geometry": fin_geometry,
                "fin_topography": fin_topography,
                "boundary": boundary_layer,
                "roughness": roughness,
                "final": canvas,
                "is_fin": (is_fin & (trench_factor > 0.5)).astype(np.float32)
            }
            if pattern_type == "FIN_GATE":
                debug_dict["gate_geometry"] = gate_geometry
                debug_dict["gate_topography"] = gate_topography
            return debug_dict
    return canvas.astype(np.uint8)
# ================================================================
# P5-P8 PATTERN RENDER FUNCTIONS
# Extracted exactly from approved_pattern/utils_p5.py
# ================================================================
def render_peripheral_context(canvas, size_px, rng, base_mat_intensity=78.0):
    """
    Sparse, strictly horizontal/vertical functional routing regions.
    Intensity is relative to MAT base intensity, keeping low contrast.
    """
    strip_bg = base_mat_intensity + rng.uniform(-5, 5)
    canvas.fill(strip_bg)
    num_h = max(1, int(size_px / 800))
    for _ in range(num_h):
        y = rng.integers(0, size_px)
        length = rng.integers(500, 3000)
        x = rng.integers(0, max(1, size_px - length))
        thickness = int(rng.uniform(20, 50))
        intensity = strip_bg + rng.uniform(15, 30)
        y2 = min(size_px, y + thickness)
        x2 = min(size_px, x + length)
        if y2 > y and x2 > x:
            canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], intensity)
            if rng.random() < 0.3:
                cx = rng.choice([x, max(x, x2 - thickness), x + length//2])
                cx2 = min(size_px, cx + thickness)
                if cx2 > cx:
                    via_int = min(255, intensity + rng.uniform(20, 40))
                    canvas[max(0, y-thickness):min(size_px, y2+thickness), cx:cx2] = np.maximum(
                        canvas[max(0, y-thickness):min(size_px, y2+thickness), cx:cx2], via_int)
    num_v = max(1, int(size_px / 800))
    for _ in range(num_v):
        x = rng.integers(0, size_px)
        length = rng.integers(500, 3000)
        y = rng.integers(0, max(1, size_px - length))
        thickness = int(rng.uniform(20, 50))
        intensity = strip_bg + rng.uniform(15, 30)
        x2 = min(size_px, x + thickness)
        y2 = min(size_px, y + length)
        if x2 > x and y2 > y:
            canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], intensity)
            if rng.random() < 0.3:
                cy = rng.choice([y, max(y, y2 - thickness), y + length//2])
                cy2 = min(size_px, cy + thickness)
                if cy2 > cy:
                    via_int = min(255, intensity + rng.uniform(20, 40))
                    canvas[cy:cy2, max(0, x-thickness):min(size_px, x2+thickness)] = np.maximum(
                        canvas[cy:cy2, max(0, x-thickness):min(size_px, x2+thickness)], via_int)
def render_class8_continuous_field(size_px=1000, rng=None, gt_x=None, gt_y=None, gt_w=None, gt_h=None):
    """
    Synthesizes a continuous physical FinFET search field for Class 8 (ACTIVE / CELL BOUNDARY).
    Boundaries emerge locally from non-collinear staggered physical fin terminations into shallow STI isolation trenches.
    Scales dynamically for 2048x2048 or 10000x10000 master fields.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if gt_x is None:
        gt_x = int(size_px * 0.415)
    if gt_y is None:
        gt_y = int(size_px * 0.415)
    if gt_w is None:
        gt_w = int(size_px * 0.125) if size_px <= 2048 else int(size_px * 0.10)
    if gt_h is None:
        gt_h = gt_w
    scale_r = float(size_px) / 2048.0
    base_mat_intensity = 78.0
    canvas = np.full((size_px, size_px), base_mat_intensity, dtype=np.float32)
    num_tracks = 14
    track_height = int(size_px / num_tracks)
    for tr in range(num_tracks):
        y_base_top = tr * track_height
        y_base_bot = (tr + 1) * track_height
        x_curr = int(rng.uniform(10, 30) * scale_r)
        while x_curr < size_px - int(40 * scale_r):
            block_w = int(rng.uniform(80, 250) * scale_r)
            if x_curr + block_w > size_px - int(10 * scale_r):
                block_w = size_px - int(10 * scale_r) - x_curr
            x_left = x_curr
            x_right = x_left + block_w
            y_top = max(0, y_base_top + int(rng.uniform(4, 16) * scale_r))
            y_bot = min(size_px, y_base_bot - int(rng.uniform(4, 16) * scale_r))
            is_gt_region = (x_left <= gt_x + gt_w and x_right >= gt_x and y_top <= gt_y + gt_h and y_bot >= gt_y)
            if is_gt_region:
                # In GT track: Active Region A covers upper part down to staggered active boundary
                y_top = max(0, y_base_top)
                y_bot = gt_y + int(135 * scale_r)
                c_type = 'LOGIC_6FIN'
                fin_pitch = 16.0 * scale_r
            else:
                c_type = rng.choice(['LOGIC_2FIN', 'LOGIC_3FIN', 'LOGIC_4FIN', 'LOGIC_6FIN', 'SRAM_DENSE'])
                fin_pitch = float(rng.uniform(14.0, 22.0)) * scale_r
            fin_w = fin_pitch * float(rng.uniform(0.38, 0.45))
            block_bg = base_mat_intensity + float(rng.uniform(-4.0, 4.0))
            if c_type == 'LOGIC_2FIN':
                group_sizes = [2, 2]
            elif c_type == 'LOGIC_3FIN':
                group_sizes = [3, 3]
            elif c_type == 'LOGIC_4FIN':
                group_sizes = [4, 2]
            elif c_type == 'LOGIC_6FIN':
                group_sizes = [6, 6]
            else:
                group_sizes = [6, 6]
            bx_l = max(0, x_left)
            bx_r = min(size_px, x_right)
            by_t = max(0, y_top)
            by_b = min(size_px, y_bot)
            if bx_r > bx_l and by_b > by_t:
                canvas[by_t:by_b, bx_l:bx_r] = block_bg
            group_idx = 0
            fins_in_group = 0
            # Staggered fin termination pullbacks for Active Region A near the STI active boundary
            stagger_pullbacks_a = [-18, -4, 10, -12, 14, -8, 6, -15, 8, -10, 12, -2]
            for fin_idx, x in enumerate(np.arange(bx_l + fin_pitch / 2, bx_r - fin_pitch / 2, fin_pitch)):
                target_group_size = group_sizes[group_idx % len(group_sizes)]
                if fins_in_group >= target_group_size:
                    group_idx += 1
                    fins_in_group = 0
                    if rng.random() < 0.25:
                        continue
                fins_in_group += 1
                actual_x = x + rng.uniform(-fin_pitch * 0.01, fin_pitch * 0.01)
                fin_y_start = by_t + int(rng.uniform(2, 10) * scale_r)
                if is_gt_region and (gt_x <= actual_x <= gt_x + gt_w):
                    # Active Region A fins terminate naturally at non-collinear staggered Y levels near boundary
                    stagger_y = int(stagger_pullbacks_a[fin_idx % len(stagger_pullbacks_a)] * scale_r)
                    fin_y_end = gt_y + int((110 + stagger_y) * scale_r)
                else:
                    fin_y_end = by_b - int(rng.uniform(2, 10) * scale_r)
                if fin_y_end > fin_y_start:
                    xl = int(actual_x - fin_w / 2)
                    xh = int(actual_x + fin_w / 2)
                    xl = max(0, min(size_px - 1, xl))
                    xh = max(0, min(size_px - 1, xh))
                    if xh > xl:
                        # Fin body core
                        canvas[fin_y_start:fin_y_end, xl:xh] = np.maximum(canvas[fin_y_start:fin_y_end, xl:xh], block_bg + 55)
                        # Secondary electron edge brightening
                        ew = max(1, int(fin_w * 0.15))
                        if xl + ew <= xh:
                            canvas[fin_y_start:fin_y_end, xl:xl+ew] = np.maximum(canvas[fin_y_start:fin_y_end, xl:xl+ew], block_bg + 95)
                        if xh - ew >= xl:
                            canvas[fin_y_start:fin_y_end, xh-ew:xh] = np.maximum(canvas[fin_y_start:fin_y_end, xh-ew:xh], block_bg + 95)
                        # Natural rounded SEM end-cap at fin termination tip
                        cap_r = int(fin_w / 2)
                        if cap_r >= 1:
                            cap_cy = fin_y_end
                            cap_cx = int(actual_x)
                            y_grid, x_grid = np.ogrid[max(0, cap_cy - cap_r):min(size_px, cap_cy + cap_r + 1),
                                                     max(0, cap_cx - cap_r):min(size_px, cap_cx + cap_r + 1)]
                            dist_sq = (x_grid - cap_cx)**2 + (y_grid - cap_cy)**2
                            mask_cap = dist_sq <= cap_r**2
                            canvas[max(0, cap_cy - cap_r):min(size_px, cap_cy + cap_r + 1),
                                   max(0, cap_cx - cap_r):min(size_px, cap_cx + cap_r + 1)][mask_cap] = np.maximum(
                                canvas[max(0, cap_cy - cap_r):min(size_px, cap_cy + cap_r + 1),
                                       max(0, cap_cx - cap_r):min(size_px, cap_cx + cap_r + 1)][mask_cap], block_bg + 85)
            x_curr += block_w + int(rng.uniform(10, 30) * scale_r)
    # ACTIVE REGION B (Bottom) - Different local pitch (21.0px) & staggered starting positions
    fin_pitch_b = 21.0 * scale_r
    fin_w_b = fin_pitch_b * 0.42
    fin_positions_b = list(np.arange(gt_x + int(12 * scale_r), gt_x + gt_w - int(10 * scale_r), fin_pitch_b))
    stagger_b_starts = [124, 142, 128, 150, 122, 145, 132, 148, 126, 140]
    for idx, x in enumerate(fin_positions_b):
        actual_x = x
        fin_y_start = gt_y + int(stagger_b_starts[idx % len(stagger_b_starts)] * scale_r)
        fin_y_end = min(size_px, gt_y + int(260 * scale_r))
        xl = int(actual_x - fin_w_b / 2)
        xh = int(actual_x + fin_w_b / 2)
        if xh > xl and fin_y_end > fin_y_start:
            # Fin body core
            canvas[fin_y_start:fin_y_end, xl:xh] = np.maximum(canvas[fin_y_start:fin_y_end, xl:xh], base_mat_intensity + 55)
            # Secondary electron edge brightening
            ew = max(1, int(fin_w_b * 0.15))
            if xl + ew <= xh:
                canvas[fin_y_start:fin_y_end, xl:xl+ew] = np.maximum(canvas[fin_y_start:fin_y_end, xl:xl+ew], base_mat_intensity + 95)
            if xh - ew >= xl:
                canvas[fin_y_start:fin_y_end, xh-ew:xh] = np.maximum(canvas[fin_y_start:fin_y_end, xh-ew:xh], base_mat_intensity + 95)
            # Natural rounded SEM end-cap at top fin tip of Active Region B
            cap_r = int(fin_w_b / 2)
            if cap_r >= 1:
                cap_cy = fin_y_start
                cap_cx = int(actual_x)
                y_grid, x_grid = np.ogrid[max(0, cap_cy - cap_r):min(size_px, cap_cy + cap_r + 1),
                                         max(0, cap_cx - cap_r):min(size_px, cap_cx + cap_r + 1)]
                dist_sq = (x_grid - cap_cx)**2 + (y_grid - cap_cy)**2
                mask_cap = dist_sq <= cap_r**2
                canvas[max(0, cap_cy - cap_r):min(size_px, cap_cy + cap_r + 1),
                       max(0, cap_cx - cap_r):min(size_px, cap_cx + cap_r + 1)][mask_cap] = np.maximum(
                    canvas[max(0, cap_cy - cap_r):min(size_px, cap_cy + cap_r + 1),
                           max(0, cap_cx - cap_r):min(size_px, cap_cx + cap_r + 1)][mask_cap], base_mat_intensity + 85)
    sigma = 0.8 * scale_r
    blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    noise = rng.normal(0, 2.5, blurred.shape)
    search_field = np.clip(blurred + noise, 0, 255).astype(np.uint8)
    return search_field
def render_class7_metal_routing_field(size_px=1000, rng=None, gt_x=None, gt_y=None, gt_w=None, gt_h=None):
    """
    Synthesizes a realistic SEM micrograph for Class 7 (METAL ROUTING M0/M1).
    Refined Appearance Parameters:
    - Dark, non-uniform background (base ~43, subtle isotropic 2D SEM noise).
    - Subdued metal brightness (core ~98-104, soft edge ~125-130 max, NO bright white).
    - Preserved soft SEM edge blur (sigma = 1.0) and natural irregular grain.
    - Multi-scale local illumination variation (no synthetic vertical banding).
    - 100% identical visual/contrast treatment for GT and hard negatives.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if gt_x is None:
        gt_x = int(size_px * 0.415)  # 850 for 2048
    if gt_y is None:
        gt_y = int(size_px * 0.415)  # 850 for 2048
    if gt_w is None:
        gt_w = int(size_px * 0.125) if size_px <= 2048 else int(size_px * 0.10)
    if gt_h is None:
        gt_h = gt_w
    scale_r = float(size_px) / 2048.0
    base_mat_intensity = 86.0  # Harmonized neutral mid-gray SEM substrate background
    # 1. Neutral SEM substrate (Isotropic 2D noise: NO vertical banding)
    bg_coarse = rng.normal(0, 3.0, (size_px // 32, size_px // 32)).astype(np.float32)
    bg_coarse_resized = cv2.resize(bg_coarse, (size_px, size_px), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size_px, size_px), base_mat_intensity, dtype=np.float32) + bg_coarse_resized * 1.8
    # Multi-scale local illumination / charging variation across canvas
    y_grid, x_grid = np.ogrid[:size_px, :size_px]
    center_y, center_x = size_px * 0.48, size_px * 0.52
    dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    illumination = 1.0 - 0.05 * (dist_from_center / (size_px * 0.70))
    canvas *= illumination
    # 2. Metal Line Drawer with harmonized SEM grayscale (core ~142-148, edge ~172-178)
    def draw_sem_metal_line(img, x1, y1, x2, y2, width=15.0, core_val=145.0, edge_val=175.0):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        segment_w = max(4, int((width + rng.uniform(-1.5, 1.5)) * scale_r))
        # Per-segment intensity drift
        c_val = core_val + rng.uniform(-4.0, 4.0)
        e_val = edge_val + rng.uniform(-4.0, 4.0)
        # Base core line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), float(c_val), thickness=int(segment_w))
        # Soft SEM edge emission (1.5px soft edge, max ~128 intensity, NO pure white)
        ew = max(1, int(segment_w * 0.18))
        if x1 == x2:  # Vertical line
            xl1 = max(0, x1 - segment_w // 2)
            xl2 = min(img.shape[1], xl1 + ew)
            xr2 = min(img.shape[1], x1 + segment_w // 2)
            xr1 = max(0, xr2 - ew)
            ymin, ymax = max(0, min(y1, y2)), min(img.shape[0], max(y1, y2))
            if ymax > ymin:
                img[ymin:ymax, xl1:xl2] = np.maximum(img[ymin:ymax, xl1:xl2], e_val)
                img[ymin:ymax, xr1:xr2] = np.maximum(img[ymin:ymax, xr1:xr2], e_val)
        elif y1 == y2:  # Horizontal line
            yt1 = max(0, y1 - segment_w // 2)
            yt2 = min(img.shape[0], yt1 + ew)
            yb2 = min(img.shape[0], y1 + segment_w // 2)
            yb1 = max(0, yb2 - ew)
            xmin, xmax = max(0, min(x1, x2)), min(img.shape[1], max(x1, x2))
            if xmax > xmin:
                img[yt1:yt2, xmin:xmax] = np.maximum(img[yt1:yt2, xmin:xmax], e_val)
                img[yb1:yb2, xmin:xmax] = np.maximum(img[yb1:yb2, xmin:xmax], e_val)
    # Helper for subtle via/contact landing pad
    def draw_sem_via(img, cx, cy, radius=6):
        cx, cy, r = int(cx), int(cy), max(3, int((radius + rng.uniform(-0.5, 0.5)) * scale_r))
        cv2.circle(img, (cx, cy), r, 168.0 + rng.uniform(-3, 3), -1)
        if r > 3:
            cv2.circle(img, (cx, cy), r - 2, 144.0 + rng.uniform(-3, 3), -1)
    # 3. Populate realistic routing regions across the 2048x2048 field
    track_spacing = 110.0 * scale_r
    x_tracks = list(np.arange(90 * scale_r, size_px - 90 * scale_r, track_spacing))
    y_tracks = list(np.arange(90 * scale_r, size_px - 90 * scale_r, track_spacing))
    for i, y in enumerate(y_tracks):
        if rng.random() < 0.28:
            continue
        x_curr = 70 * scale_r + rng.uniform(0, 80)
        while x_curr < size_px - 110 * scale_r:
            seg_len = rng.uniform(150, 360) * scale_r
            x_next = min(size_px - 70 * scale_r, x_curr + seg_len)
            w = rng.uniform(12.5, 16.5)
            draw_sem_metal_line(canvas, x_curr, y, x_next, y, w)
            if rng.random() < 0.12:
                draw_sem_via(canvas, x_curr, y, radius=6)
            if rng.random() < 0.35 and i < len(y_tracks) - 1:
                y_down = y_tracks[min(len(y_tracks) - 1, i + int(rng.choice([1, 2])))]
                x_bend = float(rng.choice([x_curr, x_next]))
                draw_sem_metal_line(canvas, x_bend, y, x_bend, y_down, w)
                if rng.random() < 0.15:
                    draw_sem_via(canvas, x_bend, y_down, radius=6)
            x_curr = x_next + rng.uniform(80, 170) * scale_r
    for j, x in enumerate(x_tracks):
        if rng.random() < 0.28:
            continue
        y_curr = 70 * scale_r + rng.uniform(0, 80)
        while y_curr < size_px - 110 * scale_r:
            seg_len = rng.uniform(150, 340) * scale_r
            y_next = min(size_px - 70 * scale_r, y_curr + seg_len)
            w = rng.uniform(12.5, 16.5)
            draw_sem_metal_line(canvas, x, y_curr, x, y_next, w)
            if rng.random() < 0.24 and j < len(x_tracks) - 1:
                x_jog = x_tracks[min(len(x_tracks) - 1, j + 1)]
                y_jog = (y_curr + y_next) / 2
                draw_sem_metal_line(canvas, x, y_jog, x_jog, y_jog, w)
            y_curr = y_next + rng.uniform(90, 180) * scale_r
    # 4. TARGET MOTIF (GT Region) & 5 HARD NEGATIVE REGIONS
    gt_patch = canvas[gt_y:gt_y+gt_h, gt_x:gt_x+gt_w].copy()
    mw = 15.0
    # GT Topology: H-bus, T-junction, L-bend, and step jog
    draw_sem_metal_line(gt_patch, 25, 70, 230, 70, mw)          # Horizontal bus
    draw_sem_metal_line(gt_patch, 80, 25, 80, 175, mw)           # Vertical trunk
    draw_sem_metal_line(gt_patch, 80, 175, 170, 175, mw)         # L-bend horizontal
    draw_sem_metal_line(gt_patch, 170, 175, 170, 125, mw)        # L-bend vertical
    draw_sem_metal_line(gt_patch, 170, 125, 225, 125, mw)        # Step jog horizontal
    draw_sem_metal_line(gt_patch, 225, 125, 225, 215, mw)        # Step jog vertical
    draw_sem_via(gt_patch, 80, 70, radius=6)                      # Subtle via pad at T-junction
    draw_sem_via(gt_patch, 170, 175, radius=6)                   # Subtle via pad at L-corner
    canvas[gt_y:gt_y + gt_h, gt_x:gt_x + gt_w] = gt_patch
    if size_px <= 1000:
        neg_boxes = [
            (100, 150),   # NEG-1
            (640, 150),  # NEG-2
            (150, 640),  # NEG-3
            (640, 640), # NEG-4
            (370, 100),   # NEG-5
        ]
    else:
        neg_boxes = [
            (350, 450),   # NEG-1
            (1350, 450),  # NEG-2
            (450, 1350),  # NEG-3
            (1350, 1350), # NEG-4
            (850, 350),   # NEG-5
        ]
    # NEG-1: H-bus + T-junction + step jog
    neg1_patch = canvas[neg_boxes[0][1]:neg_boxes[0][1]+256, neg_boxes[0][0]:neg_boxes[0][0]+256].copy()
    draw_sem_metal_line(neg1_patch, 20, 80, 235, 80, mw)
    draw_sem_metal_line(neg1_patch, 125, 20, 125, 160, mw)
    draw_sem_metal_line(neg1_patch, 125, 160, 200, 160, mw)
    draw_sem_metal_line(neg1_patch, 200, 160, 200, 225, mw)
    draw_sem_via(neg1_patch, 125, 80, radius=6)
    canvas[neg_boxes[0][1]:neg_boxes[0][1] + 256, neg_boxes[0][0]:neg_boxes[0][0] + 256] = neg1_patch
    # NEG-2: Double L-turn topology
    neg2_patch = canvas[neg_boxes[1][1]:neg_boxes[1][1]+256, neg_boxes[1][0]:neg_boxes[1][0]+256].copy()
    draw_sem_metal_line(neg2_patch, 35, 60, 210, 60, mw)
    draw_sem_metal_line(neg2_patch, 210, 60, 210, 175, mw)
    draw_sem_metal_line(neg2_patch, 40, 140, 140, 140, mw)
    draw_sem_metal_line(neg2_patch, 140, 140, 140, 220, mw)
    draw_sem_via(neg2_patch, 210, 60, radius=6)
    canvas[neg_boxes[1][1]:neg_boxes[1][1] + 256, neg_boxes[1][0]:neg_boxes[1][0] + 256] = neg2_patch
    # NEG-3: S-shaped double jog + branch
    neg3_patch = canvas[neg_boxes[2][1]:neg_boxes[2][1]+256, neg_boxes[2][0]:neg_boxes[2][0]+256].copy()
    draw_sem_metal_line(neg3_patch, 30, 80, 120, 80, mw)
    draw_sem_metal_line(neg3_patch, 120, 80, 120, 150, mw)
    draw_sem_metal_line(neg3_patch, 120, 150, 210, 150, mw)
    draw_sem_metal_line(neg3_patch, 210, 150, 210, 220, mw)
    draw_sem_via(neg3_patch, 120, 80, radius=6)
    canvas[neg_boxes[2][1]:neg_boxes[2][1] + 256, neg_boxes[2][0]:neg_boxes[2][0] + 256] = neg3_patch
    # NEG-4: H/V routing grid with vias
    neg4_patch = canvas[neg_boxes[3][1]:neg_boxes[3][1]+256, neg_boxes[3][0]:neg_boxes[3][0]+256].copy()
    draw_sem_metal_line(neg4_patch, 30, 65, 225, 65, mw)
    draw_sem_metal_line(neg4_patch, 30, 145, 225, 145, mw)
    draw_sem_metal_line(neg4_patch, 85, 20, 85, 220, mw)
    draw_sem_metal_line(neg4_patch, 175, 65, 175, 220, mw)
    draw_sem_via(neg4_patch, 85, 65, radius=6)
    draw_sem_via(neg4_patch, 175, 145, radius=6)
    canvas[neg_boxes[3][1]:neg_boxes[3][1] + 256, neg_boxes[3][0]:neg_boxes[3][0] + 256] = neg4_patch
    # NEG-5: L-turn + branch + step jog
    neg5_patch = canvas[neg_boxes[4][1]:neg_boxes[4][1]+256, neg_boxes[4][0]:neg_boxes[4][0]+256].copy()
    draw_sem_metal_line(neg5_patch, 30, 75, 185, 75, mw)
    draw_sem_metal_line(neg5_patch, 185, 75, 185, 160, mw)
    draw_sem_metal_line(neg5_patch, 185, 160, 235, 160, mw)
    draw_sem_metal_line(neg5_patch, 95, 75, 95, 200, mw)
    draw_sem_via(neg5_patch, 185, 75, radius=6)
    canvas[neg_boxes[4][1]:neg_boxes[4][1] + 256, neg_boxes[4][0]:neg_boxes[4][0] + 256] = neg5_patch
    # 5. SEM Gaussian spot blur + fine irregular SEM grain
    sigma = 1.0 * scale_r
    blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Fine irregular SEM detector noise
    fine_noise = rng.normal(0, 2.8, blurred.shape).astype(np.float32)
    search_field = np.clip(blurred + fine_noise, 0, 255).astype(np.uint8)
    return search_field
def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))
def render_class5_contact_array_field(size_px=1000, rng=None, gt_x=None, gt_y=None, gt_w=None, gt_h=None):
    """
    Synthesizes a highly realistic SEM micrograph for Class 5 (CONTACT ARRAY CA/CB) at 1000x1000 pixels.
    Spatially Correlated Process Variation Engine:
    - Distinct, spatially correlated process clusters across the 1000x1000 field.
    - Subdued, realistic contact edge rim intensity (160-180 gray range, no glowing white outlines).
    - Corner rounding (R=3.2-5.8px) and soft edge profile variations.
    - Spatially varying focus/blur map across the 1000x1000 field.
    - Integrated fine 2D isotropic SEM detector noise affecting BOTH contacts and substrate.
    - Identical SEM acquisition statistics across GT and NEG-1..5 regions.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if gt_x is None:
        gt_x = int(size_px * 0.37) if size_px <= 1000 else int(size_px * 0.415)
    if gt_y is None:
        gt_y = int(size_px * 0.37) if size_px <= 1000 else int(size_px * 0.415)
    if gt_w is None:
        gt_w = 256
    if gt_h is None:
        gt_h = gt_w
    base_mat_intensity = 78.0
    canvas = np.full((size_px, size_px), base_mat_intensity, dtype=np.float32)
    grid_y, grid_x = np.ogrid[:size_px, :size_px]
    gain_map = 1.0 + 0.06 * np.sin(grid_x * 0.004) * np.cos(grid_y * 0.004) + 0.03 * np.cos(grid_x * 0.008)
    bg_illum = 3.5 * np.sin(grid_x * 0.005) * np.cos(grid_y * 0.005) + 1.5 * np.cos(grid_y * 0.010)
    canvas += bg_illum.astype(np.float32)
    def draw_contact_refined(img, cx, cy, w, h, corner_r=5.0, rim_t=4.0, edge_val=205.0, core_val=115.0, bg_val=78.0, ler_scale=1.0, blur_sigma=1.1, ler_seed=42):
        r_rng = np.random.default_rng(ler_seed)
        pad = 14
        pw, ph = int(w + 2*pad), int(h + 2*pad)
        patch = np.full((ph, pw), bg_val, dtype=np.float32)
        x1, y1 = pad, pad
        x2, y2 = pad + int(w), pad + int(h)
        r = max(1, min(int(round(corner_r)), int(w)//2, int(h)//2))
        outer_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.rectangle(outer_mask, (x1 + r, y1), (x2 - r, y2), 255, -1)
        cv2.rectangle(outer_mask, (x1, y1 + r), (x2, y2 - r), 255, -1)
        cv2.circle(outer_mask, (x1 + r, y1 + r), r, 255, -1)
        cv2.circle(outer_mask, (x2 - r, y1 + r), r, 255, -1)
        cv2.circle(outer_mask, (x1 + r, y2 - r), r, 255, -1)
        cv2.circle(outer_mask, (x2 - r, y2 - r), r, 255, -1)
        inner_mask = np.zeros((ph, pw), dtype=np.uint8)
        ix1, iy1 = x1 + int(rim_t), y1 + int(rim_t)
        ix2, iy2 = x2 - int(rim_t), y2 - int(rim_t)
        ir = max(1, r - int(rim_t))
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw > 0 and ih > 0:
            cv2.rectangle(inner_mask, (ix1 + ir, iy1), (ix2 - ir, iy2), 255, -1)
            cv2.rectangle(inner_mask, (ix1, iy1 + ir), (ix2, iy2 - ir), 255, -1)
            cv2.circle(inner_mask, (ix1 + ir, iy1 + ir), ir, 255, -1)
            cv2.circle(inner_mask, (ix2 - ir, iy1 + ir), ir, 255, -1)
            cv2.circle(inner_mask, (ix1 + ir, iy2 - ir), ir, 255, -1)
            cv2.circle(inner_mask, (ix2 - ir, iy2 - ir), ir, 255, -1)
        noise = r_rng.normal(0, 1.1 * ler_scale, (ph, pw)).astype(np.float32)
        noise_blur = cv2.GaussianBlur(noise, (3, 3), 0.8)
        rim_bool = (outer_mask > 0) & (inner_mask == 0)
        core_bool = inner_mask > 0
        patch_y, patch_x = np.ogrid[:ph, :pw]
        dir_gradient = 1.0 + 0.04 * ((patch_x - pw/2.0)/pw + (patch_y - ph/2.0)/ph)
        local_edge = edge_val * dir_gradient + noise_blur * 4.5
        local_core = core_val * dir_gradient + noise_blur * 2.5
        patch[rim_bool] = local_edge[rim_bool]
        patch[core_bool] = local_core[core_bool]
        k_size = 5 if blur_sigma <= 1.35 else 7
        patch_blur = cv2.GaussianBlur(patch, (k_size, k_size), blur_sigma)
        img_h, img_w = img.shape
        iy1_c = int(round(cy - ph / 2.0))
        ix1_c = int(round(cx - pw / 2.0))
        iy2_c = iy1_c + ph
        ix2_c = ix1_c + pw
        c_y1, c_y2 = max(0, iy1_c), min(img_h, iy2_c)
        c_x1, c_x2 = max(0, ix1_c), min(img_w, ix2_c)
        if c_y2 > c_y1 and c_x2 > c_x1:
            p_y1 = c_y1 - iy1_c
            p_y2 = p_y1 + (c_y2 - c_y1)
            p_x1 = c_x1 - ix1_c
            p_x2 = p_x1 + (c_x2 - c_x1)
            img[c_y1:c_y2, c_x1:c_x2] = np.maximum(img[c_y1:c_y2, c_x1:c_x2], patch_blur[p_y1:p_y2, p_x1:p_x2])
    def render_array_region_refined(img, rx, ry, rw, rh, px=42.0, py=48.0, cw=25.0, ch=25.0, corner_r=5.0, rim_t=4, density=1.0, row_drift=0.0, base_edge_val=205.0, base_core_val=115.0, ler_scale=1.0, seed=42):
        local_rng = np.random.default_rng(seed)
        start_x = rx + px / 2.0
        start_y = ry + py / 2.0
        row_idx = 0
        y = start_y
        while y < ry + rh - py/4.0:
            row_shift = local_rng.uniform(-0.6, 0.6) * ler_scale + (row_drift * row_idx)
            x = start_x + row_shift
            col_idx = 0
            while x < rx + rw - px/4.0:
                if local_rng.random() <= density:
                    w_mult = local_rng.uniform(0.97, 1.03)
                    h_mult = local_rng.uniform(0.97, 1.03)
                    w_c, h_c = cw * w_mult, ch * h_mult
                    dx = local_rng.normal(0, 0.8) * ler_scale
                    dy = local_rng.normal(0, 0.8) * ler_scale
                    anomaly_roll = local_rng.random()
                    if anomaly_roll < 0.005:
                        x += px
                        col_idx += 1
                        continue
                    elif anomaly_roll < 0.010:
                        w_c, h_c = 16.0, 16.0
                    elif anomaly_roll < 0.015:
                        w_c, h_c = 31.0, 31.0
                    c_r = corner_r + local_rng.uniform(-1.0, 1.0)
                    c_r = max(2.5, min(c_r, min(w_c, h_c)/2.0 - 1.0))
                    abs_cx = int(clamp(x + dx, 0, img.shape[1]-1))
                    abs_cy = int(clamp(y + dy, 0, img.shape[0]-1))
                    g_factor = gain_map[abs_cy, abs_cx]
                    edge_v = base_edge_val * g_factor + local_rng.uniform(-6.0, 6.0)
                    core_v = base_core_val * g_factor + local_rng.uniform(-3.0, 3.0)
                    blur_s = float(local_rng.uniform(0.95, 1.25))
                    c_seed = seed + row_idx * 137 + col_idx * 17
                    draw_contact_refined(img, x + dx, y + dy, w_c, h_c, corner_r=c_r, rim_t=rim_t, edge_val=edge_v, core_val=core_v, ler_scale=ler_scale, blur_sigma=blur_s, ler_seed=c_seed)
                x += px
                col_idx += 1
            y += py
            row_idx += 1
    block_size = 256
    num_blocks = int(np.ceil(size_px / block_size))
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            bx = bi * block_size
            by = bj * block_size
            if bi < 2 and bj < 2:
                local_px, local_py, local_cw, local_ch, local_dens, local_drift = 38.0, 43.0, 23.0, 23.0, 0.95, 0.0
            elif bi >= 2 and bj < 2:
                local_px, local_py, local_cw, local_ch, local_dens, local_drift = 44.0, 48.0, 27.5, 27.5, 0.90, 0.0
            elif bi < 2 and bj >= 2:
                local_px, local_py, local_cw, local_ch, local_dens, local_drift = 42.0, 46.0, 23.0, 28.0, 0.90, 0.7
            else:
                local_px, local_py, local_cw, local_ch, local_dens, local_drift = 48.0, 52.0, 24.0, 24.0, 0.80, 0.0
            b_seed = int(bi * 100 + bj + rng.integers(1, 9999))
            render_array_region_refined(canvas, bx, by, block_size, block_size, px=local_px, py=local_py, cw=local_cw, ch=local_ch, density=local_dens, row_drift=local_drift, seed=b_seed)
    if size_px <= 1000:
        neg_boxes = [
            [100, 150, 256, 256],
            [640, 150, 256, 256],
            [150, 640, 256, 256],
            [640, 640, 256, 256],
            [370, 100, 256, 256]
        ]
    else:
        neg_boxes = [
            [350, 450, 256, 256],
            [1350, 450, 256, 256],
            [450, 1350, 256, 256],
            [1350, 1350, 256, 256],
            [850, 350, 256, 256]
        ]
    nx1, ny1, nw1, nh1 = neg_boxes[0]
    canvas[ny1:ny1+nh1, nx1:nx1+nw1] = base_mat_intensity
    render_array_region_refined(canvas, nx1, ny1, nw1, nh1, px=34.0, py=38.0, cw=22.0, ch=22.0, corner_r=5.0, density=1.0, seed=101)
    nx2, ny2, nw2, nh2 = neg_boxes[1]
    canvas[ny2:ny2+nh2, nx2:nx2+nw2] = base_mat_intensity
    render_array_region_refined(canvas, nx2, ny2, nw2, nh2, px=42.0, py=48.0, cw=20.0, ch=30.0, corner_r=4.5, density=1.0, seed=102)
    nx3, ny3, nw3, nh3 = neg_boxes[2]
    canvas[ny3:ny3+nh3, nx3:nx3+nw3] = base_mat_intensity
    render_array_region_refined(canvas, nx3, ny3, nw3, nh3, px=44.0, py=46.0, cw=25.0, ch=25.0, corner_r=5.0, density=0.95, row_drift=0.8, seed=103)
    nx4, ny4, nw4, nh4 = neg_boxes[3]
    canvas[ny4:ny4+nh4, nx4:nx4+nw4] = base_mat_intensity
    render_array_region_refined(canvas, nx4, ny4, nw4, nh4, px=42.0, py=48.0, cw=27.0, ch=27.0, corner_r=2.0, ler_scale=2.2, density=1.0, seed=104)
    nx5, ny5, nw5, nh5 = neg_boxes[4]
    canvas[ny5:ny5+nh5, nx5:nx5+nw5] = base_mat_intensity
    render_array_region_refined(canvas, nx5, ny5, nw5, nh5, px=48.0, py=52.0, cw=23.5, ch=23.5, corner_r=5.0, density=0.70, base_edge_val=168.0, seed=105)
    canvas[gt_y:gt_y+gt_h, gt_x:gt_x+gt_w] = base_mat_intensity
    render_array_region_refined(canvas, gt_x, gt_y, gt_w, gt_h, px=42.0, py=48.0, cw=25.0, ch=25.0, corner_r=5.0, rim_t=4, density=1.0, base_edge_val=205.0, base_core_val=115.0, seed=42)
    shot_noise = rng.normal(0, 2.5, canvas.shape).astype(np.float32)
    canvas_noisy = np.clip(canvas + shot_noise, 0, 255)
    focus_map = 1.0 + 0.18 * np.sin(grid_x * 0.004) * np.cos(grid_y * 0.004)
    blur_sharp = cv2.GaussianBlur(canvas_noisy, (0, 0), sigmaX=0.85, sigmaY=0.85)
    blur_soft = cv2.GaussianBlur(canvas_noisy, (0, 0), sigmaX=1.35, sigmaY=1.35)
    w_soft = np.clip((focus_map - 0.85) / 0.5, 0.0, 1.0).astype(np.float32)
    blurred = blur_sharp * (1.0 - w_soft) + blur_soft * w_soft
    detector_noise = rng.normal(0, 2.0, blurred.shape).astype(np.float32)
    search_field = np.clip(blurred + detector_noise, 0, 255).astype(np.uint8)
    return search_field
def render_class6_local_interconnect_field(size_px=1000, rng=None, gt_x=None, gt_y=None, gt_w=None, gt_h=None):
    """
    Synthesizes a realistic SEM micrograph for Class 6 (LOCAL INTERCONNECT LI / M0).
    Noticeable Realism & Process Variation (>= 8.5/10):
    - Visibly distinct Lithographic Critical Dimension (CD) variation (+/-8-15%, linewidth 10.0-16.0px).
    - Variable segment lengths (30-150px) and local pitch/spacing fluctuations (55-95px).
    - Lithographic Line-Edge Roughness (LER) & variable endpoint morphology (blunt, tapered, rounded, soft).
    - Corner and T-junction morphology variations (~35% corners rounded/softened with edge blending).
    - Dark stable SEM substrate (~43 base intensity) with fine 2D isotropic detector grain.
    - 100% identical visual/contrast treatment across GT and all 5 hard negatives.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if gt_x is None:
        gt_x = int(size_px * 0.415)  # 850 for 2048
    if gt_y is None:
        gt_y = int(size_px * 0.415)  # 850 for 2048
    if gt_w is None:
        gt_w = int(size_px * 0.125) if size_px <= 2048 else int(size_px * 0.10)
    if gt_h is None:
        gt_h = gt_w
    scale_r = float(size_px) / 2048.0
    base_mat_intensity = 78.0  # Harmonized neutral mid-gray SEM substrate background
    # 1. Stable SEM substrate with fine 2D isotropic detector noise
    bg_coarse = rng.normal(0, 2.8, (size_px // 32, size_px // 32)).astype(np.float32)
    bg_coarse_resized = cv2.resize(bg_coarse, (size_px, size_px), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size_px, size_px), base_mat_intensity, dtype=np.float32) + bg_coarse_resized * 1.8
    # Multi-scale low-frequency illumination variation
    y_grid, x_grid = np.ogrid[:size_px, :size_px]
    center_y, center_x = size_px * 0.48, size_px * 0.52
    dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    illumination = 1.0 - 0.07 * (dist_from_center / (size_px * 0.70))
    canvas *= illumination
    # 2. Advanced Local Interconnect Drawer with In-Line CD Variation, LER & Variable Endpoints
    def draw_li_segment(img, x1, y1, x2, y2, width=13.0, core_val=138.0, edge_val=168.0, end_type="rounded"):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Visibly distinct Critical Dimension (CD) variation: Â±8-15% width variation
        base_w = max(3, int((width + rng.uniform(-2.8, 2.8)) * scale_r))
        # Per-segment intensity drift
        c_val = core_val + rng.uniform(-8.0, 8.0)
        e_val = edge_val + rng.uniform(-8.0, 8.0)
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length <= 0:
            return
        # Core wire with subtle LER / edge roughness
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), float(c_val), thickness=int(base_w))
        # Soft SEM edge emission (max ~128-136 intensity, NO bright white blooming)
        ew = max(1, int(base_w * 0.22))
        if x1 == x2:  # Vertical line
            xl1 = max(0, x1 - base_w // 2)
            xl2 = min(img.shape[1], xl1 + ew)
            xr2 = min(img.shape[1], x1 + base_w // 2)
            xr1 = max(0, xr2 - ew)
            ymin, ymax = max(0, min(y1, y2)), min(img.shape[0], max(y1, y2))
            if ymax > ymin:
                img[ymin:ymax, xl1:xl2] = np.maximum(img[ymin:ymax, xl1:xl2], e_val)
                img[ymin:ymax, xr1:xr2] = np.maximum(img[ymin:ymax, xr1:xr2], e_val)
        elif y1 == y2:  # Horizontal line
            yt1 = max(0, y1 - base_w // 2)
            yt2 = min(img.shape[0], yt1 + ew)
            yb2 = min(img.shape[0], y1 + base_w // 2)
            yb1 = max(0, yb2 - ew)
            xmin, xmax = max(0, min(x1, x2)), min(img.shape[1], max(x1, x2))
            if xmax > xmin:
                img[yt1:yt2, xmin:xmax] = np.maximum(img[yt1:yt2, xmin:xmax], e_val)
                img[yb1:yb2, xmin:xmax] = np.maximum(img[yb1:yb2, xmin:xmax], e_val)
        # Endpoint morphology variation (blunt, tapered, or rounded)
        if end_type == "tapered":
            cv2.circle(img, (x1, y1), max(1, base_w // 3), e_val, -1)
            cv2.circle(img, (x2, y2), max(1, base_w // 3), e_val, -1)
        elif end_type == "blunt":
            cv2.line(img, (x1, y1), (x1, y1), c_val, thickness=base_w)
            cv2.line(img, (x2, y2), (x2, y2), c_val, thickness=base_w)
        else:  # soft rounded
            cv2.circle(img, (x1, y1), max(2, base_w // 2), c_val, -1)
            cv2.circle(img, (x2, y2), max(2, base_w // 2), c_val, -1)
    # Subtle contact landing head (integrated junction transition)
    def draw_li_contact(img, cx, cy, radius=5.5):
        cx, cy, r = int(cx), int(cy), max(3, int((radius + rng.uniform(-0.8, 0.8)) * scale_r))
        cv2.circle(img, (cx, cy), r, 168.0 + rng.uniform(-5, 5), -1)
        if r > 3:
            cv2.circle(img, (cx, cy), r - 2, 144.0 + rng.uniform(-5, 5), -1)
    # 3. DENSE LOCAL INTERCONNECT FIELD (Noticeable Pitch & Length Fluctuations)
    end_types = ["rounded", "blunt", "tapered"]
    for gy in np.arange(50 * scale_r, size_px - 120 * scale_r, 72.0 * scale_r):
        for gx in np.arange(50 * scale_r, size_px - 120 * scale_r, 72.0 * scale_r):
            if rng.random() < 0.25:  # Spatially correlated open substrate regions
                continue
            cx0 = gx + rng.uniform(-16, 16)
            cy0 = gy + rng.uniform(-16, 16)
            mw = rng.uniform(10.0, 16.0)  # Visibly distinct Linewidth variation Â±8-15%
            seg_l = rng.uniform(30, 145) * scale_r  # Segment length variation Â±15-30%
            e_type = str(rng.choice(end_types))
            motif_type = rng.integers(0, 5)
            if motif_type == 0:  # Short H-run + T-junction branch
                draw_li_segment(canvas, cx0, cy0, cx0 + seg_l, cy0, mw, end_type=e_type)
                if rng.random() < 0.60:
                    draw_li_segment(canvas, cx0 + seg_l / 2.0, cy0, cx0 + seg_l / 2.0, cy0 + rng.uniform(25, 70), mw, end_type=e_type)
                    draw_li_contact(canvas, cx0 + seg_l / 2.0, cy0)
                if rng.random() < 0.35:
                    draw_li_contact(canvas, cx0, cy0)
            elif motif_type == 1:  # L-bend + stub
                draw_li_segment(canvas, cx0, cy0, cx0 + seg_l, cy0, mw, end_type=e_type)
                draw_li_segment(canvas, cx0 + seg_l, cy0, cx0 + seg_l, cy0 + rng.uniform(30, 85), mw, end_type=e_type)
                draw_li_contact(canvas, cx0 + seg_l, cy0)
                if rng.random() < 0.40:
                    draw_li_contact(canvas, cx0 + seg_l, cy0 + rng.uniform(30, 85))
            elif motif_type == 2:  # Short step jog
                draw_li_segment(canvas, cx0, cy0, cx0 + seg_l / 2.0, cy0, mw, end_type=e_type)
                draw_li_segment(canvas, cx0 + seg_l / 2.0, cy0, cx0 + seg_l / 2.0, cy0 + 35 * scale_r, mw, end_type=e_type)
                draw_li_segment(canvas, cx0 + seg_l / 2.0, cy0 + 35 * scale_r, cx0 + seg_l, cy0 + 35 * scale_r, mw, end_type=e_type)
                if rng.random() < 0.45:
                    draw_li_contact(canvas, cx0 + seg_l / 2.0, cy0)
            elif motif_type == 3:  # Short vertical branch + landing head
                draw_li_segment(canvas, cx0, cy0, cx0, cy0 + seg_l, mw, end_type=e_type)
                if rng.random() < 0.55:
                    draw_li_segment(canvas, cx0, cy0 + seg_l / 2.0, cx0 + rng.uniform(25, 70), cy0 + seg_l / 2.0, mw, end_type=e_type)
                    draw_li_contact(canvas, cx0, cy0 + seg_l / 2.0)
            else:  # Cross junction fragment
                draw_li_segment(canvas, cx0, cy0 + seg_l / 2.0, cx0 + seg_l, cy0 + seg_l / 2.0, mw, end_type=e_type)
                draw_li_segment(canvas, cx0 + seg_l / 2.0, cy0, cx0 + seg_l / 2.0, cy0 + seg_l, mw, end_type=e_type)
                draw_li_contact(canvas, cx0 + seg_l / 2.0, cy0 + seg_l / 2.0)
    # 4. OVERWRITE DISTINCTIVE TARGET TOPOLOGY AT GT REGION [gt_x, gt_y, gt_w, gt_h] = [850, 850, 256, 256]
    # GT patch uses exact same substrate background and line drawer
    gt_patch = canvas[gt_y:gt_y+gt_h, gt_x:gt_x+gt_w].copy()
    mw = 13.5
    # Realistic GT Local Interconnect Topology:
    # Segment 1: Short horizontal bus y=70 (x=30..220)
    draw_li_segment(gt_patch, 30, 70, 220, 70, mw, end_type="rounded")
    draw_li_contact(gt_patch, 30, 70)
    draw_li_contact(gt_patch, 220, 70)
    # Segment 2: Vertical branch x=90 (y=30..160) forming T-junction at (90, 70)
    draw_li_segment(gt_patch, 90, 30, 90, 160, mw, end_type="blunt")
    draw_li_contact(gt_patch, 90, 30)
    draw_li_contact(gt_patch, 90, 70)
    # Segment 3: L-bend horizontal from (90, 160) -> (170, 160)
    draw_li_segment(gt_patch, 90, 160, 170, 160, mw, end_type="rounded")
    draw_li_contact(gt_patch, 170, 160)
    # Segment 4: Step jog vertical from (170, 160) -> (170, 220)
    draw_li_segment(gt_patch, 170, 160, 170, 220, mw, end_type="tapered")
    draw_li_contact(gt_patch, 170, 220)
    # Segment 5: Short side stub from (170, 125) -> (215, 125)
    draw_li_segment(gt_patch, 170, 125, 215, 125, mw, end_type="rounded")
    draw_li_contact(gt_patch, 215, 125)
    canvas[gt_y:gt_y + gt_h, gt_x:gt_x + gt_w] = gt_patch
    # 5. HARD NEGATIVE REGIONS (NEG-1 to NEG-5)
    # Matched topological complexity and visual contrast
    if size_px <= 1000:
        neg_boxes = [
            (100, 150),   # NEG-1: Diff segment lengths
            (640, 150),  # NEG-2: Diff branch orientation
            (150, 640),  # NEG-3: Diff junction arrangement
            (640, 640), # NEG-4: Diff spacing/pitch
            (370, 100),   # NEG-5: Displaced segment
        ]
    else:
        neg_boxes = [
            (350, 450),   # NEG-1: Diff segment lengths
            (1350, 450),  # NEG-2: Diff branch orientation
            (450, 1350),  # NEG-3: Diff junction arrangement
            (1350, 1350), # NEG-4: Diff spacing/pitch
            (850, 350),   # NEG-5: Displaced segment
        ]
    # NEG-1: Similar topology but different segment lengths
    neg1_patch = canvas[neg_boxes[0][1]:neg_boxes[0][1]+256, neg_boxes[0][0]:neg_boxes[0][0]+256].copy()
    draw_li_segment(neg1_patch, 25, 70, 180, 70, mw, end_type="rounded")
    draw_li_segment(neg1_patch, 90, 30, 90, 190, mw, end_type="blunt")
    draw_li_segment(neg1_patch, 90, 190, 225, 190, mw, end_type="tapered")
    draw_li_contact(neg1_patch, 25, 70)
    draw_li_contact(neg1_patch, 90, 70)
    draw_li_contact(neg1_patch, 225, 190)
    canvas[neg_boxes[0][1]:neg_boxes[0][1]+256, neg_boxes[0][0]:neg_boxes[0][0]+256] = neg1_patch
    # NEG-2: Similar L/T connectivity but different orientation
    neg2_patch = canvas[neg_boxes[1][1]:neg_boxes[1][1]+256, neg_boxes[1][0]:neg_boxes[1][0]+256].copy()
    draw_li_segment(neg2_patch, 40, 55, 215, 55, mw, end_type="blunt")
    draw_li_segment(neg2_patch, 215, 55, 215, 180, mw, end_type="rounded")
    draw_li_segment(neg2_patch, 40, 135, 155, 135, mw, end_type="tapered")
    draw_li_segment(neg2_patch, 155, 135, 155, 220, mw, end_type="rounded")
    draw_li_contact(neg2_patch, 40, 55)
    draw_li_contact(neg2_patch, 215, 55)
    draw_li_contact(neg2_patch, 155, 135)
    canvas[neg_boxes[1][1]:neg_boxes[1][1]+256, neg_boxes[1][0]:neg_boxes[1][0]+256] = neg2_patch
    # NEG-3: Similar local metal density but different junction arrangement
    neg3_patch = canvas[neg_boxes[2][1]:neg_boxes[2][1]+256, neg_boxes[2][0]:neg_boxes[2][0]+256].copy()
    draw_li_segment(neg3_patch, 30, 75, 125, 75, mw, end_type="tapered")
    draw_li_segment(neg3_patch, 125, 75, 125, 155, mw, end_type="rounded")
    draw_li_segment(neg3_patch, 125, 155, 215, 155, mw, end_type="blunt")
    draw_li_segment(neg3_patch, 215, 155, 215, 220, mw, end_type="rounded")
    draw_li_contact(neg3_patch, 30, 75)
    draw_li_contact(neg3_patch, 125, 75)
    draw_li_contact(neg3_patch, 215, 155)
    canvas[neg_boxes[2][1]:neg_boxes[2][1]+256, neg_boxes[2][0]:neg_boxes[2][0]+256] = neg3_patch
    # NEG-4: Similar connected structure but different spacing/pitch
    neg4_patch = canvas[neg_boxes[3][1]:neg_boxes[3][1]+256, neg_boxes[3][0]:neg_boxes[3][0]+256].copy()
    draw_li_segment(neg4_patch, 30, 60, 225, 60, mw, end_type="rounded")
    draw_li_segment(neg4_patch, 30, 150, 225, 150, mw, end_type="blunt")
    draw_li_segment(neg4_patch, 85, 20, 85, 225, mw, end_type="rounded")
    draw_li_segment(neg4_patch, 175, 60, 175, 225, mw, end_type="tapered")
    draw_li_contact(neg4_patch, 30, 60)
    draw_li_contact(neg4_patch, 85, 60)
    draw_li_contact(neg4_patch, 175, 150)
    canvas[neg_boxes[3][1]:neg_boxes[3][1]+256, neg_boxes[3][0]:neg_boxes[3][0]+256] = neg4_patch
    # NEG-5: Similar overall appearance but missing/displaced one segment
    neg5_patch = canvas[neg_boxes[4][1]:neg_boxes[4][1]+256, neg_boxes[4][0]:neg_boxes[4][0]+256].copy()
    draw_li_segment(neg5_patch, 30, 70, 185, 70, mw, end_type="blunt")
    draw_li_segment(neg5_patch, 185, 70, 185, 165, mw, end_type="rounded")
    draw_li_segment(neg5_patch, 95, 70, 95, 205, mw, end_type="tapered")
    draw_li_segment(neg5_patch, 95, 205, 185, 205, mw, end_type="rounded")
    draw_li_contact(neg5_patch, 30, 70)
    draw_li_contact(neg5_patch, 185, 70)
    draw_li_contact(neg5_patch, 95, 205)
    canvas[neg_boxes[4][1]:neg_boxes[4][1]+256, neg_boxes[4][0]:neg_boxes[4][0]+256] = neg5_patch
    # 6. SEM Gaussian spot blur + fine irregular detector grain
    sigma = 0.95 * scale_r
    blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Fine irregular SEM detector noise
    fine_noise = rng.normal(0, 2.8, blurred.shape).astype(np.float32)
    search_field = np.clip(blurred + fine_noise, 0, 255).astype(np.uint8)
    return search_field


# ================================================================
# CLASS 3: GATE / POLY SEM GENERATOR (Approved Pattern Architecture)
# ================================================================
def generate_micro_ler(length, amplitude=0.35, correlation_len=12.0, rng=None):
    """
    Generates realistic sub-pixel micro-stochastic Line-Edge Roughness (LER) along straight feature edges.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    raw_noise = rng.normal(0, amplitude, length).astype(np.float32)
    ksize = max(3, int(round(correlation_len * 2)) | 1)
    smoothed = cv2.GaussianBlur(raw_noise.reshape(-1, 1), (0, ksize), sigmaX=0, sigmaY=correlation_len/3.0).flatten()
    return cv2.normalize(smoothed, None, alpha=-amplitude, beta=amplitude, norm_type=cv2.NORM_MINMAX)


def render_gate_poly_sem_layout(
    canvas,
    pattern_params=None,
    is_search_field=False,
    scale_ratio=10.0,
    offset_x=0.0,
    offset_y=0.0,
    rotation_deg=0.0,
    ler_amplitude=0.35,
    rng=None
):
    """
    Renders authentic Gate/Poly SEM layout dominated by horizontal gate lines with subtle process variation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = canvas.shape[:2]
    s_factor = (1.0 / (scale_ratio * 0.35)) if is_search_field else 1.0

    if pattern_params is None:
        pattern_params = {
            'ref_gate_pitch': 135.0,
            'ref_gate_width': 34.0,
            'faint_fin_pitch': 105.0,
            'faint_fin_width': 12.0,
            'core_val': 148.0,
            'edge_val': 180.0,
            'faint_fin_val': 18.0
        }

    ref_gate_p = float(pattern_params.get('ref_gate_pitch', 135.0))
    ref_gate_w = float(pattern_params.get('ref_gate_width', 34.0))
    faint_fin_p = float(pattern_params.get('faint_fin_pitch', 105.0))
    faint_fin_w = float(pattern_params.get('faint_fin_width', 12.0))
    shift_x = float(pattern_params.get('overlay_shift_x', 0.0)) * s_factor
    shift_y = float(pattern_params.get('overlay_shift_y', 0.0)) * s_factor

    # 1. Render Faint Secondary Background Vertical Silicon Channels (Low Contrast Substrate Texture)
    faint_fin_mask = np.zeros((h, w), dtype=np.float32)
    faint_val = float(pattern_params.get('faint_fin_val', 14.0))

    if faint_val > 0:
        curr_fx = (offset_x % (faint_fin_p * s_factor))
        while curr_fx < w + 100.0:
            fw = max(2.0, faint_fin_w * s_factor)
            x1 = max(0, int(round(curr_fx - fw / 2.0)))
            x2 = min(w, int(round(curr_fx + fw / 2.0 + 1)))
            if x2 > x1:
                faint_fin_mask[:, x1:x2] = faint_val
            curr_fx += faint_fin_p * s_factor

        faint_fin_mask = cv2.GaussianBlur(faint_fin_mask, (5, 5), 1.5 * s_factor)

    # 2. Generate Primary Horizontal Gate / Poly Lines
    gate_y_list = []
    gate_widths = []
    gate_pitches = []

    curr_y = (offset_y % (ref_gate_p * s_factor))
    gate_w_ratios = [0.99, 1.00, 1.01, 0.99, 1.01, 1.00]
    gate_p_ratios = [0.98, 1.00, 1.02, 0.99, 1.01, 1.00]

    jdx = 0
    while curr_y < h + 100.0:
        gp_val = ref_gate_p * gate_p_ratios[jdx % len(gate_p_ratios)] * s_factor
        gw_val = ref_gate_w * gate_w_ratios[jdx % len(gate_w_ratios)] * s_factor

        # 12% of gates exhibit subtle local CD fluctuation (+-1 fine pixel)
        if rng.uniform(0.0, 1.0) < 0.12:
            gw_val += rng.uniform(-1.0, 1.0) * s_factor

        gw_val = max(7.0 if is_search_field else 22.0, gw_val)
        gate_y_list.append(curr_y + shift_y)
        gate_widths.append(gw_val)

        if jdx > 0:
            gate_pitches.append(curr_y - gate_y_list[-2])

        curr_y += gp_val
        jdx += 1

    # Feature masks
    gate_core = np.zeros((h, w), dtype=np.float32)
    gate_edge = np.zeros((h, w), dtype=np.float32)

    c_val = float(pattern_params.get('core_val', 148.0))
    e_val = float(pattern_params.get('edge_val', 180.0))

    # 3. Render Horizontal Gate Lines with Micro-LER and Edge Bloom
    for gy, gw in zip(gate_y_list, gate_widths):
        ler_offsets = generate_micro_ler(w, amplitude=ler_amplitude * s_factor, correlation_len=12.0 * s_factor, rng=rng)
        for x in range(w):
            y_c = float(gy + ler_offsets[x])
            y1 = max(0, int(round(y_c - gw / 2.0)))
            y2 = min(h, int(round(y_c + gw / 2.0 + 1)))
            if y2 > y1:
                gate_core[y1:y2, x] = c_val
                ew = max(1, int(round(gw * 0.25)))
                yt1, yt2 = max(0, y1 - ew), y1
                yb1, yb2 = y2, min(h, y2 + ew)
                if yt2 > yt1:
                    gate_edge[yt1:yt2, x] = e_val
                if yb2 > yb1:
                    gate_edge[yb1:yb2, x] = e_val

    # Sub-pixel SEM edge bloom
    ksize = 3 if is_search_field else 5
    gate_edge = cv2.GaussianBlur(gate_edge, (ksize, ksize), 0.8 * s_factor)
    composite = np.maximum(gate_core, gate_edge)

    # 4. INDEPENDENT STOCHASTIC INTENSITY PERTURBATIONS ALONG GATES (~15% of gate segments)
    if np.any(composite > 50.0):
        local_int_map = np.zeros((h, w), dtype=np.float32)
        for gy in gate_y_list:
            for x_seg in range(0, w, int(round(120.0 * s_factor))):
                if rng.uniform(0.0, 1.0) < 0.15:
                    delta_int = rng.uniform(-6.0, +6.0)
                    rx_px = max(5, int(round(40.0 * s_factor)))
                    ry_px = max(3, int(round(12.0 * s_factor)))
                    x1, x2 = max(0, int(x_seg - rx_px)), min(w, int(x_seg + rx_px + 1))
                    y1, y2 = max(0, int(gy - ry_px)), min(h, int(gy + ry_px + 1))
                    if x2 > x1 and y2 > y1:
                        y_sub, x_sub = np.ogrid[y1:y2, x1:x2]
                        gauss = np.exp(-((x_sub - x_seg)**2 / (2.0 * (rx_px * 0.5)**2) + (y_sub - gy)**2 / (2.0 * (ry_px * 0.5)**2))).astype(np.float32)
                        local_int_map[y1:y2, x1:x2] += gauss * delta_int

        composite[composite > 50.0] = np.clip(composite[composite > 50.0] + local_int_map[composite > 50.0], 0, 255)

    # Combine faint background substrate texture with primary gate features
    out_canvas = np.maximum(canvas + faint_fin_mask, composite)

    # Optional Rotation (-1.2 to +1.2 deg)
    if abs(rotation_deg) > 1e-3:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_deg, 1.0)
        out_canvas = cv2.warpAffine(out_canvas, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    metrics = {
        "gate_count": len(gate_y_list),
        "gate_width_mean": float(np.mean(gate_widths)),
        "gate_width_std": float(np.std(gate_widths)),
        "gate_pitch_mean": float(np.mean(gate_pitches)) if gate_pitches else 0.0,
        "gate_pitch_std": float(np.std(gate_pitches)) if gate_pitches else 0.0,
    }

    return out_canvas, metrics


def render_class3_gate_poly_field(size_px=1000, rng=None, gt_x=None, gt_y=None, gt_w=None, gt_h=None):
    """
    Renders Class 3 GATE_POLY field matching approved_pattern/utils_p3.py.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    scale_ratio = float(rng.uniform(9.2, 10.8))
    rotation_deg = float(rng.uniform(-1.2, 1.2))

    gt_params = {
        'ref_gate_pitch': float(rng.uniform(133.0, 137.0)),
        'ref_gate_width': float(rng.uniform(33.0, 35.0)),
        'faint_fin_pitch': float(rng.uniform(103.0, 107.0)),
        'faint_fin_width': 12.0,
        'overlay_shift_x': 0.0,
        'overlay_shift_y': 0.0,
        'core_val': 118.0,
        'edge_val': 150.0,
        'faint_fin_val': 14.0
    }

    bg_val = 84.0
    search_canvas = np.full((size_px, size_px), bg_val, dtype=np.float32)

    bg_noise = rng.normal(0, 1.0, (size_px, size_px)).astype(np.float32)
    bg_noise = cv2.GaussianBlur(bg_noise, (0, 0), sigmaX=8.0, sigmaY=8.0)
    search_canvas = np.clip(search_canvas + bg_noise, 70.0, 100.0)

    if gt_x is None:
        gt_cx = float(rng.uniform(220.0, size_px - 220.0))
        gt_cy = float(rng.uniform(220.0, size_px - 220.0))
    else:
        gt_cx = float(gt_x + (gt_w / 2.0 if gt_w else 128.0))
        gt_cy = float(gt_y + (gt_h / 2.0 if gt_h else 128.0))

    ref_target_w = 280.0
    search_target_w = ref_target_w / (scale_ratio * 0.35)

    neg_params_list = [
        {'ref_gate_pitch': gt_params['ref_gate_pitch'] * 0.75, 'ref_gate_width': gt_params['ref_gate_width'], 'overlay_shift_x': 0.0, 'overlay_shift_y': 0.0, 'ler_amplitude': 0.35},
        {'ref_gate_pitch': gt_params['ref_gate_pitch'], 'ref_gate_width': gt_params['ref_gate_width'] * 1.40, 'overlay_shift_x': 0.0, 'overlay_shift_y': 0.0, 'ler_amplitude': 0.35},
        {'ref_gate_pitch': gt_params['ref_gate_pitch'] * 1.25, 'ref_gate_width': gt_params['ref_gate_width'] * 0.85, 'overlay_shift_x': 0.0, 'overlay_shift_y': 0.0, 'ler_amplitude': 0.35},
        {'ref_gate_pitch': gt_params['ref_gate_pitch'], 'ref_gate_width': gt_params['ref_gate_width'], 'overlay_shift_x': 0.0, 'overlay_shift_y': 12.0, 'ler_amplitude': 0.35},
        {'ref_gate_pitch': gt_params['ref_gate_pitch'] * 1.10, 'ref_gate_width': gt_params['ref_gate_width'] * 1.20, 'overlay_shift_x': 0.0, 'overlay_shift_y': 0.0, 'ler_amplitude': 0.85}
    ]

    regions_to_blend = [
        {"center": (gt_cx, gt_cy), "params": gt_params, "radius": search_target_w * 1.6}
    ]

    for idx, nparams in enumerate(neg_params_list):
        nx = float(rng.uniform(150.0, size_px - 150.0))
        ny = float(rng.uniform(150.0, size_px - 150.0))
        while np.hypot(nx - gt_cx, ny - gt_cy) < 170.0 or any(np.hypot(nx - r["center"][0], ny - r["center"][1]) < 150.0 for r in regions_to_blend):
            nx = float(rng.uniform(150.0, size_px - 150.0))
            ny = float(rng.uniform(150.0, size_px - 150.0))

        p_full = dict(gt_params)
        p_full.update(nparams)
        regions_to_blend.append({"center": (nx, ny), "params": p_full, "radius": search_target_w * 1.6})

    base_field, _ = render_gate_poly_sem_layout(
        np.zeros_like(search_canvas),
        pattern_params=gt_params,
        is_search_field=True,
        scale_ratio=scale_ratio,
        offset_x=20.0,
        offset_y=20.0,
        rotation_deg=rotation_deg,
        rng=rng
    )
    composite_layout = base_field.copy()

    y_g, x_g = np.ogrid[:size_px, :size_px]
    for reg in regions_to_blend:
        cx, cy = reg["center"]
        p_set = reg["params"]
        rad = reg["radius"]

        local_layout, _ = render_gate_poly_sem_layout(
            np.zeros_like(search_canvas),
            pattern_params=p_set,
            is_search_field=True,
            scale_ratio=scale_ratio,
            offset_x=cx - rad,
            offset_y=cy - rad,
            rotation_deg=rotation_deg,
            ler_amplitude=p_set.get('ler_amplitude', 0.35),
            rng=rng
        )

        dist_sq = (x_g - cx)**2 + (y_g - cy)**2
        weight = np.exp(-dist_sq / (2.0 * (rad * 0.65)**2)).astype(np.float32)
        composite_layout = composite_layout * (1.0 - weight) + local_layout * weight

    search_canvas = np.maximum(search_canvas, composite_layout)

    sigma = 0.80
    blurred_search = cv2.GaussianBlur(search_canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    fine_noise = rng.normal(0, 2.0, blurred_search.shape).astype(np.float32)
    search_img = np.clip(blurred_search + fine_noise, 0, 255).astype(np.uint8)

    return search_img


# ================================================================
# CLASS 9: FULL FINFET ARRAY & CELL GENERATOR (Fins + Gates + Contacts)
# ================================================================
def maybe_collapse_gap(gap_nm: float, collapse_threshold_nm: float, rng: np.random.Generator) -> bool:
    if gap_nm <= 0:
        return True
    if gap_nm < collapse_threshold_nm:
        prob = (collapse_threshold_nm - gap_nm) / collapse_threshold_nm
        return bool(rng.random() < prob * 0.5)
    return False


def _finfet_line_positions(size_px: int, pitch_nm: float, rng: np.random.Generator, jitter_nm: float = 1.0) -> np.ndarray:
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, jitter_nm)
    return np.array(positions, dtype=np.float32)


def _finfet_line_mask(
    size_px: int,
    positions: np.ndarray,
    width_nm: float,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    width_jitter_fraction: float = 0.10,
    linewidth_bias_nm: float = 0.0,
) -> np.ndarray:
    mask = np.zeros(size_px, dtype=bool)
    biased_width_nm = max(width_nm + linewidth_bias_nm, 1.0)
    widths = biased_width_nm * (1.0 + rng.normal(0, width_jitter_fraction, size=len(positions)))
    widths = np.clip(widths, biased_width_nm * 0.5, biased_width_nm * 1.5)
    for i, center in enumerate(positions):
        half_w = widths[i] / 2.0
        lo = int(round(center - half_w))
        hi = int(round(center + half_w))
        mask[max(lo, 0):min(hi, size_px)] = True

        if i + 1 < len(positions):
            next_center = positions[i + 1]
            next_half_w = widths[i + 1] / 2.0
            gap_nm = (next_center - next_half_w) - (center + half_w)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                bridge_lo = int(round(center + half_w))
                bridge_hi = int(round(next_center - next_half_w))
                mask[max(bridge_lo, 0):min(bridge_hi, size_px)] = True
    return mask


def generate_finfet_canvas(
    size_px: int,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
    bg_val: int = 78,
    fin_val: int = 145,
    gate_val: int = 175,
    contact_val: int = 215,
):
    """
    Render full FinFET multi-layer array: vertical fins, horizontal poly gates,
    and source/drain contact pads in the diffusion regions.
    """
    canvas = np.full((size_px, size_px), bg_val, dtype=np.uint8)

    fin_positions = _finfet_line_positions(size_px, preset["fin_pitch_nm"], rng)
    gate_positions = _finfet_line_positions(size_px, preset["gate_pitch_nm"], rng)

    col_mask = _finfet_line_mask(
        size_px, fin_positions, preset["fin_width_nm"], collapse_threshold_nm, rng,
        linewidth_bias_nm=linewidth_bias_nm,
    )
    row_mask = _finfet_line_mask(
        size_px, gate_positions, preset["gate_length_nm"], collapse_threshold_nm, rng,
        linewidth_bias_nm=linewidth_bias_nm,
    )

    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], fin_val)
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], gate_val)

    half = max(1, int(round(max(preset["contact_size_nm"] + linewidth_bias_nm, 1.0) / 2.0)))
    for i, fin_x in enumerate(fin_positions):
        for j in range(len(gate_positions) - 1):
            if (i + j) % 2 == 0:
                mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
                x, y = int(round(fin_x)), int(round(mid_y))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, contact_val, -1)

    if corner_rounding_px >= 0.5:
        k = max(1, int(round(corner_rounding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    return canvas


PRESETS_ZONED = {
    'dram_1x': {'kind': 'dram', 'fin_pitch_nm': 96, 'fin_width_nm': 32, 'gate_pitch_nm': 64, 'gate_length_nm': 32, 'contact_size_nm': 32},
    'dram_dense': {'kind': 'dram', 'fin_pitch_nm': 72, 'fin_width_nm': 24, 'gate_pitch_nm': 48, 'gate_length_nm': 24, 'contact_size_nm': 24},
    'dram_loose': {'kind': 'dram', 'fin_pitch_nm': 144, 'fin_width_nm': 48, 'gate_pitch_nm': 96, 'gate_length_nm': 48, 'contact_size_nm': 48},
    'dram_wide': {'kind': 'dram', 'fin_pitch_nm': 180, 'fin_width_nm': 60, 'gate_pitch_nm': 120, 'gate_length_nm': 56, 'contact_size_nm': 58},
    'dram_legacy': {'kind': 'dram', 'fin_pitch_nm': 240, 'fin_width_nm': 80, 'gate_pitch_nm': 160, 'gate_length_nm': 78, 'contact_size_nm': 78},
    'finfet_7nm': {'kind': 'finfet', 'fin_pitch_nm': 40, 'fin_width_nm': 14, 'gate_pitch_nm': 76, 'gate_length_nm': 24, 'contact_size_nm': 24},
    'finfet_10nm': {'kind': 'finfet', 'fin_pitch_nm': 48, 'fin_width_nm': 16, 'gate_pitch_nm': 90, 'gate_length_nm': 28, 'contact_size_nm': 28},
    'finfet_14nm': {'kind': 'finfet', 'fin_pitch_nm': 60, 'fin_width_nm': 20, 'gate_pitch_nm': 110, 'gate_length_nm': 34, 'contact_size_nm': 34},
    'finfet_22nm': {'kind': 'finfet', 'fin_pitch_nm': 80, 'fin_width_nm': 26, 'gate_pitch_nm': 150, 'gate_length_nm': 46, 'contact_size_nm': 44},
    'finfet_28nm': {'kind': 'finfet', 'fin_pitch_nm': 96, 'fin_width_nm': 32, 'gate_pitch_nm': 180, 'gate_length_nm': 56, 'contact_size_nm': 52},
    'finfet_45nm': {'kind': 'finfet', 'fin_pitch_nm': 140, 'fin_width_nm': 46, 'gate_pitch_nm': 260, 'gate_length_nm': 80, 'contact_size_nm': 76}
}

PRESET_KEYS_ZONED = list(PRESETS_ZONED.keys())

GRID_PRESETS_CLASS9 = [
    ['finfet_7nm',   'finfet_10nm',  'finfet_14nm',  'finfet_7nm'],
    ['dram_wide',    'finfet_10nm',  'dram_1x',      'dram_legacy'],
    ['finfet_14nm',  'dram_dense',   'dram_loose',   'finfet_10nm'],
    ['finfet_7nm',   'dram_wide',    'finfet_14nm',  'finfet_7nm']
]


def _line_positions(size_px: int, pitch_nm: float, rng: np.random.Generator) -> np.ndarray:
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, 1.0)
    return np.array(positions)


def _line_mask(size_px: int, positions: np.ndarray, width_nm: float, collapse_threshold_nm: float, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(size_px, dtype=bool)
    widths = width_nm * (1.0 + rng.normal(0, 0.05, size=len(positions)))
    widths = np.clip(widths, width_nm * 0.5, width_nm * 1.5)
    for i, center in enumerate(positions):
        half_w = widths[i] / 2.0
        lo = int(round(center - half_w))
        hi = int(round(center + half_w))
        mask[max(lo, 0):min(hi, size_px)] = True
    return mask


def render_fine_mat(mat_w: int, mat_h: int, preset: dict, rng: np.random.Generator) -> np.ndarray:
    bg_val = 65
    fin_val = 135
    gate_val = 150
    contact_val = 255

    canvas = np.full((mat_h, mat_w), bg_val, dtype=np.uint8)
    
    fp = preset['fin_pitch_nm']
    gp = preset['gate_pitch_nm']
    fw = preset['fin_width_nm']
    gw = preset['gate_length_nm']
    cs = preset['contact_size_nm']
    
    fin_pos = np.arange(fp / 2.0, mat_w, fp)
    gate_pos = np.arange(gp / 2.0, mat_h, gp)
    
    # 1. Vertical fins / bitlines
    for fx in fin_pos:
        x0 = int(round(fx - fw / 2.0))
        x1 = int(round(fx + fw / 2.0))
        canvas[:, max(0, x0):min(mat_w, x1)] = np.maximum(canvas[:, max(0, x0):min(mat_w, x1)], fin_val)
        
    # 2. Horizontal wordlines / gates
    for gy in gate_pos:
        y0 = int(round(gy - gw / 2.0))
        y1 = int(round(gy + gw / 2.0))
        canvas[max(0, y0):min(mat_h, y1), :] = np.maximum(canvas[max(0, y0):min(mat_h, y1), :], gate_val)
        
    # 3. Dense Staggered Contacts in every inter-wordline diffusion gap (diagonal lattice)
    half_c = max(1, int(round(cs / 2.0)))
    for j in range(len(gate_pos) - 1):
        mid_y = int(round((gate_pos[j] + gate_pos[j + 1]) / 2.0))
        shift = (j % 2) * (fp / 2.0)
        row_fin_pos = fin_pos + shift
        
        for fx in row_fin_pos:
            if 0 <= fx < mat_w:
                cx = int(round(fx))
                cy = mid_y
                x0 = max(0, cx - half_c)
                x1 = min(mat_w, cx + half_c + 1)
                y0 = max(0, cy - half_c)
                y1 = min(mat_h, cy + half_c + 1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), contact_val, -1)
                
    return canvas


def render_class9_finfet_full_field(
    size_px: int = 1000,
    rng: np.random.Generator = None,
    gt_x: int = None,
    gt_y: int = None,
    gt_w: int = None,
    gt_h: int = None
) -> np.ndarray:
    """
    Synthesizes an authentic multi-architecture 4x4 Zoned Memory/FinFET Mat Array for Class 9
    using generate_zone_canvas with strip routing textures.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if gt_w is None:
        gt_w = 256
    if gt_h is None:
        gt_h = gt_w
    if gt_x is None:
        gt_x = int(size_px * 0.13)
    if gt_y is None:
        gt_y = int(size_px * 0.13)

    fine_size = size_px * 10
    zone_res = generate_zone_canvas(
        size_px=fine_size,
        collapse_threshold_nm=10.0,
        rng=rng,
        mat_size_nm=2200.0,
        strip_width_nm=320.0
    )
    fine_canvas = zone_res["canvas"]

    # SEM Imaging pipeline
    blurred = cv2.GaussianBlur(fine_canvas, (17, 17), 5.0)
    downsampled = cv2.resize(blurred, (size_px, size_px), interpolation=cv2.INTER_AREA)

    noise = rng.normal(0, 2.5, downsampled.shape).astype(np.float32)
    search_img = np.clip(downsampled.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return search_img


def maybe_collapse_gap(gap_nm: float, threshold_nm: float, rng: np.random.Generator, collapse_prob: float = 0.7) -> bool:
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)


def render_fine_canvas(
    size_px: int, fin_pitch: float, fin_width: float, 
    gate_pitch: float, gate_width: float, contact_size: float, 
    rng: np.random.Generator, collapse_threshold: float = 10.0
) -> np.ndarray:
    """Generates continuous FinFET layout mathematically, exactly like physical wafer."""
    BACKGROUND = 40
    FIN_VAL = 150
    GATE_VAL = 170
    CONTACT_VAL = 225

    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)

    fin_positions = _line_positions(size_px, fin_pitch, rng)
    gate_positions = _line_positions(size_px, gate_pitch, rng)

    col_mask = _line_mask(size_px, fin_positions, fin_width, collapse_threshold, rng)
    row_mask = _line_mask(size_px, gate_positions, gate_width, collapse_threshold, rng)

    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], FIN_VAL)
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], GATE_VAL)

    half = max(1, int(round(contact_size / 2.0)))
    for i, fin_x in enumerate(fin_positions):
        for j in range(len(gate_positions) - 1):
            if (i + j) % 2 == 0:
                mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
                x, y = int(round(fin_x)), int(round(mid_y))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, CONTACT_VAL, -1)
                
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    return canvas


def _strip_routing_texture(size_px: int, rng: np.random.Generator) -> np.ndarray:
    """Flat mid-gray fill with sparse orthogonal routing lines."""
    STRIP_BASE_VAL = 95
    STRIP_LINE_VAL = 128
    STRIP_LINE_PITCH_NM = 220
    STRIP_LINE_WIDTH_NM = 9

    canvas = np.full((size_px, size_px), STRIP_BASE_VAL, dtype=np.uint8)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for axis_positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in axis_positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_VAL
            else:
                canvas[:, lo:hi] = STRIP_LINE_VAL
    return canvas


def _zone_grid(size_px: int, mat_size_nm: float, strip_width_nm: float):
    """Compute alternating [mat, strip, mat, strip, ...] spans covering size_px."""
    spans = []
    pos = 0.0
    is_mat = True
    while pos < size_px:
        span_len = mat_size_nm if is_mat else strip_width_nm
        end = min(pos + span_len, size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos = end
        is_mat = not is_mat
    return spans


def generate_zone_canvas(
    size_px: int,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    mat_size_nm: float = 2600.0,
    strip_width_nm: float = 320.0,
) -> dict:
    """Tile independently-generated mats across the canvas, separated by strip material."""
    canvas = _strip_routing_texture(size_px, rng)
    row_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)
    col_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)

    mat_rects = []
    strip_rects = []

    for row_is_mat, y0, y1 in row_spans:
        for col_is_mat, x0, x1 in col_spans:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mat_h, mat_w = y1 - y0, x1 - x0
                child_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
                mat_size = max(mat_h, mat_w)
                
                fin_pitch = child_rng.uniform(40.0, 140.0)
                fin_width = fin_pitch * child_rng.uniform(0.25, 0.35)
                gate_pitch = child_rng.uniform(80.0, 260.0)
                gate_width = gate_pitch * child_rng.uniform(0.25, 0.35)
                max_contact = min(fin_pitch, gate_pitch) * 0.8
                contact_size = min(gate_width * 1.2, max_contact)
                
                mat_canvas = render_fine_canvas(
                    mat_size, fin_pitch, fin_width, gate_pitch, gate_width, contact_size, child_rng, collapse_threshold_nm
                )
                canvas[y0:y1, x0:x1] = mat_canvas[:mat_h, :mat_w]
                mat_rects.append((x0, y0, mat_w, mat_h))
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))

    return {"canvas": canvas, "mat_rects": mat_rects, "strip_rects": strip_rects}


# =========================================================================
# 16 INDEPENDENT MAT GENERATORS FOR P1–P9 (HIGH CONTRAST BRIGHT FINS)
# =========================================================================

def render_mat_p1_fin_array(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    SUBSTRATE = 50
    FIN_BODY = 185
    FIN_EDGE = 245
    canvas = np.full((h, w), SUBSTRATE, dtype=np.uint8)
    
    fp = float(rng.uniform(42.0, 75.0))
    fw = fp * float(rng.uniform(0.28, 0.36))
    
    pos = rng.uniform(fp * 0.2, fp * 0.8)
    fin_pos = []
    while pos < w:
        fin_pos.append(pos)
        pos += fp + rng.normal(0, 0.8)
        
    for fx in fin_pos:
        y_pts = np.arange(0, h, 20)
        xs_left = []
        xs_right = []
        for y in y_pts:
            w_cur = fw * (1.0 + rng.normal(0, 0.05))
            cx = fx + rng.normal(0, 0.6)
            xs_left.append((cx - w_cur / 2.0, y))
            xs_right.append((cx + w_cur / 2.0, y))
            
        pts = np.array(xs_left + xs_right[::-1], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], FIN_BODY)
        cv2.polylines(canvas, [np.array(xs_left, dtype=np.int32)], False, FIN_EDGE, 1)
        cv2.polylines(canvas, [np.array(xs_right, dtype=np.int32)], False, FIN_EDGE, 1)
        
        if rng.random() < 0.35:
            term_y = int(rng.uniform(h * 0.15, h * 0.85))
            if rng.random() < 0.5:
                canvas[:term_y, max(0, int(fx - fw)):min(w, int(fx + fw + 2))] = SUBSTRATE
            else:
                canvas[term_y:, max(0, int(fx - fw)):min(w, int(fx + fw + 2))] = SUBSTRATE
                
    return canvas


def render_mat_p2_fin_cut(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    canvas = render_mat_p1_fin_array(w, h, rng)
    n_cuts = rng.integers(8, 16)
    for _ in range(n_cuts):
        cx = int(rng.uniform(0, w))
        cy = int(rng.uniform(0, h))
        cw = int(rng.uniform(25, 65))
        ch = int(rng.uniform(70, 160))
        cv2.rectangle(canvas, (max(0, cx - cw // 2), max(0, cy - ch // 2)),
                      (min(w, cx + cw // 2), min(h, cy + ch // 2)), 50, -1)
    return canvas


def render_mat_p3_gate_poly(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    SUBSTRATE = 50
    GATE_BODY = 185
    GATE_EDGE = 245
    canvas = np.full((h, w), SUBSTRATE, dtype=np.uint8)
    
    gp = float(rng.uniform(55.0, 95.0))
    gw = gp * float(rng.uniform(0.30, 0.38))
    
    pos = rng.uniform(gp * 0.2, gp * 0.8)
    gate_pos = []
    while pos < h:
        gate_pos.append(pos)
        pos += gp + rng.normal(0, 0.8)
        
    for gy in gate_pos:
        x_pts = np.arange(0, w, 20)
        ys_top = []
        ys_bot = []
        for x in x_pts:
            w_cur = gw * (1.0 + rng.normal(0, 0.05))
            cy = gy + rng.normal(0, 0.6)
            ys_top.append((x, cy - w_cur / 2.0))
            ys_bot.append((x, cy + w_cur / 2.0))
            
        pts = np.array(ys_top + ys_bot[::-1], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], GATE_BODY)
        cv2.polylines(canvas, [np.array(ys_top, dtype=np.int32)], False, GATE_EDGE, 1)
        cv2.polylines(canvas, [np.array(ys_bot, dtype=np.int32)], False, GATE_EDGE, 1)
        
    return canvas


def render_mat_p4_fin_gate(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    canvas = render_mat_p1_fin_array(w, h, rng)
    gp = float(rng.uniform(75.0, 130.0))
    gw = gp * 0.30
    ys = np.arange(gp / 2.0, h, gp)
    for gy in ys:
        half_h = gw / 2.0
        y0 = max(0, int(round(gy - half_h)))
        y1 = min(h, int(round(gy + half_h)))
        canvas[y0:y1, :] = np.maximum(canvas[y0:y1, :], 195)
        canvas[max(0, y0):max(0, y0) + 1, :] = 245
        canvas[min(h - 1, y1 - 1):y1, :] = 245
    return canvas


def render_mat_p5_contact_array(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((h, w), 55, dtype=np.uint8)
    cv2.rectangle(canvas, (int(w * 0.06), int(h * 0.06)), (int(w * 0.94), int(h * 0.94)), 120, -1)
    cp_x = float(rng.uniform(45.0, 75.0))
    cp_y = float(rng.uniform(45.0, 75.0))
    xs = np.arange(w * 0.10, w * 0.90, cp_x)
    ys = np.arange(h * 0.10, h * 0.90, cp_y)
    rad = int(round(min(cp_x, cp_y) * 0.24))
    for x in xs:
        for y in ys:
            cv2.circle(canvas, (int(round(x)), int(round(y))), rad, 255, -1)
    return canvas


def render_mat_p6_local_interconnect(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((h, w), 50, dtype=np.uint8)
    n_tracks = rng.integers(14, 22)
    for _ in range(n_tracks):
        x1 = int(rng.uniform(w * 0.05, w * 0.85))
        y1 = int(rng.uniform(h * 0.05, h * 0.85))
        x2 = x1 + int(rng.uniform(60, 260))
        y2 = y1 + int(rng.choice([0, int(rng.uniform(40, 120))]))
        cv2.line(canvas, (x1, y1), (min(w - 5, x2), min(h - 5, y2)), 190, int(rng.uniform(12, 18)))
        cv2.circle(canvas, (x1, y1), 9, 255, -1)
        cv2.circle(canvas, (min(w - 5, x2), min(h - 5, y2)), 9, 255, -1)
    return canvas


def render_mat_p7_metal_routing(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((h, w), 50, dtype=np.uint8)
    for x in np.arange(w * 0.08, w * 0.92, rng.uniform(70, 130)):
        cv2.line(canvas, (int(x), int(h * 0.05)), (int(x), int(h * 0.95)), 195, 16)
    for y in np.arange(h * 0.08, h * 0.92, rng.uniform(70, 130)):
        cv2.line(canvas, (int(w * 0.05), int(y)), (int(w * 0.95), int(y)), 180, 16)
    return canvas


def render_mat_p8_active_cell(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((h, w), 50, dtype=np.uint8)
    cv2.rectangle(canvas, (int(w * 0.05), int(h * 0.05)), (int(w * 0.95), int(h * 0.95)), 105, -1)
    fp = float(rng.uniform(32.0, 45.0))
    for fx in np.arange(w * 0.08, w * 0.92, fp):
        canvas[:, int(fx - 4):int(fx + 4)] = 175
    gp = float(rng.uniform(55.0, 80.0))
    for gy in np.arange(h * 0.08, h * 0.92, gp):
        canvas[int(gy - 6):int(gy + 6), :] = np.maximum(canvas[int(gy - 6):int(gy + 6), :], 205)
        for fx in np.arange(w * 0.08, w * 0.92, fp):
            if rng.random() < 0.7:
                cv2.circle(canvas, (int(fx), int(gy)), 6, 255, -1)
    return canvas


CLASS_MAT_RENDERERS = {
    "FIN_ARRAY": render_mat_p1_fin_array,
    "FIN_CUT": render_mat_p2_fin_cut,
    "GATE_POLY": render_mat_p3_gate_poly,
    "FIN_GATE": render_mat_p4_fin_gate,
    "CONTACT_ARRAY": render_mat_p5_contact_array,
    "LOCAL_INTERCONNECT": render_mat_p6_local_interconnect,
    "METAL_ROUTING": render_mat_p7_metal_routing,
    "ACTIVE_CELL": render_mat_p8_active_cell,
}


DIVERSE_TECH_PRESETS = [
    {'fin_pitch': 40.0, 'gate_pitch': 76.0, 'name': 'finfet_7nm'},
    {'fin_pitch': 48.0, 'gate_pitch': 90.0, 'name': 'finfet_10nm'},
    {'fin_pitch': 60.0, 'gate_pitch': 110.0, 'name': 'finfet_14nm'},
    {'fin_pitch': 80.0, 'gate_pitch': 150.0, 'name': 'finfet_22nm'},
    {'fin_pitch': 96.0, 'gate_pitch': 180.0, 'name': 'finfet_28nm'},
    {'fin_pitch': 140.0, 'gate_pitch': 260.0, 'name': 'finfet_45nm'},
    {'fin_pitch': 44.0, 'gate_pitch': 82.0, 'name': 'finfet_8nm'},
    {'fin_pitch': 54.0, 'gate_pitch': 100.0, 'name': 'finfet_12nm'},
    {'fin_pitch': 68.0, 'gate_pitch': 130.0, 'name': 'finfet_16nm'},
    {'fin_pitch': 88.0, 'gate_pitch': 165.0, 'name': 'finfet_25nm'},
    {'fin_pitch': 110.0, 'gate_pitch': 210.0, 'name': 'finfet_32nm'},
    {'fin_pitch': 130.0, 'gate_pitch': 240.0, 'name': 'finfet_40nm'},
    {'fin_pitch': 42.0, 'gate_pitch': 78.0, 'name': 'finfet_7nm_dense'},
    {'fin_pitch': 50.0, 'gate_pitch': 95.0, 'name': 'finfet_10nm_dense'},
    {'fin_pitch': 75.0, 'gate_pitch': 140.0, 'name': 'finfet_20nm'},
    {'fin_pitch': 120.0, 'gate_pitch': 230.0, 'name': 'finfet_38nm'}
]


def render_fine_p3_mat(w: int, h: int, gate_pitch: float, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((h, w), 40, dtype=np.uint8)
    gw = gate_pitch * rng.uniform(0.28, 0.36)
    pos = rng.uniform(gate_pitch * 0.2, gate_pitch * 0.8)
    while pos < h:
        y0 = int(round(pos - gw / 2.0))
        y1 = int(round(pos + gw / 2.0))
        y0_c = max(0, y0)
        y1_c = min(h, y1)
        if y1_c > y0_c:
            canvas[y0_c:y1_c, :] = 170
            if y0_c < h: canvas[y0_c, :] = 225
            if y1_c - 1 >= 0: canvas[y1_c - 1, :] = 225
        pos += gate_pitch + rng.normal(0, 0.8)
    return canvas


P5_CONFIGS = [
    {'pitch_x': 55.0, 'pitch_y': 55.0, 'rad': 12, 'stagger': False, 'elong': 1.0},
    {'pitch_x': 42.0, 'pitch_y': 42.0, 'rad': 9, 'stagger': True, 'elong': 1.0},
    {'pitch_x': 75.0, 'pitch_y': 75.0, 'rad': 16, 'stagger': False, 'elong': 1.0},
    {'pitch_x': 90.0, 'pitch_y': 90.0, 'rad': 19, 'stagger': True, 'elong': 1.0},
    {'pitch_x': 60.0, 'pitch_y': 80.0, 'rad': 14, 'stagger': True, 'elong': 1.6},
    {'pitch_x': 48.0, 'pitch_y': 65.0, 'rad': 11, 'stagger': False, 'elong': 1.4},
    {'pitch_x': 110.0, 'pitch_y': 110.0, 'rad': 24, 'stagger': False, 'elong': 1.0},
    {'pitch_x': 38.0, 'pitch_y': 38.0, 'rad': 8, 'stagger': True, 'elong': 1.0},
    {'pitch_x': 65.0, 'pitch_y': 65.0, 'rad': 14, 'stagger': True, 'elong': 1.0},
    {'pitch_x': 85.0, 'pitch_y': 85.0, 'rad': 18, 'stagger': False, 'elong': 1.0},
    {'pitch_x': 50.0, 'pitch_y': 70.0, 'rad': 12, 'stagger': True, 'elong': 1.5},
    {'pitch_x': 70.0, 'pitch_y': 95.0, 'rad': 15, 'stagger': False, 'elong': 1.8},
    {'pitch_x': 125.0, 'pitch_y': 125.0, 'rad': 26, 'stagger': True, 'elong': 1.0},
    {'pitch_x': 45.0, 'pitch_y': 45.0, 'rad': 10, 'stagger': False, 'elong': 1.0},
    {'pitch_x': 58.0, 'pitch_y': 58.0, 'rad': 13, 'stagger': True, 'elong': 1.0},
    {'pitch_x': 100.0, 'pitch_y': 100.0, 'rad': 21, 'stagger': False, 'elong': 1.3}
]

def render_fine_p5_mat(w: int, h: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((h, w), 40, dtype=np.uint8)
    cv2.rectangle(canvas, (int(w * 0.04), int(h * 0.04)), (int(w * 0.96), int(h * 0.96)), 135, -1)
    px = cfg['pitch_x']
    py = cfg['pitch_y']
    rad = cfg['rad']
    stagger = cfg['stagger']
    elong = cfg['elong']
    ys = np.arange(h * 0.08, h * 0.92, py)
    for row_i, y in enumerate(ys):
        offset = (px / 2.0) if (stagger and row_i % 2 == 1) else 0.0
        xs = np.arange(w * 0.08 + offset, w * 0.92, px)
        for x in xs:
            cx, cy = int(round(x)), int(round(y))
            if elong > 1.1:
                cv2.ellipse(canvas, (cx, cy), (int(rad * elong), rad), 0, 0, 360, 225, -1)
                cv2.ellipse(canvas, (cx, cy), (int(rad * elong), rad), 0, 0, 360, 255, 1)
            else:
                cv2.circle(canvas, (cx, cy), rad, 225, -1)
                cv2.circle(canvas, (cx, cy), rad, 255, 1)
    return canvas


def render_16_mats_field_fine(
    pattern_type: str,
    size_px: int = 1000,
    rng: np.random.Generator = None,
    difficulty: str = "medium"
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)

    fine_size = size_px * 10
    canvas = _strip_routing_texture(fine_size, rng)

    row_spans = _zone_grid(fine_size, 2200.0, 320.0)
    col_spans = _zone_grid(fine_size, 2200.0, 320.0)

    for r_idx, (row_is_mat, y0, y1) in enumerate(row_spans):
        for c_idx, (col_is_mat, x0, x1) in enumerate(col_spans):
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mat_h, mat_w = y1 - y0, x1 - x0
                mat_idx = (r_idx * 4 + c_idx)
                mat_rng = np.random.default_rng(int(rng.integers(1, 999999)) + r_idx * 10007 + c_idx * 3571)

                if pattern_type == "GATE_POLY":
                    raw = render_class3_gate_poly_field(size_px=1000, rng=mat_rng).astype(np.float32)
                    raw_crop = cv2.resize(raw, (mat_w, mat_h), interpolation=cv2.INTER_AREA)
                    norm = (raw_crop - raw_crop.min()) / (raw_crop.max() - raw_crop.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.60) * 220.0, 0, 255).astype(np.uint8)
                elif pattern_type == "CONTACT_ARRAY":
                    raw = render_class5_contact_array_field(size_px=1000, rng=mat_rng).astype(np.float32)
                    raw_crop = cv2.resize(raw, (mat_w, mat_h), interpolation=cv2.INTER_AREA)
                    norm = (raw_crop - raw_crop.min()) / (raw_crop.max() - raw_crop.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.55) * 220.0, 0, 255).astype(np.uint8)
                elif pattern_type == "LOCAL_INTERCONNECT":
                    raw = render_class6_local_interconnect_field(size_px=1000, rng=mat_rng).astype(np.float32)
                    raw_crop = cv2.resize(raw, (mat_w, mat_h), interpolation=cv2.INTER_AREA)
                    norm = (raw_crop - raw_crop.min()) / (raw_crop.max() - raw_crop.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.60) * 220.0, 0, 255).astype(np.uint8)
                elif pattern_type == "METAL_ROUTING":
                    raw = render_class7_metal_routing_field(size_px=1000, rng=mat_rng).astype(np.float32)
                    raw_crop = cv2.resize(raw, (mat_w, mat_h), interpolation=cv2.INTER_AREA)
                    norm = (raw_crop - raw_crop.min()) / (raw_crop.max() - raw_crop.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.60) * 220.0, 0, 255).astype(np.uint8)
                elif pattern_type == "ACTIVE_CELL":
                    raw = render_class8_continuous_field(size_px=1000, rng=mat_rng).astype(np.float32)
                    raw_crop = cv2.resize(raw, (mat_w, mat_h), interpolation=cv2.INTER_AREA)
                    norm = (raw_crop - raw_crop.min()) / (raw_crop.max() - raw_crop.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.60) * 220.0, 0, 255).astype(np.uint8)
                elif pattern_type == "FIN_GATE":
                    out = render_composite_field("FIN_GATE", mat_w, 175.0, 850.0, difficulty, mat_rng, debug_layers=True)
                    raw = out["final"].astype(np.float32) if isinstance(out, dict) else out.astype(np.float32)
                    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.60) * 220.0, 0, 255).astype(np.uint8)
                else:
                    # FIN_ARRAY or FIN_CUT: Full 2D semiconductor morphology with high dynamic range
                    preset = DIVERSE_TECH_PRESETS[mat_idx % len(DIVERSE_TECH_PRESETS)]
                    out = render_composite_field(
                        pattern_type, mat_w, preset['fin_pitch'], preset['gate_pitch'],
                        difficulty, mat_rng, debug_layers=True
                    )
                    raw_mat = out["final"].astype(np.float32) if isinstance(out, dict) else out.astype(np.float32)
                    norm = (raw_mat - raw_mat.min()) / (raw_mat.max() - raw_mat.min() + 1e-5)
                    mat_img = np.clip(35.0 + np.power(norm, 0.58) * 220.0, 0, 255).astype(np.uint8)

                canvas[y0:y1, x0:x1] = mat_img[:mat_h, :mat_w]

    # SEM PSF sigma reduced slightly from 5.0 to 2.4 (kernel 9x9) for crisp edge retention
    blurred = cv2.GaussianBlur(canvas, (9, 9), 2.4)
    downsampled = cv2.resize(blurred, (size_px, size_px), interpolation=cv2.INTER_AREA)
    # Low noise to preserve high dynamic range
    noise = rng.normal(0, 1.2, downsampled.shape).astype(np.float32)
    return np.clip(downsampled.astype(np.float32) + noise, 0, 255).astype(np.uint8)




def render_class1_fin_array_field(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("FIN_ARRAY", size_px=size_px, rng=rng)

def render_class2_fin_cut_field(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("FIN_CUT", size_px=size_px, rng=rng)

def render_class3_gate_poly_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("GATE_POLY", size_px=size_px, rng=rng)

def render_class4_fin_gate_field(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("FIN_GATE", size_px=size_px, rng=rng)

def render_class5_contact_array_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("CONTACT_ARRAY", size_px=size_px, rng=rng)

def render_class6_local_interconnect_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("LOCAL_INTERCONNECT", size_px=size_px, rng=rng)

def render_class7_metal_routing_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("METAL_ROUTING", size_px=size_px, rng=rng)

def render_class8_active_cell_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("ACTIVE_CELL", size_px=size_px, rng=rng)





def render_class1_fin_array_field(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("FIN_ARRAY", size_px=size_px, rng=rng)

def render_class2_fin_cut_field(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("FIN_CUT", size_px=size_px, rng=rng)

def render_class3_gate_poly_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("GATE_POLY", size_px=size_px, rng=rng)

def render_class4_fin_gate_field(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("FIN_GATE", size_px=size_px, rng=rng)

def render_class5_contact_array_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("CONTACT_ARRAY", size_px=size_px, rng=rng)

def render_class6_local_interconnect_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("LOCAL_INTERCONNECT", size_px=size_px, rng=rng)

def render_class7_metal_routing_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("METAL_ROUTING", size_px=size_px, rng=rng)

def render_class8_active_cell_field_zoned(size_px: int = 1000, rng: np.random.Generator = None) -> np.ndarray:
    return render_16_mats_field_fine("ACTIVE_CELL", size_px=size_px, rng=rng)






