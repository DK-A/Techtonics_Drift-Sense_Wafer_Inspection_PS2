"""SEM acquisition artifacts. Upstream: aayushraina21/drift-sense-synthetic-data."""
import cv2
import numpy as np


def gaussian_psf_blur(img, spot_size_nm, pixel_size_nm, astigmatism_ratio=1.0):
    sigma_x = max(spot_size_nm / pixel_size_nm, 1e-6)
    sigma_y = max(sigma_x * astigmatism_ratio, 1e-6)
    k = int(2 * round(3 * max(sigma_x, sigma_y)) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma_x, sigmaY=sigma_y)


def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    out = img.astype(np.float64) * (1.0 - strength * (r ** 2))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img
    norm = img.astype(np.float64) / 255.0
    return np.clip(np.power(np.clip(norm, 0, 1), gamma) * 255.0, 0, 255).astype(np.uint8)


def apply_barrel_distortion(img, k):
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    factor = 1.0 + k * (nx ** 2 + ny ** 2)
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def add_charging_streaks(img, streak_prob, intensity, rng):
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape
    out = img.astype(np.float64)
    n_streaks = rng.poisson(max(streak_prob * (h / 100.0), 0))
    for _ in range(n_streaks):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.uint8)


def downsample_area_average(img, factor):
    h, w = img.shape
    return cv2.resize(img, (int(round(w / factor)), int(round(h / factor))),
                      interpolation=cv2.INTER_AREA)


def apply_raster_drift(img, shear_amplitude_px, jitter_std_px, rng):
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    h, w = img.shape
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0 else np.zeros(h)
    row_shift = (shear + jitter).astype(np.float32)
    map_x = (np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None])
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def add_shot_noise(img, dose, rng):
    img_f = img.astype(np.float64)
    counts = np.clip(img_f / 255.0 * dose, 0, None)
    noisy = rng.poisson(counts).astype(np.float64) / dose * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_detector_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, size=img.shape),
                   0, 255).astype(np.uint8)


def add_speckle_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    out = img.astype(np.float64) * (1.0 + rng.normal(0, sigma, size=img.shape))
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img, prob, rng):
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def image_reference(crop, pixel_size_nm, spot_size_nm, dose, rng,
                    detector_noise_sigma=2.0, drift_jitter_px=0.2,
                    astigmatism_ratio=1.0, vignette_strength=0.0, gamma=1.0,
                    barrel_distortion_k=0.0, charging_streak_prob=0.0,
                    charging_streak_intensity=0.0, speckle_sigma=0.0,
                    salt_pepper_prob=0.0):
    img = gaussian_psf_blur(crop, spot_size_nm, pixel_size_nm, astigmatism_ratio)
    img = apply_raster_drift(img, 0.0, drift_jitter_px, rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img


def image_search(full_canvas, pixel_size_ref_nm, pixel_size_search_nm, spot_size_nm,
                 dose, rng, shear_amplitude_px=1.5, drift_jitter_px=0.5,
                 detector_noise_sigma=5.0, astigmatism_ratio=1.0, vignette_strength=0.0,
                 gamma=1.0, barrel_distortion_k=0.0, charging_streak_prob=0.0,
                 charging_streak_intensity=0.0, speckle_sigma=0.0, salt_pepper_prob=0.0):
    factor = pixel_size_search_nm / pixel_size_ref_nm
    blurred = gaussian_psf_blur(full_canvas, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio)
    img = downsample_area_average(blurred, factor)
    img = apply_raster_drift(img, shear_amplitude_px, drift_jitter_px, rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img
