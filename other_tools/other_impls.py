from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

try:
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3 as FIDInceptionV3

    PYTORCH_FID_AVAILABLE = True
except ImportError:
    calculate_frechet_distance = None
    FIDInceptionV3 = None
    PYTORCH_FID_AVAILABLE = False

try:
    from scipy import linalg
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import uniform_filter as _ndimage_uniform_filter

    SCIPY_AVAILABLE = True
except ImportError:
    linalg = None
    gaussian_filter = None
    _ndimage_uniform_filter = None
    SCIPY_AVAILABLE = False

try:
    from torchvision.models import Inception_V3_Weights, inception_v3

    TORCHVISION_INCEPTION_AVAILABLE = True
except ImportError:
    Inception_V3_Weights = None
    inception_v3 = None
    TORCHVISION_INCEPTION_AVAILABLE = False


EPS = 1e-8


def _progress(iterable, *, total: int | None, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _load_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _ensure_rgb_uint8_bhwc(images: Sequence[str | Path] | np.ndarray) -> np.ndarray:
    if isinstance(images, np.ndarray):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected BHWC RGB array, got shape {images.shape}")
        if images.dtype != np.uint8:
            images = np.clip(images, 0, 255).astype(np.uint8)
        return images

    loaded = [_load_rgb_image(path) for path in images]
    if not loaded:
        raise ValueError("No images provided")
    return np.stack(loaded, axis=0)


def _resize_rgb_pair(real_rgb: np.ndarray, generated_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if real_rgb.shape[:2] == generated_rgb.shape[:2]:
        return real_rgb, generated_rgb
    resized = cv2.resize(generated_rgb, (real_rgb.shape[1], real_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return real_rgb, resized


def _to_gray_float01(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _rgb_uint8_to_torch(images: np.ndarray, device: torch.device, normalize_to_neg1: bool) -> torch.Tensor:
    tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).contiguous() / 255.0
    if normalize_to_neg1:
        tensor = tensor * 2.0 - 1.0
    return tensor


def _mean_ignore_none(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def summarize_metric_lists(metric_lists: dict[str, Sequence[float | None]]) -> dict[str, float | None]:
    return {name: _mean_ignore_none(values) for name, values in metric_lists.items()}


# Default fusion weights for 5 scales (common values used with MS-SSIM in the literature).
_DEFAULT_MS_SSIM_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window_2d(win: int = 11, sigma: float = 1.5) -> np.ndarray:
    if win % 2 == 0:
        raise ValueError("Gaussian window size must be odd.")
    ax = np.arange(-(win // 2), win // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    s = float(g.sum())
    return g / (s + EPS)


def _ssim_cs_means_single_scale(
    ref: np.ndarray,
    dist: np.ndarray,
    *,
    win: int,
    win_sigma: float,
    L: float,
    K1: float,
    K2: float,
) -> tuple[float, float]:
    """Mean SSIM and mean contrast–structure term on one scale (grayscale, same shape)."""
    ref_f = ref.astype(np.float64, copy=False)
    dist_f = dist.astype(np.float64, copy=False)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    w = _gaussian_window_2d(win, win_sigma)

    def conv(x: np.ndarray) -> np.ndarray:
        return cv2.filter2D(x, cv2.CV_64F, w, borderType=cv2.BORDER_REFLECT101)

    mu1 = conv(ref_f)
    mu2 = conv(dist_f)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    s1_sq = conv(ref_f * ref_f) - mu1_sq
    s2_sq = conv(dist_f * dist_f) - mu2_sq
    s12 = conv(ref_f * dist_f) - mu12
    s1_sq = np.maximum(s1_sq, 0.0)
    s2_sq = np.maximum(s2_sq, 0.0)

    lum = (2.0 * mu12 + C1) / (mu1_sq + mu2_sq + C1 + EPS)
    cs = (2.0 * s12 + C2) / (s1_sq + s2_sq + C2 + EPS)
    ssim_map = lum * cs
    return float(np.mean(ssim_map)), float(np.mean(cs))


def _ms_ssim_downsample2x(gray: np.ndarray) -> np.ndarray:
    """Low-pass (2×2 average) then subsample; same idea as Wang MS-SSIM reference code."""
    g = gray.astype(np.float64, copy=False)
    if _ndimage_uniform_filter is not None:
        smoothed = _ndimage_uniform_filter(g, size=2, mode="nearest")
    else:
        smoothed = cv2.blur(g, (2, 2))
    return smoothed[::2, ::2]


def _compute_ms_ssim_pair(
    real_gray: np.ndarray,
    dist_gray: np.ndarray,
    *,
    weights: Sequence[float] = _DEFAULT_MS_SSIM_WEIGHTS,
    win: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float = 1.0,
) -> float:
    """
    Multiscale Structural Similarity (MS-SSIM) for a single aligned grayscale pair in [0, 1].

    Z. Wang, E. P. Simoncelli, and A. C. Bovik, "Multiscale structural similarity for image quality
    assessment," in Proc. 37th Asilomar Conf. Signals, Syst. Comput., 2003, pp. 1398–1402.
    """
    ref = np.asarray(real_gray, dtype=np.float64)
    dist = np.asarray(dist_gray, dtype=np.float64)
    if ref.shape != dist.shape:
        raise ValueError("MS-SSIM expects aligned images of the same shape.")
    w_all = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w_all.size < 1 or float(w_all.sum()) <= 0:
        raise ValueError("weights must be non-empty with positive sum.")

    rh, rw = int(ref.shape[0]), int(ref.shape[1])
    max_levels = int(w_all.size)
    n_scales = 0
    for _ in range(max_levels):
        if min(rh, rw) < win:
            break
        n_scales += 1
        rh = (rh + 1) // 2
        rw = (rw + 1) // 2
    n_scales = max(1, min(n_scales, max_levels))

    w = w_all[:n_scales] / float(w_all[:n_scales].sum())
    product = 1.0
    for level in range(n_scales):
        ssim_m, cs_m = _ssim_cs_means_single_scale(
            ref, dist, win=win, win_sigma=win_sigma, L=L, K1=K1, K2=K2
        )
        if level < n_scales - 1:
            product *= float(cs_m ** w[level])
            ref = _ms_ssim_downsample2x(ref)
            dist = _ms_ssim_downsample2x(dist)
        else:
            product *= float(ssim_m ** w[level])

    return float(product)


def calculate_ms_ssim_score(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
    **kwargs,
) -> float:
    """
    Mean MS-SSIM over aligned reference / generated pairs (luminance, per image then average).

    Extra keyword args are passed to :func:`_compute_ms_ssim_pair` (e.g. ``weights``, ``win``).

    Z. Wang, E. P. Simoncelli, and A. C. Bovik, Asilomar 2003.
    """
    del device, batch_size
    real_bhwc = _ensure_rgb_uint8_bhwc(real_images)
    generated_bhwc = _ensure_rgb_uint8_bhwc(generated_images)
    if len(real_bhwc) != len(generated_bhwc):
        raise ValueError("real_images and generated_images must have the same length")
    scores: list[float] = []
    for real_rgb, generated_rgb in zip(real_bhwc, generated_bhwc):
        real_rgb, generated_rgb = _resize_rgb_pair(real_rgb, generated_rgb)
        g1 = _to_gray_float01(real_rgb)
        g2 = _to_gray_float01(generated_rgb)
        scores.append(_compute_ms_ssim_pair(g1, g2, **kwargs))
    return float(np.mean(scores))


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if gaussian_filter is not None:
        return gaussian_filter(image, sigma=sigma)
    ksize = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def _spectral_residual_saliency(gray_image: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(gray_image)
    log_amplitude = np.log(np.abs(spectrum) + EPS)
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    residual = log_amplitude - avg_log_amplitude
    saliency = np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))
    saliency = np.abs(saliency) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency -= saliency.min()
    saliency /= float(saliency.max() + EPS)
    return saliency


def _gradient_magnitude(gray_image: np.ndarray) -> np.ndarray:
    grad_x = cv2.Scharr(gray_image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray_image, cv2.CV_32F, 0, 1)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _gradient_magnitude_scharr_piq(gray_image: np.ndarray) -> np.ndarray:
    """Scharr magnitude with 1px zero pad; matches piq.gradient_map + conv2d(padding=1)."""
    gray = np.asarray(gray_image, dtype=np.float32)
    kx = (
        np.array([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=np.float32)
        / 16.0
    )
    ky = kx.T
    padded = np.pad(gray, 1, mode="constant", constant_values=0.0)
    gx = cv2.filter2D(padded, cv2.CV_32F, kx)[1:-1, 1:-1]
    gy = cv2.filter2D(padded, cv2.CV_32F, ky)[1:-1, 1:-1]
    return np.sqrt(gx * gx + gy * gy)


# SR-SIM (Zhang et al., ICIP 2012) aligned with MATLAB reference and piq.srsim
# https://github.com/photosynthesis-team/piq/blob/master/piq/srsim.py
_SR_SIM_C1 = 0.40
_SR_SIM_C2 = 225.0
_SR_SIM_GM_ALPHA = 0.50


def _sr_sim_similarity_map(map_x: np.ndarray, map_y: np.ndarray, constant: float) -> np.ndarray:
    """(2*x*y + C) / (x^2 + y^2 + C), same as piq.functional.similarity_map with alpha=0."""
    return (2.0 * map_x * map_y + constant) / (map_x * map_x + map_y * map_y + constant + EPS)


def _preprocess_luminance_sr_sim_piq(gray_255: np.ndarray) -> np.ndarray:
    """Optional anti-aliasing pool when min(H,W) is large (matches piq SR-SIM)."""
    h, w = gray_255.shape
    ksize = max(1, round(min(h, w) / 256))
    if ksize <= 1:
        return gray_255.astype(np.float64, copy=False)
    t = torch.from_numpy(gray_255.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    padding = ksize // 2
    up_pad = (ksize - 1) // 2
    down_pad = padding
    t = F.pad(t, (up_pad, down_pad, up_pad, down_pad), mode="replicate")
    t = F.avg_pool2d(t, ksize)
    return t.squeeze().numpy().astype(np.float64, copy=False)


def _spectral_residual_visual_saliency_sr_sim(
    gray_255: np.ndarray,
    *,
    scale: float = 0.25,
    kernel_size: int = 3,
    sigma: float = 3.8,
    gaussian_size: int = 10,
) -> np.ndarray:
    """
    Spectral residual visual saliency (Hou & Zhang CVPR07) matching piq SR-SIM reference
    (torch.fft + avg_pool2d on replicate-padded log-amplitude, then Gaussian blur + resize).
    Returns map in [0, 1], same spatial size as input gray_255.
    """
    gray_255 = np.asarray(gray_255, dtype=np.float64)
    h0, w0 = gray_255.shape
    x = torch.from_numpy(gray_255.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    eps = torch.finfo(torch.float32).eps

    nh = max(1, int(round(h0 * scale)))
    nw = max(1, int(round(w0 * scale)))
    if min(nh, nw) < max(kernel_size, gaussian_size):
        scale_eff = min(1.0, min(h0, w0) / float(max(kernel_size, gaussian_size) + 1))
        nh = max(1, int(round(h0 * scale_eff)))
        nw = max(1, int(round(w0 * scale_eff)))
    in_img = F.interpolate(x, size=(nh, nw), mode="area")

    imagefft = torch.fft.fft2(in_img)
    log_amplitude = torch.log(imagefft.abs() + eps)
    phase = torch.angle(imagefft)

    padding = kernel_size // 2
    up_pad = (kernel_size - 1) // 2
    down_pad = padding
    pad_hw = (up_pad, down_pad, up_pad, down_pad)
    padded_la = F.pad(log_amplitude, pad=pad_hw, mode="replicate")
    pooled = F.avg_pool2d(padded_la, kernel_size=kernel_size, stride=1)
    spectral_residual = log_amplitude - pooled

    comp = torch.complex(
        torch.exp(spectral_residual) * torch.cos(phase),
        torch.exp(spectral_residual) * torch.sin(phase),
    )
    saliency_map = torch.abs(torch.fft.ifft2(comp)) ** 2

    sm = saliency_map.squeeze(0).squeeze(0).detach().cpu().numpy()
    # Odd kernel for OpenCV; sigma matches piq SR-SIM defaults (see piq gaussian_size=10 handling).
    gk = gaussian_size if gaussian_size % 2 == 1 else gaussian_size + 1
    sm = cv2.GaussianBlur(sm.astype(np.float32), (gk, gk), sigmaX=sigma, sigmaY=sigma)
    smin = float(np.min(sm))
    smax = float(np.max(sm))
    sm = (sm - smin) / (smax - smin + EPS)
    sm = cv2.resize(sm, (w0, h0), interpolation=cv2.INTER_CUBIC)
    return sm.astype(np.float64)


def _compute_sr_sim_pair(real_gray: np.ndarray, generated_gray: np.ndarray) -> float:
    """
    SR-SIM with spectral-residual similarity, gradient-magnitude similarity (Scharr),
    and pooling sum(SVRS * GM**alpha * max(SVRS_ref, SVRS_dist)) / sum(max(SVRS_ref, SVRS_dist)).
    Luminance is scaled to [0, 255] for constants C1, C2 as in the reference implementation.
    """
    ref_255 = np.clip(real_gray.astype(np.float64) * 255.0, 0.0, 255.0)
    dist_255 = np.clip(generated_gray.astype(np.float64) * 255.0, 0.0, 255.0)

    ref_p = _preprocess_luminance_sr_sim_piq(ref_255)
    dist_p = _preprocess_luminance_sr_sim_piq(dist_255)

    svrs_x = _spectral_residual_visual_saliency_sr_sim(ref_p)
    svrs_y = _spectral_residual_visual_saliency_sr_sim(dist_p)

    grad_x = _gradient_magnitude_scharr_piq(ref_p.astype(np.float32))
    grad_y = _gradient_magnitude_scharr_piq(dist_p.astype(np.float32))

    s_sr = _sr_sim_similarity_map(svrs_x, svrs_y, _SR_SIM_C1)
    s_gm = _sr_sim_similarity_map(grad_x, grad_y, _SR_SIM_C2)
    svrs_max = np.maximum(svrs_x, svrs_y)
    score = s_sr * (s_gm ** _SR_SIM_GM_ALPHA) * svrs_max

    denom = float(np.sum(svrs_max)) + EPS
    return float(np.sum(score) / denom)


def _compute_fsim_pair(real_gray: np.ndarray, generated_gray: np.ndarray) -> float:
    t_pc = 0.85
    t_g = 160.0

    pc_real = _spectral_residual_saliency(real_gray)
    pc_generated = _spectral_residual_saliency(generated_gray)
    gm_real = _gradient_magnitude(real_gray)
    gm_generated = _gradient_magnitude(generated_gray)

    s_pc = (2.0 * pc_real * pc_generated + t_pc) / (pc_real * pc_real + pc_generated * pc_generated + t_pc)
    s_g = (2.0 * gm_real * gm_generated + t_g) / (gm_real * gm_real + gm_generated * gm_generated + t_g)
    weight = np.maximum(pc_real, pc_generated)
    return float(np.sum(s_pc * s_g * weight) / (np.sum(weight) + EPS))


def _compute_vif_pair(real_gray: np.ndarray, generated_gray: np.ndarray) -> float:
    sigma_nsq = 2.0
    numerator = 0.0
    denominator = 0.0
    ref = real_gray.astype(np.float32) * 255.0
    dist = generated_gray.astype(np.float32) * 255.0

    for scale in range(4):
        sigma = 1.2 * (2 ** scale)
        if scale > 0:
            ref = _gaussian_blur(ref, sigma=1.0)[::2, ::2]
            dist = _gaussian_blur(dist, sigma=1.0)[::2, ::2]

        mu_ref = _gaussian_blur(ref, sigma)
        mu_dist = _gaussian_blur(dist, sigma)

        sigma_ref_sq = _gaussian_blur(ref * ref, sigma) - mu_ref * mu_ref
        sigma_dist_sq = _gaussian_blur(dist * dist, sigma) - mu_dist * mu_dist
        sigma_ref_dist = _gaussian_blur(ref * dist, sigma) - mu_ref * mu_dist

        sigma_ref_sq = np.maximum(0.0, sigma_ref_sq)
        sigma_dist_sq = np.maximum(0.0, sigma_dist_sq)

        g = sigma_ref_dist / (sigma_ref_sq + EPS)
        sv_sq = sigma_dist_sq - g * sigma_ref_dist

        g = np.where(sigma_ref_sq < EPS, 0.0, g)
        sv_sq = np.where(sigma_ref_sq < EPS, sigma_dist_sq, sv_sq)
        sigma_ref_sq = np.maximum(sigma_ref_sq, EPS)
        sv_sq = np.maximum(sv_sq, EPS)

        numerator += float(np.sum(np.log1p((g * g) * sigma_ref_sq / (sv_sq + sigma_nsq))))
        denominator += float(np.sum(np.log1p(sigma_ref_sq / sigma_nsq)))

    if denominator <= EPS:
        return 1.0 if numerator <= EPS else 0.0
    return float(max(numerator / denominator, 0.0))


def _load_torchvision_inception_model(device: torch.device, as_feature_extractor: bool) -> torch.nn.Module:
    if not TORCHVISION_INCEPTION_AVAILABLE:
        raise ImportError("torchvision inception_v3 is not available.")
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=None, transform_input=False, init_weights=False).to(device)
    state_dict = load_state_dict_from_url(weights.url, progress=True, check_hash=False)
    model.load_state_dict(state_dict)
    if as_feature_extractor:
        model.fc = torch.nn.Identity()
    model.eval()
    return model


def _compute_pairwise_metric_lists(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    *,
    compute_lpips: bool,
    lpips_model: torch.nn.Module | None,
    device: torch.device,
    lpips_batch_size: int | None = None,
) -> dict[str, list[float | None]]:
    real_bhwc = _ensure_rgb_uint8_bhwc(real_images)
    generated_bhwc = _ensure_rgb_uint8_bhwc(generated_images)
    if len(real_bhwc) != len(generated_bhwc):
        raise ValueError("real_images and generated_images must have the same length")

    sr_sim_scores: list[float | None] = []
    fsim_scores: list[float | None] = []
    vif_scores: list[float | None] = []
    for real_rgb, generated_rgb in _progress(
        zip(real_bhwc, generated_bhwc),
        total=len(real_bhwc),
        desc="Pairwise SR-SIM/FSIM/VIF",
    ):
        real_rgb, generated_rgb = _resize_rgb_pair(real_rgb, generated_rgb)
        real_gray = _to_gray_float01(real_rgb)
        generated_gray = _to_gray_float01(generated_rgb)
        sr_sim_scores.append(_compute_sr_sim_pair(real_gray, generated_gray))
        fsim_scores.append(_compute_fsim_pair(real_gray, generated_gray))
        vif_scores.append(_compute_vif_pair(real_gray, generated_gray))

    if compute_lpips and lpips_model is not None:
        with torch.no_grad():
            batch_size = int(lpips_batch_size) if lpips_batch_size is not None else len(real_bhwc)
            batch_size = max(1, batch_size)
            lpips_scores = []
            starts = range(0, len(real_bhwc), batch_size)
            for start in _progress(
                starts,
                total=(len(real_bhwc) + batch_size - 1) // batch_size,
                desc=f"LPIPS ({batch_size}/batch)",
            ):
                real_batch = real_bhwc[start : start + batch_size]
                gen_batch = generated_bhwc[start : start + batch_size]
                real_tensor = _rgb_uint8_to_torch(real_batch, device=device, normalize_to_neg1=True)
                generated_tensor = _rgb_uint8_to_torch(gen_batch, device=device, normalize_to_neg1=True)
                distances = lpips_model(real_tensor, generated_tensor)
                lpips_scores.extend([float(v) for v in distances.reshape(-1).detach().cpu().tolist()])
    else:
        lpips_scores = [None] * len(real_bhwc)

    return {
        "sr_sim": sr_sim_scores,
        "lpips": lpips_scores,
        "fsim": fsim_scores,
        "vif": vif_scores,
    }


class InceptionV3FeatureExtractor:
    def __init__(self, device: str | torch.device = "cuda", resize_input: bool = True):
        if not PYTORCH_FID_AVAILABLE:
            raise ImportError("pytorch-fid is not installed.")
        self.device = torch.device(device)
        self.model = FIDInceptionV3(resize_input=resize_input).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, images: Sequence[str | Path] | np.ndarray, batch_size: int = 32) -> np.ndarray:
        image_bhwc = _ensure_rgb_uint8_bhwc(images)
        batches: list[np.ndarray] = []
        starts = range(0, len(image_bhwc), batch_size)
        for start in _progress(
            starts,
            total=(len(image_bhwc) + batch_size - 1) // batch_size,
            desc="FID features",
        ):
            batch = image_bhwc[start : start + batch_size]
            tensor = _rgb_uint8_to_torch(batch, device=self.device, normalize_to_neg1=False)
            activations = self.model(tensor)[0]
            activations = activations.squeeze(-1).squeeze(-1).detach().cpu().numpy()
            batches.append(activations)
        return np.concatenate(batches, axis=0)


class TorchvisionInceptionFeatureExtractor:
    def __init__(self, device: str | torch.device = "cuda"):
        if not TORCHVISION_INCEPTION_AVAILABLE:
            raise ImportError("torchvision inception_v3 is not available.")
        self.device = torch.device(device)
        self.model = _load_torchvision_inception_model(self.device, as_feature_extractor=True)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_features(self, images: Sequence[str | Path] | np.ndarray, batch_size: int = 32) -> np.ndarray:
        image_bhwc = _ensure_rgb_uint8_bhwc(images)
        batches: list[np.ndarray] = []
        starts = range(0, len(image_bhwc), batch_size)
        for start in _progress(
            starts,
            total=(len(image_bhwc) + batch_size - 1) // batch_size,
            desc="Inception features",
        ):
            batch = image_bhwc[start : start + batch_size]
            tensor = _rgb_uint8_to_torch(batch, device=self.device, normalize_to_neg1=False)
            tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            tensor = (tensor - self.mean) / self.std
            activations = self.model(tensor)
            if hasattr(activations, "logits"):
                activations = activations.logits
            elif isinstance(activations, tuple):
                activations = activations[0]
            batches.append(activations.detach().cpu().numpy())
        return np.concatenate(batches, axis=0)


class InceptionScoreModel:
    def __init__(self, device: str | torch.device = "cuda"):
        if not TORCHVISION_INCEPTION_AVAILABLE:
            raise ImportError("torchvision inception_v3 is not available.")
        self.device = torch.device(device)
        self.model = _load_torchvision_inception_model(self.device, as_feature_extractor=False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict_proba(self, images: Sequence[str | Path] | np.ndarray, batch_size: int = 32) -> np.ndarray:
        image_bhwc = _ensure_rgb_uint8_bhwc(images)
        probs: list[np.ndarray] = []
        starts = range(0, len(image_bhwc), batch_size)
        for start in _progress(
            starts,
            total=(len(image_bhwc) + batch_size - 1) // batch_size,
            desc="Inception Score probs",
        ):
            batch = image_bhwc[start : start + batch_size]
            tensor = _rgb_uint8_to_torch(batch, device=self.device, normalize_to_neg1=False)
            tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            tensor = (tensor - self.mean) / self.std
            logits = self.model(tensor)
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif isinstance(logits, tuple):
                logits = logits[0]
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(probs, axis=0)


def _frechet_distance_from_features(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)

    if PYTORCH_FID_AVAILABLE and calculate_frechet_distance is not None:
        return float(calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated))

    if not SCIPY_AVAILABLE or linalg is None:
        raise ImportError("scipy is required for FID fallback calculation.")

    covmean, _ = linalg.sqrtm(sigma_real @ sigma_generated, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_real - mu_generated
    return float(diff @ diff + np.trace(sigma_real + sigma_generated - 2.0 * covmean))


def calculate_fid_score(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
) -> float:
    if PYTORCH_FID_AVAILABLE:
        extractor = InceptionV3FeatureExtractor(device=device, resize_input=True)
    else:
        extractor = TorchvisionInceptionFeatureExtractor(device=device)
    real_features = extractor.extract_features(real_images, batch_size=batch_size)
    generated_features = extractor.extract_features(generated_images, batch_size=batch_size)
    return _frechet_distance_from_features(real_features, generated_features)


def calculate_sr_sim_score(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
) -> float:
    del device, batch_size
    result = summarize_metric_lists(
        _compute_pairwise_metric_lists(
            real_images,
            generated_images,
            compute_lpips=False,
            lpips_model=None,
            device=torch.device("cpu"),
        )
    )["sr_sim"]
    if result is None:
        raise RuntimeError("SR-SIM returned no valid values.")
    return result


def calculate_lpips_score(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 16,
    net: str = "alex",
) -> float:
    if lpips_lib is None:
        raise ImportError("lpips is not installed.")
    device = torch.device(device)
    model = lpips_lib.LPIPS(net=net).to(device).eval()
    pairwise = []
    real_bhwc = _ensure_rgb_uint8_bhwc(real_images)
    generated_bhwc = _ensure_rgb_uint8_bhwc(generated_images)
    for start in range(0, len(real_bhwc), batch_size):
        batch = _compute_pairwise_metric_lists(
            real_bhwc[start : start + batch_size],
            generated_bhwc[start : start + batch_size],
            compute_lpips=True,
            lpips_model=model,
            device=device,
        )["lpips"]
        pairwise.extend(batch)
    result = _mean_ignore_none(pairwise)
    if result is None:
        raise RuntimeError("LPIPS returned no valid values.")
    return result


def calculate_fsim_score(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
) -> float:
    del device, batch_size
    result = summarize_metric_lists(
        _compute_pairwise_metric_lists(
            real_images,
            generated_images,
            compute_lpips=False,
            lpips_model=None,
            device=torch.device("cpu"),
        )
    )["fsim"]
    if result is None:
        raise RuntimeError("FSIM returned no valid values.")
    return result


def calculate_vif_score(
    real_images: Sequence[str | Path] | np.ndarray,
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
) -> float:
    del device, batch_size
    result = summarize_metric_lists(
        _compute_pairwise_metric_lists(
            real_images,
            generated_images,
            compute_lpips=False,
            lpips_model=None,
            device=torch.device("cpu"),
        )
    )["vif"]
    if result is None:
        raise RuntimeError("VIF returned no valid values.")
    return result


def calculate_inception_score(
    generated_images: Sequence[str | Path] | np.ndarray,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
    splits: int = 10,
) -> tuple[float, float]:
    if splits <= 0:
        raise ValueError("splits must be positive")
    predictor = InceptionScoreModel(device=device)
    probs = predictor.predict_proba(generated_images, batch_size=batch_size)
    num_images = probs.shape[0]
    if num_images == 0:
        raise ValueError("No images provided for inception score")

    effective_splits = min(splits, num_images)
    split_scores = []
    for split in np.array_split(probs, effective_splits, axis=0):
        p_y = np.mean(split, axis=0, keepdims=True)
        kl = split * (np.log(split + EPS) - np.log(p_y + EPS))
        split_scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(split_scores)), float(np.std(split_scores))


@dataclass
class InferenceMetricSuite:
    device: str | torch.device = "cuda"
    lpips_net: str = "alex"
    fid_batch_size: int = 32
    inception_batch_size: int = 32
    inception_splits: int = 10

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self._lpips_model: torch.nn.Module | None = None

    def _get_lpips_model(self) -> torch.nn.Module | None:
        if lpips_lib is None:
            return None
        if self._lpips_model is None:
            try:
                self._lpips_model = lpips_lib.LPIPS(net=self.lpips_net).to(self.device).eval()
            except Exception as exc:
                print(f"[WARN] Failed to initialize LPIPS model: {exc}")
                return None
        return self._lpips_model

    def compute_pairwise_batch(
        self,
        real_images: Sequence[str | Path] | np.ndarray,
        generated_images: Sequence[str | Path] | np.ndarray,
    ) -> dict[str, list[float | None]]:
        return _compute_pairwise_metric_lists(
            real_images,
            generated_images,
            compute_lpips=True,
            lpips_model=self._get_lpips_model(),
            device=self.device,
        )

    def compute_dataset_metrics_from_paths(
        self,
        real_images: Sequence[str | Path],
        generated_images: Sequence[str | Path],
    ) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {
            "fid": None,
            "inception_score_mean": None,
            "inception_score_std": None,
        }

        try:
            metrics["fid"] = calculate_fid_score(
                real_images,
                generated_images,
                device=self.device,
                batch_size=self.fid_batch_size,
            )
        except Exception as exc:
            print(f"[WARN] Failed to compute FID: {exc}")

        try:
            is_mean, is_std = calculate_inception_score(
                generated_images,
                device=self.device,
                batch_size=self.inception_batch_size,
                splits=self.inception_splits,
            )
            metrics["inception_score_mean"] = is_mean
            metrics["inception_score_std"] = is_std
        except Exception as exc:
            print(f"[WARN] Failed to compute Inception Score: {exc}")

        return metrics


__all__ = [
    "InferenceMetricSuite",
    "InceptionV3FeatureExtractor",
    "calculate_fid_score",
    "calculate_fsim_score",
    "calculate_inception_score",
    "calculate_lpips_score",
    "calculate_ms_ssim_score",
    "calculate_sr_sim_score",
    "calculate_vif_score",
    "summarize_metric_lists",
]