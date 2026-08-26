#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


@dataclass
class CameraParams:
    target_size: int
    latent_factor: int
    latent_size: int
    phase_x: int
    phase_y: int
    wb_red_gain: float
    wb_blue_gain: float
    optical_blur_radius: float
    motion_length: int
    motion_angle_deg: float
    shot_noise_scale: float
    dark_current_scale: float
    read_noise_std: float
    dark_noise_boost: float
    column_noise_std: float
    row_noise_std: float
    hot_pixel_prob: float
    hot_pixel_strength: float
    banding_strength: float
    banding_period: float
    bit_depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage camera sampling pipeline for ultra-low-resolution face generation."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of GT images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write outputs.")
    parser.add_argument(
        "--target-size",
        type=int,
        default=32,
        choices=[8, 12, 16, 24, 32, 64],
        help="Sensor-domain spatial size.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Process only the first N images after sorting.",
    )
    parser.add_argument("--seed", type=int, default=20260409, help="Random seed.")
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Global degradation strength multiplier.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["jpg", "jpeg", "png", "webp"],
        help="Image extensions to scan.",
    )
    parser.add_argument(
        "--isp-gamma",
        type=float,
        default=0.92,
        help="Weak ISP gamma exponent applied after raw demosaic.",
    )
    parser.add_argument(
        "--no-surveillance-grade",
        action="store_true",
        help="Disable monitor-style desat/dim on final LQ (after ISP).",
    )
    return parser.parse_args()


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(np.float32) / 255.0


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def sample_params(rng: np.random.Generator, target_size: int, strength: float) -> CameraParams:
    severity = (16.0 / target_size) * strength
    latent_factor = int(rng.integers(20, 33))
    latent_size = target_size * latent_factor + latent_factor
    motion_length = int(rng.choice([0, 0, 1, 2, 3, 4, 5]))
    return CameraParams(
        target_size=target_size,
        latent_factor=latent_factor,
        latent_size=latent_size,
        phase_x=int(rng.integers(0, latent_factor)),
        phase_y=int(rng.integers(0, latent_factor)),
        wb_red_gain=float(rng.uniform(0.97, 1.03)),
        wb_blue_gain=float(rng.uniform(0.97, 1.03)),
        optical_blur_radius=float(rng.uniform(0.45, 1.20 * max(1.0, severity))),
        motion_length=motion_length,
        motion_angle_deg=float(rng.uniform(0.0, 180.0)),
        shot_noise_scale=float(rng.uniform(85.0 / severity, 170.0 / severity)),
        dark_current_scale=float(rng.uniform(0.4, 2.5 * severity + 0.4)),
        read_noise_std=float(rng.uniform(0.0012, 0.0038 * severity + 0.0012)),
        dark_noise_boost=float(rng.uniform(2.0, 4.2)),
        column_noise_std=float(rng.uniform(0.0007, 0.0035 * severity + 0.0007)),
        row_noise_std=float(rng.uniform(0.0002, 0.0012 * severity + 0.0003)),
        hot_pixel_prob=float(rng.uniform(0.0008, 0.0035 + 0.0015 * severity)),
        hot_pixel_strength=float(rng.uniform(0.015, 0.07 + 0.02 * severity)),
        banding_strength=float(rng.uniform(0.0015, 0.007 * severity + 0.0015)),
        banding_period=float(rng.uniform(2.5, 6.0)),
        bit_depth=int(rng.choice([6, 7, 8, 8])),
    )


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)

    size = max(3, length if length % 2 == 1 else length + 1)
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    angle = np.deg2rad(angle_deg)
    samples = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, num=max(length * 4, 8))
    xs = center + samples * np.cos(angle)
    ys = center + samples * np.sin(angle)
    xs = np.clip(np.round(xs).astype(int), 0, size - 1)
    ys = np.clip(np.round(ys).astype(int), 0, size - 1)
    kernel[ys, xs] += 1.0
    kernel /= kernel.sum()
    return kernel


def convolve_rgb(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.shape == (1, 1):
        return arr
    pad = kernel.shape[0] // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(arr)
    height, width = arr.shape[:2]
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            weight = kernel[ky, kx]
            if weight == 0.0:
                continue
            out += weight * padded[ky : ky + height, kx : kx + width]
    return out


def apply_optics(image: Image.Image, params: CameraParams) -> Image.Image:
    image = image.filter(ImageFilter.GaussianBlur(radius=params.optical_blur_radius))
    if params.motion_length > 1:
        arr = pil_to_array(image)
        arr = convolve_rgb(arr, motion_blur_kernel(params.motion_length, params.motion_angle_deg))
        image = array_to_pil(arr)
    return image


def apply_weak_color_shift(image: Image.Image, params: CameraParams) -> Image.Image:
    arr = pil_to_array(image)
    arr[..., 0] *= params.wb_red_gain
    arr[..., 2] *= params.wb_blue_gain
    return array_to_pil(np.clip(arr, 0.0, 1.0))


def area_integrate_with_phase(image: Image.Image, params: CameraParams) -> Image.Image:
    arr = pil_to_array(image)
    pad = params.latent_factor
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    start_x = pad + params.phase_x
    start_y = pad + params.phase_y
    sensor_span = params.target_size * params.latent_factor
    window = padded[start_y : start_y + sensor_span, start_x : start_x + sensor_span]
    return array_to_pil(window).resize(
        (params.target_size, params.target_size),
        Image.Resampling.BOX,
    )


def convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    height, width = arr.shape
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            weight = float(kernel[ky, kx])
            if weight == 0.0:
                continue
            out += weight * padded[ky : ky + height, kx : kx + width]
    return out


def make_rggb_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_mask = np.zeros((height, width), dtype=np.float32)
    g_mask = np.zeros((height, width), dtype=np.float32)
    b_mask = np.zeros((height, width), dtype=np.float32)
    r_mask[0::2, 0::2] = 1.0
    g_mask[0::2, 1::2] = 1.0
    g_mask[1::2, 0::2] = 1.0
    b_mask[1::2, 1::2] = 1.0
    return r_mask, g_mask, b_mask


def rgb_to_bayer_raw(arr: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    height, width = arr.shape[:2]
    r_mask, g_mask, b_mask = make_rggb_masks(height, width)
    raw = arr[..., 0] * r_mask + arr[..., 1] * g_mask + arr[..., 2] * b_mask
    return raw.astype(np.float32), (r_mask, g_mask, b_mask)


def demosaic_bilinear(raw: np.ndarray, masks: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    channels: list[np.ndarray] = []
    for mask in masks:
        weighted = convolve2d(raw * mask, kernel)
        weights = convolve2d(mask, kernel)
        channels.append(weighted / np.maximum(weights, 1e-6))
    return np.stack(channels, axis=-1)


def make_local_banding(
    shape: tuple[int, int],
    rng: np.random.Generator,
    params: CameraParams,
) -> np.ndarray:
    height, width = shape
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    vertical = bool(rng.integers(0, 2))
    coord = xx if vertical else yy
    orth = yy if vertical else xx
    center = float(rng.uniform(-0.3, 0.3))
    sigma = float(rng.uniform(0.25, 0.55))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    envelope = np.exp(-0.5 * ((orth - center) / sigma) ** 2).astype(np.float32)
    stripe = np.sin(2.0 * np.pi * coord * params.banding_period + phase).astype(np.float32)
    return params.banding_strength * envelope * stripe


def add_raw_sensor_noise(
    rgb_arr: np.ndarray,
    rng: np.random.Generator,
    params: CameraParams,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    safe_rgb = np.clip(rgb_arr, 0.0, 1.0)
    raw, masks = rgb_to_bayer_raw(safe_rgb)
    dark_weight = np.power(1.0 - np.clip(raw, 0.0, 1.0), 1.8).astype(np.float32)

    signal_counts = np.clip(raw, 0.0, 1.0) * params.shot_noise_scale
    signal_raw = rng.poisson(signal_counts).astype(np.float32) / params.shot_noise_scale
    dark_counts = rng.poisson(dark_weight * params.dark_current_scale).astype(np.float32) / params.shot_noise_scale

    read_std_map = params.read_noise_std * (0.30 + params.dark_noise_boost * dark_weight)
    read_noise = rng.normal(0.0, 1.0, size=raw.shape).astype(np.float32) * read_std_map

    column_offsets = rng.normal(0.0, params.column_noise_std, size=(1, raw.shape[1])).astype(np.float32)
    row_offsets = rng.normal(0.0, params.row_noise_std, size=(raw.shape[0], 1)).astype(np.float32)
    fixed_pattern = (column_offsets + row_offsets) * (0.15 + 0.75 * dark_weight)

    banding = make_local_banding(raw.shape, rng, params) * (0.15 + 0.55 * dark_weight)

    hot_mask = rng.random(raw.shape) < params.hot_pixel_prob
    hot_strength = rng.uniform(
        0.35 * params.hot_pixel_strength,
        params.hot_pixel_strength,
        size=raw.shape,
    ).astype(np.float32)
    hot_pixels = hot_mask.astype(np.float32) * hot_strength * (0.15 + 0.85 * dark_weight)

    noisy_raw = signal_raw + dark_counts + read_noise + fixed_pattern + banding + hot_pixels
    return np.clip(noisy_raw, 0.0, 1.0), masks


def quantize_to_bit_depth(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    levels = (1 << bit_depth) - 1
    return np.round(np.clip(arr, 0.0, 1.0) * levels) / levels


def apply_weak_isp(arr: np.ndarray, gamma: float) -> np.ndarray:
    safe = np.clip(arr, 0.0, 1.0).astype(np.float32)
    channel_mean = safe.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_mean.mean())
    gains = np.clip(gray_mean / np.maximum(channel_mean, 1e-4), 0.94, 1.06)
    gains = 1.0 + 0.55 * (gains - 1.0)
    balanced = np.clip(safe * gains.reshape(1, 1, 3), 0.0, 1.0)
    balanced = np.clip((balanced - 0.01) / 0.99, 0.0, 1.0)
    return np.clip(np.power(balanced, gamma), 0.0, 1.0)


def apply_surveillance_monitor_grade(
    pil: Image.Image,
    rng: np.random.Generator,
    strength: float,
) -> Image.Image:
    """
    Simulate a typical surveillance-camera endpoint look: slightly lower saturation
    and slightly darker brightness (low-cost ISP + DVR preview, AGC tends dim).
    Uses PIL Color/Brightness enhancement instead of cinematic color grading.

    strength: same scale as global degradation intensity; no change when <= 0.
    """
    if strength <= 0.0:
        return pil
    s = float(strength)
    # Small random jitter to avoid identical look across all images.
    sat_pull = float(rng.uniform(0.06, 0.18)) * s
    bright_pull = float(rng.uniform(0.03, 0.11)) * s
    sat_factor = max(0.82, 1.0 - sat_pull)
    bright_factor = max(0.80, 1.0 - bright_pull)
    out = ImageEnhance.Color(pil.convert("RGB")).enhance(sat_factor)
    out = ImageEnhance.Brightness(out).enhance(bright_factor)
    return out


def degrade_image(
    image: Image.Image,
    rng: np.random.Generator,
    params: CameraParams,
    isp_gamma: float,
    *,
    surveillance_grade: bool = True,
    surveillance_strength: float = 1.0,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    image = center_crop_square(image).convert("RGB")
    image = apply_weak_color_shift(image, params)
    optical_hr = apply_optics(image.copy(), params)
    latent = image.resize((params.latent_size, params.latent_size), Image.Resampling.LANCZOS)
    latent = apply_optics(latent, params)
    sampled_clean = area_integrate_with_phase(latent, params)

    noisy_raw, masks = add_raw_sensor_noise(pil_to_array(sampled_clean), rng, params)
    sampled_no_quant = array_to_pil(demosaic_bilinear(noisy_raw, masks))
    quantized_raw = quantize_to_bit_depth(noisy_raw, params.bit_depth)
    lq_rgb = apply_weak_isp(demosaic_bilinear(quantized_raw, masks), isp_gamma)
    lq = array_to_pil(lq_rgb)
    # Surveillance look is applied after sensor + ISP to the final LQ only.
    if surveillance_grade:
        lq = apply_surveillance_monitor_grade(lq, rng, surveillance_strength)
    return optical_hr, sampled_clean, sampled_no_quant, lq


def restore_bicubic(lq: Image.Image, output_size: tuple[int, int]) -> Image.Image:
    return lq.resize(output_size, Image.Resampling.BICUBIC)


def rgb_to_luma(arr: np.ndarray) -> np.ndarray:
    return (
        0.2990 * arr[..., 0]
        + 0.5870 * arr[..., 1]
        + 0.1140 * arr[..., 2]
    ).astype(np.float32)


def normalize_map(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    scale = arr.max()
    if scale < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / scale


def heatmap_rgb(arr: np.ndarray) -> np.ndarray:
    t = np.clip(arr, 0.0, 1.0).astype(np.float32)
    red = np.clip(1.6 * t - 0.15, 0.0, 1.0)
    green = np.clip(1.8 * (1.0 - np.abs(t - 0.55)), 0.0, 1.0)
    blue = np.clip(1.2 - 2.4 * t, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def compute_spectrum_map(image: Image.Image) -> np.ndarray:
    gray = rgb_to_luma(pil_to_array(image))
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft)).astype(np.float32)
    normalized = normalize_map(magnitude)
    return np.power(normalized, 0.85).astype(np.float32)


def spectrum_image_from_map(spectrum_map: np.ndarray, output_size: int) -> Image.Image:
    rgb = np.repeat(spectrum_map[..., None], 3, axis=-1)
    return array_to_pil(rgb).resize((output_size, output_size), Image.Resampling.NEAREST)


def diff_image_from_maps(map_a: np.ndarray, map_b: np.ndarray, output_size: int) -> Image.Image:
    diff = np.abs(map_a - map_b).astype(np.float32)
    scale = float(np.percentile(diff, 99.5))
    if scale > 1e-8:
        diff = np.clip(diff / scale, 0.0, 1.0)
    else:
        diff = np.zeros_like(diff, dtype=np.float32)
    return array_to_pil(heatmap_rgb(diff)).resize((output_size, output_size), Image.Resampling.NEAREST)


def make_labeled_row(images: list[Image.Image], labels: list[str]) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("images must not be empty")

    panel_width = images[0].width
    panel_height = images[0].height
    label_height = max(26, panel_height // 9)
    canvas = Image.new("RGB", (panel_width * len(images), panel_height + label_height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (image, label) in enumerate(zip(images, labels)):
        x0 = idx * panel_width
        canvas.paste(image, (x0, label_height))
        draw.rectangle((x0, 0, x0 + panel_width, label_height), fill=(18, 18, 18))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x0 + max(8, (panel_width - text_w) // 2)
        text_y = max(4, (label_height - text_h) // 2 - 1)
        draw.text((text_x, text_y), label, fill=(236, 236, 236), font=font)
    return canvas


def make_spectrum_preview(
    clean_spectrum: Image.Image,
    noisy_spectrum: Image.Image,
    lq_spectrum: Image.Image,
) -> Image.Image:
    return make_labeled_row(
        [clean_spectrum, noisy_spectrum, lq_spectrum],
        ["Clean Spectrum", "Noisy Spectrum", "Final LQ Spectrum"],
    )


def make_spectrum_diff_preview(
    clean_vs_noisy: Image.Image,
    noisy_vs_lq: Image.Image,
    clean_vs_lq: Image.Image,
) -> Image.Image:
    return make_labeled_row(
        [clean_vs_noisy, noisy_vs_lq, clean_vs_lq],
        ["Clean vs Noisy", "Noisy vs LQ", "Clean vs LQ"],
    )


def make_preview(
    optical_hr: Image.Image,
    clean_restored: Image.Image,
    noisy_restored: Image.Image,
    lq_restored: Image.Image,
    preview_size: int,
) -> Image.Image:
    optical_view = optical_hr.resize((preview_size, preview_size), Image.Resampling.LANCZOS)
    clean_view = clean_restored.resize((preview_size, preview_size), Image.Resampling.LANCZOS)
    noisy_view = noisy_restored.resize((preview_size, preview_size), Image.Resampling.LANCZOS)
    lq_view = lq_restored.resize((preview_size, preview_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (preview_size * 4, preview_size), color=(12, 12, 12))
    canvas.paste(optical_view, (0, 0))
    canvas.paste(clean_view, (preview_size, 0))
    canvas.paste(noisy_view, (preview_size * 2, 0))
    canvas.paste(lq_view, (preview_size * 3, 0))
    return canvas


def save_contact_sheet(previews: list[Image.Image], path: Path, tile_size: int) -> None:
    if not previews:
        return
    columns = 2
    rows = (len(previews) + columns - 1) // columns
    panel_count = previews[0].width // tile_size
    sheet = Image.new("RGB", (tile_size * columns * panel_count, tile_size * rows), color=(8, 8, 8))
    for idx, preview in enumerate(previews):
        row = idx // columns
        col = idx % columns
        sheet.paste(preview, (col * tile_size * panel_count, row * tile_size))
    sheet.save(path, quality=92)


def iter_images(input_dir: Path, extensions: Iterable[str]) -> list[Path]:
    suffixes = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def main() -> None:
    args = parse_args()
    image_paths = iter_images(args.input_dir, args.extensions)
    if args.num_samples is not None:
        image_paths = image_paths[: args.num_samples]
    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    rng = np.random.default_rng(args.seed)
    lq_dir = args.output_dir / f"lq_{args.target_size}"
    lq_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        params = sample_params(rng, args.target_size, args.strength)
        image = Image.open(image_path).convert("RGB")
        _optical_hr, _sampled_clean, _sampled_no_quant, lq = degrade_image(
            image,
            rng,
            params,
            args.isp_gamma,
            surveillance_grade=not bool(getattr(args, "no_surveillance_grade", False)),
            surveillance_strength=float(args.strength),
        )
        lq_path = lq_dir / f"{image_path.stem}.png"
        lq.save(lq_path)

    print(f"Processed {len(image_paths)} images into {lq_dir}")


if __name__ == "__main__":
    main()
