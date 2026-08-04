"""Geometry and color reprojection checks for generated datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pye57

from .points import PointCloud, read_scan_fields


def _nearest_panorama_x(source_x: np.ndarray, width: int) -> np.ndarray:
    """Quantize cyclic panorama coordinates without producing column ``width``."""

    return np.mod(np.rint(source_x).astype(np.int64), width)


def _visible_pixel_samples(
    xyz_camera: np.ndarray,
    rgb: np.ndarray,
    *,
    width: int,
    height: int,
    focal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = xyz_camera[:, 2]
    valid = np.isfinite(depth) & (depth > 1e-4)
    points = xyz_camera[valid]
    colors = rgb[valid]
    depth = depth[valid]
    pixel_x = np.rint(focal * points[:, 0] / depth + width / 2.0 - 0.5).astype(np.int64)
    pixel_y = np.rint(focal * points[:, 1] / depth + height / 2.0 - 0.5).astype(np.int64)
    inside = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    pixel_x, pixel_y, depth, colors = (
        pixel_x[inside],
        pixel_y[inside],
        depth[inside],
        colors[inside],
    )
    linear = pixel_y * width + pixel_x
    order = np.argsort(depth, kind="stable")
    _, first = np.unique(linear[order], return_index=True)
    visible = order[first]
    return pixel_x[visible], pixel_y[visible], depth[visible], colors[visible]


def crop_reprojection_overlay(
    cloud: PointCloud,
    image_path: Path,
    output_path: Path,
    rotation_cw: np.ndarray,
    translation_cw: np.ndarray,
    focal: float,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read crop image: {image_path}")
    height, width = image.shape[:2]
    xyz_camera = cloud.xyz.astype(np.float64) @ rotation_cw.T + translation_cw
    pixel_x, pixel_y, depth, colors_rgb = _visible_pixel_samples(
        xyz_camera,
        cloud.rgb,
        width=width,
        height=height,
        focal=focal,
    )
    colors_bgr = colors_rgb[:, ::-1]
    sampled = image[pixel_y, pixel_x]
    color_error = np.abs(sampled.astype(np.int16) - colors_bgr.astype(np.int16))

    rendered = np.zeros_like(image)
    mask = np.zeros((height, width), dtype=np.uint8)
    rendered[pixel_y, pixel_x] = colors_bgr
    mask[pixel_y, pixel_x] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)
    dilated_render = cv2.dilate(rendered, kernel)
    overlay = image.copy()
    selected = dilated_mask > 0
    overlay[selected] = np.rint(
        0.45 * image[selected].astype(np.float32)
        + 0.55 * dilated_render[selected].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Failed to write {output_path}")
    return {
        "image": str(image_path),
        "overlay": str(output_path),
        "visible_pixels": int(len(pixel_x)),
        "pixel_coverage": float(len(pixel_x) / (width * height)),
        "depth_min_m": float(depth.min()) if len(depth) else None,
        "depth_max_m": float(depth.max()) if len(depth) else None,
        "color_mae_bgr": color_error.mean(axis=0).astype(float).tolist() if len(color_error) else None,
        "color_mae_mean": float(color_error.mean()) if len(color_error) else None,
    }


def scan_panorama_reprojection(
    e57_path: Path,
    scan_index: int,
    panorama_path: Path,
    output_path: Path,
    preview_width: int = 3000,
) -> dict[str, Any]:
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if panorama is None:
        raise ValueError(f"Unable to read panorama: {panorama_path}")
    height, width = panorama.shape[:2]
    with pye57.E57(str(e57_path.resolve())) as e57:
        fields = read_scan_fields(e57, scan_index)
    xyz = np.column_stack(
        [fields["cartesianX"], fields["cartesianY"], fields["cartesianZ"]]
    )
    rgb = np.column_stack([fields["colorRed"], fields["colorGreen"], fields["colorBlue"]])
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    theta = np.arctan2(xyz[:, 0], xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], np.hypot(xyz[:, 0], xyz[:, 2]))
    source_x = np.mod(width * (theta / (2 * np.pi) + 0.5) - 0.5, width)
    source_y = np.clip(height * (phi / np.pi + 0.5) - 0.5, 0, height - 1)
    # Equirectangular longitude is cyclic. Values just below ``width`` can
    # round to ``width`` at the nearest-pixel step, so wrap after rounding as
    # well as before it.
    pixel_x = _nearest_panorama_x(source_x, width)
    pixel_y = np.rint(source_y).astype(np.int64)
    sampled_bgr = panorama[pixel_y, pixel_x]
    color_error = np.abs(sampled_bgr.astype(np.int16) - rgb[:, ::-1].astype(np.int16))

    preview_height = preview_width // 2
    preview = cv2.resize(panorama, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    px = np.clip(np.rint(source_x * preview_width / width).astype(np.int64), 0, preview_width - 1)
    py = np.clip(np.rint(source_y * preview_height / height).astype(np.int64), 0, preview_height - 1)
    distance = np.linalg.norm(xyz, axis=1)
    linear = py * preview_width + px
    order = np.argsort(distance, kind="stable")
    _, first = np.unique(linear[order], return_index=True)
    visible = order[first]
    point_layer = np.zeros_like(preview)
    point_mask = np.zeros((preview_height, preview_width), dtype=np.uint8)
    point_layer[py[visible], px[visible]] = rgb[visible, ::-1]
    point_mask[py[visible], px[visible]] = 255
    point_layer = cv2.dilate(point_layer, np.ones((3, 3), dtype=np.uint8))
    point_mask = cv2.dilate(point_mask, np.ones((3, 3), dtype=np.uint8))
    selected = point_mask > 0
    preview[selected] = np.rint(
        0.4 * preview[selected].astype(np.float32)
        + 0.6 * point_layer[selected].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Failed to write {output_path}")
    return {
        "scan_index": scan_index,
        "panorama": str(panorama_path),
        "overlay": str(output_path),
        "point_count": int(len(xyz)),
        "preview_visible_pixels": int(len(visible)),
        "color_mae_bgr": color_error.mean(axis=0).astype(float).tolist(),
        "color_mae_mean": float(color_error.mean()),
        "color_median_abs_error": float(np.median(color_error)),
    }
