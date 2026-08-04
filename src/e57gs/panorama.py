"""Render Realsee equirectangular panoramas into perspective views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import numpy.typing as npt

from .geometry import VirtualView, overlapping_virtual_views, perspective_intrinsics


@dataclass(frozen=True)
class RenderedView:
    panorama_index: int
    view_index: int
    panorama_guid: str
    scan_guid: str
    panorama_name: str
    image_name: str
    image_path: str
    ownership_mask_path: str
    training_mask_path: str
    yaw_deg: float
    pitch_deg: float
    camera_from_pano: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def perspective_rays(width: int, height: int, intrinsics: npt.ArrayLike) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    rows, columns = np.indices((height, width), dtype=np.float32)
    x = (columns + 0.5 - matrix[0, 2]) / matrix[0, 0]
    y = (rows + 0.5 - matrix[1, 2]) / matrix[1, 1]
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.astype(np.float32)


def panorama_maps(
    pano_width: int,
    pano_height: int,
    rays_camera: np.ndarray,
    view: VirtualView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return OpenCV source maps and panorama-space rays for one view."""

    rays_pano = rays_camera @ view.camera_from_pano.astype(np.float32)
    yaw = np.arctan2(rays_pano[..., 0], rays_pano[..., 2])
    pitch = -np.arctan2(
        rays_pano[..., 1],
        np.hypot(rays_pano[..., 0], rays_pano[..., 2]),
    )
    map_x = pano_width * (1.0 + yaw / np.pi) / 2.0 - 0.5
    map_y = pano_height * (1.0 - 2.0 * pitch / np.pi) / 2.0 - 0.5
    map_x = np.mod(map_x, pano_width).astype(np.float32)
    map_y = np.clip(map_y, -0.5, pano_height - 0.5).astype(np.float32)
    return map_x, map_y, rays_pano


def virtual_forward_directions(views: Iterable[VirtualView]) -> np.ndarray:
    forward = np.array([0.0, 0.0, 1.0])
    directions = [view.pano_from_camera @ forward for view in views]
    return np.asarray(directions, dtype=np.float32)


def render_panorama_views(
    panorama_path: Path,
    output_images_dir: Path,
    ownership_masks_dir: Path,
    training_masks_dir: Path,
    *,
    panorama_index: int,
    panorama_guid: str,
    scan_guid: str,
    panorama_name: str,
    output_size: int = 2048,
    fov_deg: float = 90.0,
    polar_cap_deg: float = 20.0,
    jpeg_quality: int = 95,
    views: tuple[VirtualView, ...] | None = None,
) -> list[RenderedView]:
    if views is None:
        views = overlapping_virtual_views()
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if panorama is None:
        raise ValueError(f"Unable to decode panorama: {panorama_path}")
    pano_height, pano_width = panorama.shape[:2]
    if pano_width != pano_height * 2:
        raise ValueError(f"Panorama is not 2:1: {panorama_path}")
    intrinsics = perspective_intrinsics(output_size, output_size, fov_deg)
    rays_camera = perspective_rays(output_size, output_size, intrinsics)
    forward_directions = virtual_forward_directions(views)
    latitude_limit = np.deg2rad(90.0 - polar_cap_deg)

    rendered: list[RenderedView] = []
    for view in views:
        map_x, map_y, rays_pano = panorama_maps(
            pano_width, pano_height, rays_camera, view
        )
        image = cv2.remap(
            panorama,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        flattened_rays = rays_pano.reshape(-1, 3)
        owner = np.argmax(flattened_rays @ forward_directions.T, axis=1).reshape(
            output_size, output_size
        )
        latitude = np.arctan2(
            -rays_pano[..., 1],
            np.hypot(rays_pano[..., 0], rays_pano[..., 2]),
        )
        quality_mask = np.abs(latitude) <= latitude_limit
        ownership_mask = (quality_mask & (owner == view.index)).astype(np.uint8) * 255
        training_mask = quality_mask.astype(np.uint8) * 255

        camera_dir = f"pano_camera{view.index:02d}"
        image_name = f"{camera_dir}/{panorama_index:02d}_{panorama_name}.jpg"
        image_path = output_images_dir / image_name
        ownership_path = ownership_masks_dir / f"{image_name}.png"
        training_path = training_masks_dir / f"{image_name}.png"
        for parent in (image_path.parent, ownership_path.parent, training_path.parent):
            parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ):
            raise OSError(f"Failed to write {image_path}")
        if not cv2.imwrite(str(ownership_path), ownership_mask):
            raise OSError(f"Failed to write {ownership_path}")
        if not cv2.imwrite(str(training_path), training_mask):
            raise OSError(f"Failed to write {training_path}")

        rendered.append(
            RenderedView(
                panorama_index=panorama_index,
                view_index=view.index,
                panorama_guid=panorama_guid,
                scan_guid=scan_guid,
                panorama_name=panorama_name,
                image_name=image_name,
                image_path=f"images/{image_name}",
                ownership_mask_path=f"masks_colmap/{image_name}.png",
                training_mask_path=f"masks_train/{image_name}.png",
                yaw_deg=view.yaw_deg,
                pitch_deg=view.pitch_deg,
                camera_from_pano=view.camera_from_pano.astype(float).tolist(),
            )
        )
    return rendered
