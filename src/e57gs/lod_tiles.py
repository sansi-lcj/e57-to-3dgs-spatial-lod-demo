"""Build deterministic additive LoD tiles from an OpenSplat Gaussian PLY."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .gaussian_ply import GaussianPly, read_gaussian_ply


def _voxel_codes(
    xyz: np.ndarray,
    origin: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    cells = np.floor((xyz - origin) / voxel_size_m).astype(np.int64)
    dimensions = cells.max(axis=0) + 1
    return cells[:, 0] + dimensions[0] * (
        cells[:, 1] + dimensions[1] * cells[:, 2]
    )


def nested_lod_indices(
    xyz: np.ndarray,
    *,
    origin: np.ndarray,
    base_voxel_m: float,
    mid_voxel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition indices into disjoint base, mid-delta and fine-delta sets."""

    points = np.asarray(xyz, dtype=np.float64)
    world_origin = np.asarray(origin, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError("xyz must not be empty")
    if not np.isfinite(points).all() or not np.isfinite(world_origin).all():
        raise ValueError("xyz and origin must be finite")
    if not 0 < mid_voxel_m < base_voxel_m:
        raise ValueError("voxel sizes must satisfy 0 < mid < base")

    mid_codes = _voxel_codes(points, world_origin, mid_voxel_m)
    _, mid_first = np.unique(mid_codes, return_index=True)
    mid_representatives = np.sort(mid_first.astype(np.int64, copy=False))

    base_codes = _voxel_codes(
        points[mid_representatives], world_origin, base_voxel_m
    )
    _, base_first_in_mid = np.unique(base_codes, return_index=True)
    base_indices = np.sort(mid_representatives[base_first_in_mid])

    classes = np.full(len(points), 2, dtype=np.uint8)
    classes[mid_representatives] = 1
    classes[base_indices] = 0
    mid_delta_indices = np.flatnonzero(classes == 1).astype(np.int64, copy=False)
    fine_delta_indices = np.flatnonzero(classes == 2).astype(np.int64, copy=False)
    return base_indices, mid_delta_indices, fine_delta_indices


def _header_with_vertex_count(model: GaussianPly, count: int) -> bytes:
    return re.sub(
        rb"(?m)^element vertex \d+\r?$",
        f"element vertex {count}".encode("ascii"),
        model.header,
        count=1,
    )


def _write_subset(
    path: Path,
    model: GaussianPly,
    vertices: np.memmap,
    indices: np.ndarray,
    *,
    tangent_scale_floor_m: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("xb") as destination:
            destination.write(_header_with_vertex_count(model, len(indices)))
            for start in range(0, len(indices), 100_000):
                chunk = vertices[indices[start : start + 100_000]]
                if tangent_scale_floor_m is not None:
                    chunk = chunk.copy()
                    log_scales = np.column_stack(
                        [chunk[f"scale_{axis}"] for axis in range(3)]
                    )
                    axis_order = np.argsort(log_scales, axis=1)
                    floor = np.float32(np.log(tangent_scale_floor_m))
                    for rank in (1, 2):
                        ranked_axes = axis_order[:, rank]
                        for axis in range(3):
                            mask = ranked_axes == axis
                            chunk[f"scale_{axis}"][mask] = np.maximum(
                                chunk[f"scale_{axis}"][mask], floor
                            )
                destination.write(chunk.tobytes())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _tile_assets(
    *,
    level_id: str,
    indices: np.ndarray,
    xyz: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    tile_size_m: float,
    output_dir: Path,
    model: GaussianPly,
    vertices: np.memmap,
    tangent_scale_floor_m: float | None,
) -> list[dict[str, Any]]:
    tile_xy = np.floor((xyz[indices, :2] - bounds_min[:2]) / tile_size_m).astype(
        np.int32
    )
    grid_width = int(math.floor((bounds_max[0] - bounds_min[0]) / tile_size_m)) + 1
    tile_ids = tile_xy[:, 0].astype(np.int64) + grid_width * tile_xy[:, 1]
    order = np.argsort(tile_ids, kind="stable")
    sorted_ids = tile_ids[order]
    sorted_indices = indices[order]
    split_points = np.flatnonzero(np.diff(sorted_ids)) + 1
    groups = np.split(sorted_indices, split_points)
    ids = np.split(sorted_ids, split_points)
    assets: list[dict[str, Any]] = []
    for group, group_ids in zip(groups, ids, strict=True):
        tile_id = int(group_ids[0])
        tile_y, tile_x = divmod(tile_id, grid_width)
        stem = f"x{tile_x:02d}_y{tile_y:02d}"
        relative_ply = Path(level_id) / f"{stem}.ply"
        _write_subset(
            output_dir / relative_ply,
            model,
            vertices,
            group,
            tangent_scale_floor_m=tangent_scale_floor_m,
        )
        lower = np.asarray(
            [
                bounds_min[0] + tile_x * tile_size_m,
                bounds_min[1] + tile_y * tile_size_m,
                bounds_min[2],
            ]
        )
        upper = np.minimum(
            lower + np.asarray([tile_size_m, tile_size_m, bounds_max[2] - bounds_min[2]]),
            bounds_max,
        )
        assets.append(
            {
                "id": stem,
                "sourcePath": str((output_dir / relative_ply).resolve()),
                "url": str(relative_ply.with_suffix(".spz")).replace(os.sep, "/"),
                "splats": int(len(group)),
                "bounds": {
                    "min": lower.astype(float).tolist(),
                    "max": upper.astype(float).tolist(),
                },
            }
        )
    return assets


def build_additive_lod_plys(
    input_path: Path,
    output_dir: Path,
    *,
    base_voxel_m: float = 0.06,
    mid_voxel_m: float = 0.02,
    tile_size_m: float = 2.0,
    mid_load_distance_m: float = 4.5,
    mid_unload_distance_m: float = 5.5,
    fine_load_distance_m: float = 2.75,
    fine_unload_distance_m: float = 3.6,
    base_tangent_scale_m: float | None = None,
    mid_tangent_scale_m: float | None = None,
    mid_view_distance_m: float = 14.0,
    mid_view_half_angle_degrees: float = 85.0,
    fine_view_distance_m: float = 14.0,
    fine_view_half_angle_degrees: float = 65.0,
) -> dict[str, Any]:
    """Write disjoint additive LoD PLYs and a source conversion manifest."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive")
    if base_tangent_scale_m is not None and base_tangent_scale_m <= 0:
        raise ValueError("base LoD tangent scale must be positive")
    if mid_tangent_scale_m is not None and mid_tangent_scale_m <= 0:
        raise ValueError("mid LoD tangent scale must be positive")
    for distance, angle in (
        (mid_view_distance_m, mid_view_half_angle_degrees),
        (fine_view_distance_m, fine_view_half_angle_degrees),
    ):
        if distance <= 0 or not 0 < angle < 180:
            raise ValueError("view distance and half angle must be positive")
    if not 0 < fine_load_distance_m < fine_unload_distance_m:
        raise ValueError("fine load distance must be below unload distance")
    if not 0 < mid_load_distance_m < mid_unload_distance_m:
        raise ValueError("mid load distance must be below unload distance")

    model = read_gaussian_ply(input_path)
    vertices = np.memmap(
        input_path,
        mode="r",
        dtype=model.dtype,
        offset=model.data_offset,
        shape=model.vertex_count,
    )
    xyz = np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float64)
    bounds_min = xyz.min(axis=0)
    bounds_max = xyz.max(axis=0)
    base_indices, mid_indices, fine_indices = nested_lod_indices(
        xyz,
        origin=bounds_min,
        base_voxel_m=base_voxel_m,
        mid_voxel_m=mid_voxel_m,
    )
    output_dir.mkdir(parents=True)
    base_ply = output_dir / "d0" / "base.ply"
    _write_subset(
        base_ply,
        model,
        vertices,
        base_indices,
        tangent_scale_floor_m=base_tangent_scale_m,
    )

    levels = [
        {
            "id": "d1",
            "refine": "ADD",
            "loadDistanceM": mid_load_distance_m,
            "unloadDistanceM": mid_unload_distance_m,
            "viewDistanceM": mid_view_distance_m,
            "viewHalfAngleDegrees": mid_view_half_angle_degrees,
            "retainViewDistanceM": mid_view_distance_m + tile_size_m,
            "retainViewHalfAngleDegrees": min(179.0, mid_view_half_angle_degrees + 12.0),
            "assets": _tile_assets(
                level_id="d1",
                indices=mid_indices,
                xyz=xyz,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                tile_size_m=tile_size_m,
                output_dir=output_dir,
                model=model,
                vertices=vertices,
                tangent_scale_floor_m=mid_tangent_scale_m,
            ),
        },
        {
            "id": "d2",
            "refine": "ADD",
            "loadDistanceM": fine_load_distance_m,
            "unloadDistanceM": fine_unload_distance_m,
            "viewDistanceM": fine_view_distance_m,
            "viewHalfAngleDegrees": fine_view_half_angle_degrees,
            "retainViewDistanceM": fine_view_distance_m + tile_size_m,
            "retainViewHalfAngleDegrees": min(179.0, fine_view_half_angle_degrees + 12.0),
            "assets": _tile_assets(
                level_id="d2",
                indices=fine_indices,
                xyz=xyz,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                tile_size_m=tile_size_m,
                output_dir=output_dir,
                model=model,
                vertices=vertices,
                tangent_scale_floor_m=None,
            ),
        },
    ]
    manifest: dict[str, Any] = {
        "version": 1,
        "scheme": "additive-spatial-lod",
        "coordinateSystem": "right-handed-z-up-metres",
        "source": str(input_path.resolve()),
        "totalSplats": model.vertex_count,
        "tileSizeM": tile_size_m,
        "surfaceScaleM": {
            "baseTangentFloor": base_tangent_scale_m,
            "midTangentFloor": mid_tangent_scale_m,
        },
        "bounds": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
        },
        "base": {
            "id": "base",
            "sourcePath": str(base_ply.resolve()),
            "url": "d0/base.spz",
            "splats": int(len(base_indices)),
            "bounds": {
                "min": bounds_min.astype(float).tolist(),
                "max": bounds_max.astype(float).tolist(),
            },
        },
        "levels": levels,
        "counts": {
            "base": int(len(base_indices)),
            "midDelta": int(len(mid_indices)),
            "fineDelta": int(len(fine_indices)),
        },
    }
    if sum(manifest["counts"].values()) != model.vertex_count:
        raise AssertionError("LoD levels do not partition the source vertices")
    manifest_path = output_dir / "manifest.source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "manifest": str(manifest_path.resolve()),
        "total_splats": model.vertex_count,
        "counts": manifest["counts"],
        "tiles": {level["id"]: len(level["assets"]) for level in levels},
        "bounds": manifest["bounds"],
    }
