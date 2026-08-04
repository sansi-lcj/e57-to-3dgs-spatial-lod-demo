"""Decode, register, downsample, and write E57 point data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pye57
from pye57 import libe57

from .geometry import quaternion_wxyz_to_matrix


@dataclass(frozen=True)
class PointCloud:
    xyz: npt.NDArray[np.float32]
    rgb: npt.NDArray[np.uint8]
    normals: npt.NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        count = self.xyz.shape[0]
        if self.xyz.shape != (count, 3):
            raise ValueError("xyz must be shaped (N, 3)")
        if self.rgb.shape != (count, 3):
            raise ValueError("rgb must be shaped (N, 3)")
        if self.normals is not None and self.normals.shape != (count, 3):
            raise ValueError("normals must be shaped (N, 3)")


FIELD_DTYPES: dict[str, Any] = {
    "cartesianX": np.float64,
    "cartesianY": np.float64,
    "cartesianZ": np.float64,
    "colorRed": np.uint8,
    "colorGreen": np.uint8,
    "colorBlue": np.uint8,
    "cartesianInvalidState": np.int8,
    "nor:normalX": np.float32,
    "nor:normalY": np.float32,
    "nor:normalZ": np.float32,
}


def read_scan_fields(e57: pye57.E57, scan_index: int) -> dict[str, np.ndarray]:
    """Read standard fields and Realsee's NOR extension in one pass."""

    header = e57.get_header(scan_index)
    fields = [field for field in header.point_fields if field in FIELD_DTYPES]
    required = {"cartesianX", "cartesianY", "cartesianZ", "colorRed", "colorGreen", "colorBlue"}
    if not required.issubset(fields):
        missing = sorted(required.difference(fields))
        raise ValueError(f"Scan {scan_index} is missing required fields: {missing}")
    buffers = libe57.VectorSourceDestBuffer()
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        array = np.empty(header.point_count, dtype=FIELD_DTYPES[field])
        arrays[field] = array
        buffers.append(
            libe57.SourceDestBuffer(
                e57.image_file,
                field,
                array,
                header.point_count,
                True,
                True,
            )
        )
    read_count = header.points.reader(buffers).read()
    if read_count != header.point_count:
        raise ValueError(f"Scan {scan_index} read {read_count}/{header.point_count} points")
    return arrays


def registered_scan(e57: pye57.E57, scan_index: int) -> PointCloud:
    fields = read_scan_fields(e57, scan_index)
    header = e57.get_header(scan_index)
    xyz_local = np.column_stack(
        [fields["cartesianX"], fields["cartesianY"], fields["cartesianZ"]]
    )
    valid = np.all(np.isfinite(xyz_local), axis=1)
    if "cartesianInvalidState" in fields:
        valid &= fields["cartesianInvalidState"] == 0
    xyz_local = xyz_local[valid]
    rotation = quaternion_wxyz_to_matrix(header.rotation)
    translation = np.asarray(header.translation, dtype=np.float64)
    xyz_world = xyz_local @ rotation.T + translation
    rgb = np.column_stack(
        [fields["colorRed"][valid], fields["colorGreen"][valid], fields["colorBlue"][valid]]
    ).astype(np.uint8, copy=False)

    normal_fields = ("nor:normalX", "nor:normalY", "nor:normalZ")
    normals_world: np.ndarray | None = None
    if all(field in fields for field in normal_fields):
        normals_local = np.column_stack([fields[field][valid] for field in normal_fields])
        normals_world = normals_local @ rotation.T
        lengths = np.linalg.norm(normals_world, axis=1)
        nonzero = np.isfinite(lengths) & (lengths > 1e-12)
        normals_world[nonzero] /= lengths[nonzero, None]
        normals_world[~nonzero] = 0

    return PointCloud(
        xyz=xyz_world.astype(np.float32),
        rgb=rgb,
        normals=None if normals_world is None else normals_world.astype(np.float32),
    )


def registered_point_cloud(path: Path) -> tuple[PointCloud, list[dict[str, Any]]]:
    clouds: list[PointCloud] = []
    scan_stats: list[dict[str, Any]] = []
    with pye57.E57(str(path.resolve())) as e57:
        for scan_index in range(e57.scan_count):
            cloud = registered_scan(e57, scan_index)
            clouds.append(cloud)
            scan_stats.append(point_cloud_stats(cloud) | {"scan_index": scan_index})
    normals = None
    if all(cloud.normals is not None for cloud in clouds):
        normals = np.concatenate([cloud.normals for cloud in clouds if cloud.normals is not None])
    merged = PointCloud(
        xyz=np.concatenate([cloud.xyz for cloud in clouds]),
        rgb=np.concatenate([cloud.rgb for cloud in clouds]),
        normals=normals,
    )
    return merged, scan_stats


def voxel_downsample(cloud: PointCloud, voxel_size_m: float) -> PointCloud:
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    keys = np.floor(cloud.xyz.astype(np.float64) / voxel_size_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    group_start = np.empty(len(order), dtype=bool)
    group_start[0] = True
    group_start[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    starts = np.flatnonzero(group_start)
    counts = np.diff(np.append(starts, len(order))).astype(np.float64)

    xyz_sorted = cloud.xyz[order].astype(np.float64)
    xyz = np.add.reduceat(xyz_sorted, starts, axis=0) / counts[:, None]
    rgb_sorted = cloud.rgb[order].astype(np.float64)
    rgb = np.rint(np.add.reduceat(rgb_sorted, starts, axis=0) / counts[:, None])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    normals: np.ndarray | None = None
    if cloud.normals is not None:
        normal_sorted = cloud.normals[order].astype(np.float64)
        normals = np.add.reduceat(normal_sorted, starts, axis=0)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        normals[~valid] = 0
        normals = normals.astype(np.float32)

    return PointCloud(xyz=xyz.astype(np.float32), rgb=rgb, normals=normals)


def point_cloud_stats(cloud: PointCloud) -> dict[str, Any]:
    result: dict[str, Any] = {
        "point_count": int(len(cloud.xyz)),
        "bounds_min_m": cloud.xyz.min(axis=0).astype(float).tolist(),
        "bounds_max_m": cloud.xyz.max(axis=0).astype(float).tolist(),
        "rgb_min": cloud.rgb.min(axis=0).astype(int).tolist(),
        "rgb_max": cloud.rgb.max(axis=0).astype(int).tolist(),
        "finite_xyz": bool(np.all(np.isfinite(cloud.xyz))),
    }
    if cloud.normals is not None:
        lengths = np.linalg.norm(cloud.normals, axis=1)
        result["normal_length_min"] = float(lengths.min())
        result["normal_length_max"] = float(lengths.max())
        result["zero_normal_count"] = int(np.count_nonzero(lengths <= 1e-12))
    return result


def write_binary_ply(path: Path, cloud: PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_normals = cloud.normals is not None
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by e57-to-3dgs; coordinates are registered meters",
        f"element vertex {len(cloud.xyz)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header_lines.extend(["property float nx", "property float ny", "property float nz"])
    header_lines.extend(
        ["property uchar red", "property uchar green", "property uchar blue", "end_header"]
    )
    dtype_fields: list[tuple[str, Any]] = [(axis, "<f4") for axis in "xyz"]
    if has_normals:
        dtype_fields.extend((f"n{axis}", "<f4") for axis in "xyz")
    dtype_fields.extend((color, "u1") for color in ("red", "green", "blue"))
    vertices = np.empty(len(cloud.xyz), dtype=np.dtype(dtype_fields))
    for axis_index, axis in enumerate("xyz"):
        vertices[axis] = cloud.xyz[:, axis_index]
    if has_normals and cloud.normals is not None:
        for axis_index, axis in enumerate("xyz"):
            vertices[f"n{axis}"] = cloud.normals[:, axis_index]
    for axis_index, color in enumerate(("red", "green", "blue")):
        vertices[color] = cloud.rgb[:, axis_index]
    temporary_path = path.with_suffix(path.suffix + ".partial")
    with temporary_path.open("wb") as output:
        output.write(("\n".join(header_lines) + "\n").encode("ascii"))
        vertices.tofile(output)
    temporary_path.replace(path)
