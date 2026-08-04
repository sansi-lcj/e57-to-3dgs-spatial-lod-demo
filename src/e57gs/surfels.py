"""Convert a colored, normal-bearing point cloud to geometry-faithful Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class PointPly:
    data_offset: int
    vertex_count: int
    dtype: np.dtype


def read_point_ply(path: Path) -> PointPly:
    """Read the scalar vertex schema of a binary little-endian PLY."""

    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    in_vertices = False
    with path.open("rb") as source:
        if source.readline() != b"ply\n":
            raise ValueError("Not a PLY file")
        if source.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError("Expected binary_little_endian PLY")
        while True:
            line = source.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("element "):
                _, name, count = text.split()
                in_vertices = name == "vertex"
                if in_vertices:
                    vertex_count = int(count)
            elif text.startswith("property ") and in_vertices:
                parts = text.split()
                if len(parts) != 3 or parts[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property: {text}")
                properties.append((parts[2], PLY_TYPES[parts[1]]))
            elif text == "end_header":
                break
        data_offset = source.tell()

    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    dtype = np.dtype(properties)
    expected_size = data_offset + vertex_count * dtype.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"PLY size mismatch: {path.stat().st_size} != {expected_size}")
    required = {"x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"}
    missing = sorted(required.difference(dtype.names or ()))
    if missing:
        raise ValueError(f"Missing point-cloud properties: {missing}")
    return PointPly(data_offset=data_offset, vertex_count=vertex_count, dtype=dtype)


def quaternion_z_to_normal(normals: np.ndarray) -> np.ndarray:
    """Return normalized wxyz quaternions rotating +Z onto each normal."""

    values = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-12)
    unit = np.zeros_like(values)
    unit[valid] = values[valid] / lengths[valid, None]
    unit[~valid, 2] = 1.0

    quaternions = np.zeros((len(unit), 4), dtype=np.float64)
    quaternions[:, 0] = 1.0 + unit[:, 2]
    quaternions[:, 1] = -unit[:, 1]
    quaternions[:, 2] = unit[:, 0]
    opposite = quaternions[:, 0] < 1e-10
    quaternions[opposite] = (0.0, 1.0, 0.0, 0.0)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions.astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_colmap_points_from_ply(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Write fixed-size, trackless COLMAP points3D.bin records from a point PLY."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    model = read_point_ply(input_path)
    points = np.memmap(
        input_path,
        mode="r",
        dtype=model.dtype,
        offset=model.data_offset,
        shape=model.vertex_count,
    )
    record_dtype = np.dtype(
        [
            ("point_id", "<u8"),
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("error", "<f8"),
            ("track_length", "<u8"),
        ]
    )
    if record_dtype.itemsize != 51:
        raise AssertionError(f"Unexpected COLMAP point record size: {record_dtype.itemsize}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("xb") as output:
            output.write(struct.pack("<Q", model.vertex_count))
            for start in range(0, model.vertex_count, 250_000):
                stop = min(start + 250_000, model.vertex_count)
                source_chunk = points[start:stop]
                records = np.zeros(stop - start, dtype=record_dtype)
                records["point_id"] = np.arange(start + 1, stop + 1, dtype=np.uint64)
                for axis in "xyz":
                    records[axis] = source_chunk[axis]
                for color in ("red", "green", "blue"):
                    records[color] = source_chunk[color]
                output.write(records.tobytes())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    expected_size = 8 + 51 * model.vertex_count
    if output_path.stat().st_size != expected_size:
        raise ValueError(f"COLMAP points size mismatch: {output_path.stat().st_size} != {expected_size}")
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "point_count": model.vertex_count,
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }


def point_cloud_to_gaussian_surfels(
    input_path: Path,
    output_path: Path,
    *,
    density_voxel_m: float = 0.02,
    tangent_factor: float = 0.55,
    min_tangent_scale_m: float = 0.002,
    max_tangent_scale_m: float = 0.012,
    normal_scale_ratio: float = 0.12,
    opacity: float = 0.85,
    sh_degree: int = 0,
    opensplat_iteration: int | None = None,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Create SH0 Gaussian surfels while preserving every source point."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    if density_voxel_m <= 0 or tangent_factor <= 0:
        raise ValueError("density voxel and tangent factor must be positive")
    if not 0 < min_tangent_scale_m <= max_tangent_scale_m:
        raise ValueError("tangent scale bounds must be positive and ordered")
    if not 0 < normal_scale_ratio <= 1:
        raise ValueError("normal scale ratio must be in (0, 1]")
    if not 0 < opacity < 1:
        raise ValueError("opacity must be in (0, 1)")
    rest_counts = {0: 0, 1: 9, 2: 24, 3: 45}
    if sh_degree not in rest_counts:
        raise ValueError("sh_degree must be from 0 to 3")
    if opensplat_iteration is not None and opensplat_iteration < 0:
        raise ValueError("opensplat_iteration must be non-negative")

    model = read_point_ply(input_path)
    points = np.memmap(
        input_path,
        mode="r",
        dtype=model.dtype,
        offset=model.data_offset,
        shape=model.vertex_count,
    )
    xyz = np.column_stack([points[axis] for axis in "xyz"]).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Point cloud contains non-finite coordinates")
    bounds_min = xyz.min(axis=0)
    bounds_max = xyz.max(axis=0)

    voxel_indices = np.floor((xyz - bounds_min) / density_voxel_m).astype(np.int64)
    voxel_dimensions = voxel_indices.max(axis=0) + 1
    voxel_codes = (
        voxel_indices[:, 0]
        + voxel_dimensions[0]
        * (voxel_indices[:, 1] + voxel_dimensions[1] * voxel_indices[:, 2])
    )
    _, inverse, counts = np.unique(voxel_codes, return_inverse=True, return_counts=True)
    tangent_scales = density_voxel_m / np.sqrt(counts[inverse].astype(np.float64))
    tangent_scales = np.clip(
        tangent_scales * tangent_factor,
        min_tangent_scale_m,
        max_tangent_scale_m,
    ).astype(np.float32)
    del voxel_indices, voxel_codes, inverse, counts

    output_dtype = np.dtype(
        [(axis, "<f4") for axis in "xyz"]
        + [(f"n{axis}", "<f4") for axis in "xyz"]
        + [(f"f_dc_{index}", "<f4") for index in range(3)]
        + [(f"f_rest_{index}", "<f4") for index in range(rest_counts[sh_degree])]
        + [("opacity", "<f4")]
        + [(f"scale_{index}", "<f4") for index in range(3)]
        + [(f"rot_{index}", "<f4") for index in range(4)]
    )
    comment = (
        f"Generated by opensplat at iteration {opensplat_iteration}"
        if opensplat_iteration is not None
        else "Geometry-faithful Gaussian surfels from source XYZ RGB normals"
    )
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment {comment}",
        f"element vertex {model.vertex_count}",
    ]
    header.extend(f"property float {name}" for name in output_dtype.names or ())
    header.append("end_header")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    opacity_logit = np.float32(np.log(opacity / (1.0 - opacity)))
    try:
        with temporary_path.open("xb") as output:
            output.write(("\n".join(header) + "\n").encode("ascii"))
            for start in range(0, model.vertex_count, chunk_size):
                stop = min(start + chunk_size, model.vertex_count)
                source_chunk = points[start:stop]
                chunk = np.empty(stop - start, dtype=output_dtype)
                for axis in "xyz":
                    chunk[axis] = source_chunk[axis]
                    chunk[f"n{axis}"] = source_chunk[f"n{axis}"]
                for index, color in enumerate(("red", "green", "blue")):
                    rgb = source_chunk[color].astype(np.float32) / 255.0
                    chunk[f"f_dc_{index}"] = (rgb - 0.5) / SH_C0
                for index in range(rest_counts[sh_degree]):
                    chunk[f"f_rest_{index}"] = 0
                chunk["opacity"] = opacity_logit
                local_tangent = tangent_scales[start:stop]
                chunk["scale_0"] = np.log(local_tangent)
                chunk["scale_1"] = np.log(local_tangent)
                chunk["scale_2"] = np.log(local_tangent * normal_scale_ratio)
                normals = np.column_stack(
                    [source_chunk[f"n{axis}"] for axis in "xyz"]
                )
                rotations = quaternion_z_to_normal(normals)
                for index in range(4):
                    chunk[f"rot_{index}"] = rotations[:, index]
                output.write(chunk.tobytes())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "vertex_count": model.vertex_count,
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "bounds_min_m": bounds_min.astype(float).tolist(),
        "bounds_max_m": bounds_max.astype(float).tolist(),
        "density_voxel_m": density_voxel_m,
        "tangent_factor": tangent_factor,
        "tangent_scale_m": {
            "min": float(tangent_scales.min()),
            "median": float(np.median(tangent_scales)),
            "max": float(tangent_scales.max()),
        },
        "normal_scale_ratio": normal_scale_ratio,
        "opacity": opacity,
        "sh_degree": sh_degree,
        "opensplat_iteration": opensplat_iteration,
    }
