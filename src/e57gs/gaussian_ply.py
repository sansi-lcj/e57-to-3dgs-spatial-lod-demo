"""Read and conservatively filter OpenSplat binary Gaussian PLY files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

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


@dataclass(frozen=True)
class GaussianPly:
    header: bytes
    data_offset: int
    vertex_count: int
    dtype: np.dtype


def read_gaussian_ply(path: Path) -> GaussianPly:
    """Parse the scalar vertex schema used by OpenSplat."""
    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    header_lines: list[bytes] = []
    in_vertices = False

    with path.open("rb") as source:
        while True:
            line = source.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            header_lines.append(line)
            text = line.decode("ascii").strip()
            if len(header_lines) == 1 and text != "ply":
                raise ValueError("Not a PLY file")
            if len(header_lines) == 2 and text != "format binary_little_endian 1.0":
                raise ValueError("Expected binary_little_endian PLY")
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
        raise ValueError(
            f"PLY size mismatch: {path.stat().st_size} != {expected_size}"
        )

    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
    }
    missing = sorted(required.difference(dtype.names or ()))
    if missing:
        raise ValueError(f"Missing Gaussian PLY properties: {missing}")
    return GaussianPly(b"".join(header_lines), data_offset, vertex_count, dtype)


def filter_gaussian_ply(
    input_path: Path,
    output_path: Path,
    *,
    min_opacity: float,
    max_scale_m: float,
    bounds_min_m: tuple[float, float, float],
    bounds_max_m: tuple[float, float, float],
    scale_mode: str = "max",
    second_scale_m: float = 0.05,
    min_aspect_ratio: float = 4.0,
) -> dict[str, object]:
    """Filter low-opacity, over-scale and out-of-bounds Gaussian vertices."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    if not 0.0 <= min_opacity < 1.0:
        raise ValueError("min_opacity must be in [0, 1)")
    if max_scale_m <= 0:
        raise ValueError("max_scale_m must be positive")
    if scale_mode not in {"max", "needle"}:
        raise ValueError("scale_mode must be 'max' or 'needle'")
    if second_scale_m <= 0 or min_aspect_ratio <= 1:
        raise ValueError("needle scale parameters must be positive with ratio > 1")

    model = read_gaussian_ply(input_path)
    vertices = np.memmap(
        input_path,
        mode="r",
        dtype=model.dtype,
        offset=model.data_offset,
        shape=model.vertex_count,
    )
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
    log_scales = np.column_stack(
        [vertices[f"scale_{axis}"] for axis in range(3)]
    ).astype(np.float64)
    ordered_log_scales = np.sort(log_scales, axis=1)
    second_log_scale = ordered_log_scales[:, 1]
    max_log_scale = ordered_log_scales[:, 2]
    bounds_min = np.asarray(bounds_min_m, dtype=np.float64)
    bounds_max = np.asarray(bounds_max_m, dtype=np.float64)
    xyz = np.column_stack([vertices[axis] for axis in "xyz"])

    opacity_ok = opacity >= min_opacity
    if scale_mode == "max":
        scale_ok = max_log_scale <= np.log(max_scale_m)
    else:
        needle = (
            (max_log_scale > np.log(max_scale_m))
            & (second_log_scale < np.log(second_scale_m))
            & ((max_log_scale - second_log_scale) > np.log(min_aspect_ratio))
        )
        scale_ok = ~needle
    bounds_ok = np.logical_and(xyz >= bounds_min, xyz <= bounds_max).all(axis=1)
    keep = opacity_ok & scale_ok & bounds_ok
    kept_count = int(keep.sum())

    output_header = re.sub(
        rb"(?m)^element vertex \d+\r?$",
        f"element vertex {kept_count}".encode("ascii"),
        model.header,
        count=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("xb") as destination:
            destination.write(output_header)
            for start in range(0, model.vertex_count, 100_000):
                stop = min(start + 100_000, model.vertex_count)
                destination.write(vertices[start:stop][keep[start:stop]].tobytes())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    digest = hashlib.sha256()
    with output_path.open("rb") as filtered:
        for chunk in iter(lambda: filtered.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "input_vertices": model.vertex_count,
        "output_vertices": kept_count,
        "removed_vertices": model.vertex_count - kept_count,
        "removed_low_opacity": int((~opacity_ok).sum()),
        "removed_over_scale": int((~scale_ok).sum()),
        "removed_out_of_bounds": int((~bounds_ok).sum()),
        "min_opacity": min_opacity,
        "max_scale_m": max_scale_m,
        "scale_mode": scale_mode,
        "second_scale_m": second_scale_m,
        "min_aspect_ratio": min_aspect_ratio,
        "bounds_min_m": bounds_min.astype(float).tolist(),
        "bounds_max_m": bounds_max.astype(float).tolist(),
        "sha256": digest.hexdigest(),
    }
