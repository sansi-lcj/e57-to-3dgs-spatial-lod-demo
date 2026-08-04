#!/usr/bin/env python3
"""Validate and summarize an OpenSplat binary Gaussian PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[int, int, list[tuple[str, str]], str]:
    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    comment = ""
    with path.open("rb") as source:
        if source.readline() != b"ply\n":
            raise ValueError("Not a PLY file")
        if source.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError("Expected binary_little_endian PLY")
        in_vertices = False
        while True:
            line = source.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("comment "):
                comment = text.removeprefix("comment ")
            elif text.startswith("element "):
                _, name, count = text.split()
                in_vertices = name == "vertex"
                if in_vertices:
                    vertex_count = int(count)
            elif text.startswith("property ") and in_vertices:
                _, scalar_type, name = text.split()
                if scalar_type not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {scalar_type}")
                properties.append((name, PLY_TYPES[scalar_type]))
            elif text == "end_header":
                break
        offset = source.tell()
    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    return offset, vertex_count, properties, comment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    args = parser.parse_args()
    path = args.ply.resolve()

    offset, vertex_count, properties, comment = read_header(path)
    dtype = np.dtype(properties)
    expected_size = offset + vertex_count * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"PLY size mismatch: {actual_size} != {expected_size}")

    vertices = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=vertex_count)
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required.difference(vertices.dtype.names or ()))
    if missing:
        raise ValueError(f"Missing Gaussian PLY properties: {missing}")

    float_names = [name for name in vertices.dtype.names or () if vertices[name].dtype.kind == "f"]
    finite = all(bool(np.isfinite(vertices[name]).all()) for name in float_names)
    if not finite:
        raise ValueError("Gaussian PLY contains non-finite values")

    xyz = np.column_stack([vertices[axis] for axis in "xyz"])
    log_scales = np.column_stack([vertices[f"scale_{index}"] for index in range(3)])
    rotations = np.column_stack([vertices[f"rot_{index}"] for index in range(4)])
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
    scales = np.exp(log_scales.astype(np.float64))
    rotation_norms = np.linalg.norm(rotations, axis=1)
    f_rest_count = len([name for name in vertices.dtype.names or () if name.startswith("f_rest_")])
    sh_degree = {0: 0, 9: 1, 24: 2, 45: 3}.get(f_rest_count)

    result = {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": actual_size,
        "comment": comment,
        "vertex_count": vertex_count,
        "record_size_bytes": dtype.itemsize,
        "property_count": len(properties),
        "sh_degree": sh_degree,
        "all_float_properties_finite": finite,
        "bounds_min_m": xyz.min(axis=0).astype(float).tolist(),
        "bounds_max_m": xyz.max(axis=0).astype(float).tolist(),
        "opacity_probability": {
            "min": float(opacity.min()),
            "median": float(np.median(opacity)),
            "max": float(opacity.max()),
        },
        "physical_scale_m": {
            "min": float(scales.min()),
            "median": float(np.median(scales)),
            "max": float(scales.max()),
        },
        "quaternion_norm": {
            "min": float(rotation_norms.min()),
            "max": float(rotation_norms.max()),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
