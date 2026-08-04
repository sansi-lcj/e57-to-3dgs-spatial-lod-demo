from pathlib import Path

import numpy as np

from e57gs.gaussian_ply import filter_gaussian_ply, read_gaussian_ply


DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
    ]
)


def write_fixture(path: Path) -> None:
    vertices = np.zeros(4, dtype=DTYPE)
    vertices["opacity"] = [0.0, -5.0, 3.0, 3.0]
    vertices["scale_0"] = np.log([0.05, 0.05, 0.20, 0.05])
    vertices["scale_1"] = np.log(0.05)
    vertices["scale_2"] = np.log(0.05)
    vertices["x"] = [0.0, 0.0, 0.0, 2.0]
    header = ["ply", "format binary_little_endian 1.0", "element vertex 4"]
    header.extend(f"property float {name}" for name in DTYPE.names or ())
    header.append("end_header")
    path.write_bytes(("\n".join(header) + "\n").encode() + vertices.tobytes())


def test_filter_gaussian_ply_applies_all_three_guards(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    output = tmp_path / "filtered.ply"
    write_fixture(source)

    result = filter_gaussian_ply(
        source,
        output,
        min_opacity=0.03,
        max_scale_m=0.10,
        bounds_min_m=(-1.0, -1.0, -1.0),
        bounds_max_m=(1.0, 1.0, 1.0),
    )

    parsed = read_gaussian_ply(output)
    assert parsed.vertex_count == 1
    assert result["removed_low_opacity"] == 1
    assert result["removed_over_scale"] == 1
    assert result["removed_out_of_bounds"] == 1


def test_needle_filter_keeps_two_axis_plane(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    output = tmp_path / "filtered.ply"
    vertices = np.zeros(2, dtype=DTYPE)
    vertices["opacity"] = 3.0
    # First vertex is a long one-axis needle; second is a broad, thin plane.
    vertices["scale_0"] = np.log([0.20, 0.20])
    vertices["scale_1"] = np.log([0.01, 0.10])
    vertices["scale_2"] = np.log([0.01, 0.001])
    header = ["ply", "format binary_little_endian 1.0", "element vertex 2"]
    header.extend(f"property float {name}" for name in DTYPE.names or ())
    header.append("end_header")
    source.write_bytes(("\n".join(header) + "\n").encode() + vertices.tobytes())

    result = filter_gaussian_ply(
        source,
        output,
        min_opacity=0.0,
        max_scale_m=0.10,
        bounds_min_m=(-1.0, -1.0, -1.0),
        bounds_max_m=(1.0, 1.0, 1.0),
        scale_mode="needle",
        second_scale_m=0.05,
        min_aspect_ratio=4.0,
    )

    assert read_gaussian_ply(output).vertex_count == 1
    assert result["removed_over_scale"] == 1
