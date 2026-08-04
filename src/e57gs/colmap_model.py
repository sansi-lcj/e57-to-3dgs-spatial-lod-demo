"""Write a legacy COLMAP model accepted by OpenSplat and gsplat."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

from .points import PointCloud


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    image_name: str
    quaternion_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    camera_id: int = 1
    panorama_guid: str = ""
    scan_guid: str = ""
    view_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_cameras_text(path: Path, width: int, height: int, focal: float) -> None:
    content = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 SIMPLE_PINHOLE {width} {height} {focal:.17g} {width / 2:.17g} {height / 2:.17g}\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_images_text(path: Path, images: Iterable[ColmapImage]) -> None:
    image_list = list(images)
    with path.open("w", encoding="utf-8") as output:
        output.write("# Image list with two lines of data per image:\n")
        output.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        output.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        output.write(f"# Number of images: {len(image_list)}, mean observations per image: 0\n")
        for image in image_list:
            values = [
                image.image_id,
                *image.quaternion_wxyz,
                *image.translation,
                image.camera_id,
                image.image_name,
            ]
            output.write(" ".join(str(value) for value in values) + "\n\n")


def _write_points_text(path: Path, cloud: PointCloud) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("# 3D point list with one line of data per point:\n")
        output.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        output.write(f"# Number of points: {len(cloud.xyz)}, mean track length: 0\n")
        for index, (xyz, rgb) in enumerate(zip(cloud.xyz, cloud.rgb, strict=True), start=1):
            output.write(
                f"{index} {float(xyz[0]):.9g} {float(xyz[1]):.9g} {float(xyz[2]):.9g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n"
            )


def _write_cameras_binary(path: Path, width: int, height: int, focal: float) -> None:
    with path.open("wb") as output:
        output.write(struct.pack("<Q", 1))
        output.write(struct.pack("<iiQQ", 1, 0, width, height))  # SIMPLE_PINHOLE model id 0
        output.write(struct.pack("<ddd", focal, width / 2.0, height / 2.0))


def _write_images_binary(path: Path, images: Iterable[ColmapImage]) -> None:
    image_list = list(images)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(image_list)))
        for image in image_list:
            output.write(struct.pack("<i", image.image_id))
            output.write(struct.pack("<dddd", *image.quaternion_wxyz))
            output.write(struct.pack("<ddd", *image.translation))
            output.write(struct.pack("<i", image.camera_id))
            output.write(image.image_name.encode("utf-8") + b"\x00")
            output.write(struct.pack("<Q", 0))


def _write_points_binary(path: Path, cloud: PointCloud) -> None:
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(cloud.xyz)))
        for index, (xyz, rgb) in enumerate(zip(cloud.xyz, cloud.rgb, strict=True), start=1):
            output.write(
                struct.pack(
                    "<QdddBBBdQ",
                    index,
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                    int(rgb[0]),
                    int(rgb[1]),
                    int(rgb[2]),
                    0.0,
                    0,
                )
            )


def write_colmap_model(
    binary_dir: Path,
    text_dir: Path,
    *,
    width: int,
    height: int,
    focal: float,
    images: list[ColmapImage],
    cloud: PointCloud,
) -> None:
    if any(image.camera_id != 1 for image in images):
        raise ValueError("The flat training model must use one shared camera")
    if len({image.image_id for image in images}) != len(images):
        raise ValueError("COLMAP image IDs must be unique")
    if len({image.image_name for image in images}) != len(images):
        raise ValueError("COLMAP image names must be unique")
    binary_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras_text(text_dir / "cameras.txt", width, height, focal)
    _write_images_text(text_dir / "images.txt", images)
    _write_points_text(text_dir / "points3D.txt", cloud)
    _write_cameras_binary(binary_dir / "cameras.bin", width, height, focal)
    _write_images_binary(binary_dir / "images.bin", images)
    _write_points_binary(binary_dir / "points3D.bin", cloud)


def validate_with_pycolmap(model_dir: Path) -> dict[str, Any]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    image_centers = {
        image.name: np.asarray(image.projection_center(), dtype=float).tolist()
        for image in reconstruction.images.values()
    }
    return {
        "num_cameras": int(reconstruction.num_cameras()),
        "num_images": int(reconstruction.num_images()),
        "num_registered_images": int(reconstruction.num_reg_images()),
        "num_points3d": int(reconstruction.num_points3D()),
        "is_valid": bool(reconstruction.is_valid()),
        "summary": reconstruction.summary(),
        "image_centers": image_centers,
    }
