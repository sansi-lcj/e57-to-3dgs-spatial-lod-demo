"""Coordinate and camera geometry for Realsee E57 panoramas."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VirtualView:
    """One perspective view rendered from a panorama center.

    ``pano_from_camera`` maps a column vector from the perspective camera
    coordinate system (X right, Y down, Z forward) into the Realsee panorama /
    scan-local system with the same axis semantics.
    """

    index: int
    yaw_deg: float
    pitch_deg: float
    camera_from_pano: FloatArray

    @property
    def pano_from_camera(self) -> FloatArray:
        return self.camera_from_pano.T


def quaternion_wxyz_to_matrix(quaternion: npt.ArrayLike) -> FloatArray:
    """Convert a Hamilton wxyz quaternion to an active rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Quaternion must have a finite non-zero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: npt.ArrayLike) -> FloatArray:
    """Convert a proper rotation matrix to a canonical Hamilton quaternion."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {rotation.shape}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("Matrix is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("Matrix is not a proper rotation")

    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def rotation_x(angle_rad: float) -> FloatArray:
    cosine, sine = cos(angle_rad), sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def rotation_y(angle_rad: float) -> FloatArray:
    cosine, sine = cos(angle_rad), sin(angle_rad)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def overlapping_virtual_views(
    num_steps_yaw: int = 4,
    pitches_deg: tuple[float, ...] = (-35.0, 0.0, 35.0),
) -> tuple[VirtualView, ...]:
    """Return COLMAP's official overlapping panorama camera layout.

    Positive pitch is upward. The positive-pitch row is staggered by half a
    yaw step, matching ``pycolmap/panorama.py``.
    """

    if num_steps_yaw <= 0:
        raise ValueError("num_steps_yaw must be positive")
    yaw_step = 360.0 / num_steps_yaw
    views: list[VirtualView] = []
    for pitch_deg in pitches_deg:
        yaw_offset = yaw_step / 2.0 if pitch_deg > 0 else 0.0
        for yaw_index in range(num_steps_yaw):
            yaw_deg = yaw_index * yaw_step + yaw_offset
            camera_from_pano = rotation_x(radians(-pitch_deg)) @ rotation_y(radians(-yaw_deg))
            views.append(
                VirtualView(
                    index=len(views),
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    camera_from_pano=camera_from_pano,
                )
            )
    return tuple(views)


def perspective_intrinsics(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float | None = None,
) -> FloatArray:
    """Return a 3x3 pinhole K using COLMAP's half-pixel image coordinates."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if vertical_fov_deg is None:
        vertical_fov_deg = horizontal_fov_deg
    if not 0 < horizontal_fov_deg < 180 or not 0 < vertical_fov_deg < 180:
        raise ValueError("Perspective field of view must be between 0 and 180 degrees")
    fx = width / (2.0 * tan(radians(horizontal_fov_deg) / 2.0))
    fy = height / (2.0 * tan(radians(vertical_fov_deg) / 2.0))
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def colmap_world_to_camera(
    scan_to_world: npt.ArrayLike,
    camera_center_world: npt.ArrayLike,
    camera_from_pano: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Compose Realsee scan pose and virtual crop pose for COLMAP.

    Realsee panorama coordinates are identical to scan-local coordinates. The
    returned transform follows COLMAP's ``X_cam = R_cw X_world + t_cw``.
    """

    rotation_ws = np.asarray(scan_to_world, dtype=np.float64)
    center_world = np.asarray(camera_center_world, dtype=np.float64)
    rotation_cp = np.asarray(camera_from_pano, dtype=np.float64)
    if rotation_ws.shape != (3, 3) or rotation_cp.shape != (3, 3):
        raise ValueError("Rotations must be 3x3 matrices")
    if center_world.shape != (3,):
        raise ValueError("Camera center must have shape (3,)")
    rotation_cw = rotation_cp @ rotation_ws.T
    translation_cw = -(rotation_cw @ center_world)
    return rotation_cw, translation_cw


def recover_camera_center(rotation_cw: npt.ArrayLike, translation_cw: npt.ArrayLike) -> FloatArray:
    rotation = np.asarray(rotation_cw, dtype=np.float64)
    translation = np.asarray(translation_cw, dtype=np.float64)
    return -(rotation.T @ translation)
