import numpy as np

from e57gs.geometry import (
    colmap_world_to_camera,
    matrix_to_quaternion_wxyz,
    overlapping_virtual_views,
    perspective_intrinsics,
    quaternion_wxyz_to_matrix,
    recover_camera_center,
)


def test_quaternion_matrix_round_trip():
    quaternion = np.array([0.5, 0.5, -0.5, 0.5])
    rotation = quaternion_wxyz_to_matrix(quaternion)
    recovered = matrix_to_quaternion_wxyz(rotation)
    assert abs(float(np.dot(quaternion, recovered))) == pytest.approx(1.0)


def test_official_overlapping_layout():
    views = overlapping_virtual_views()
    assert len(views) == 12
    assert [view.yaw_deg for view in views[:4]] == [0.0, 90.0, 180.0, 270.0]
    assert [view.yaw_deg for view in views[4:8]] == [0.0, 90.0, 180.0, 270.0]
    assert [view.yaw_deg for view in views[8:]] == [45.0, 135.0, 225.0, 315.0]
    for view in views:
        assert np.allclose(view.camera_from_pano.T @ view.camera_from_pano, np.eye(3))
        assert np.linalg.det(view.camera_from_pano) == pytest.approx(1.0)


def test_identity_view_and_camera_center():
    identity_view = overlapping_virtual_views()[4]
    assert identity_view.pitch_deg == 0.0
    assert identity_view.yaw_deg == 0.0
    assert np.allclose(identity_view.camera_from_pano, np.eye(3))

    scan_to_world = quaternion_wxyz_to_matrix([0.9, 0.1, -0.2, 0.3])
    center = np.array([1.2, -3.4, 5.6])
    rotation_cw, translation_cw = colmap_world_to_camera(
        scan_to_world, center, identity_view.camera_from_pano
    )
    assert np.allclose(recover_camera_center(rotation_cw, translation_cw), center)
    assert np.allclose(rotation_cw, scan_to_world.T)


def test_90_degree_square_intrinsics():
    intrinsics = perspective_intrinsics(2048, 2048, 90.0)
    assert np.allclose(
        intrinsics,
        [[1024.0, 0.0, 1024.0], [0.0, 1024.0, 1024.0], [0.0, 0.0, 1.0]],
    )


import pytest
