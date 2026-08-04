import numpy as np

from e57gs.geometry import overlapping_virtual_views, perspective_intrinsics
from e57gs.panorama import panorama_maps, perspective_rays


def test_identity_crop_center_maps_to_panorama_center():
    views = overlapping_virtual_views()
    identity = views[4]
    intrinsics = perspective_intrinsics(3, 3, 90.0)
    rays = perspective_rays(3, 3, intrinsics)
    map_x, map_y, rays_pano = panorama_maps(24000, 12000, rays, identity)
    assert map_x[1, 1] == pytest.approx(11999.5, abs=1e-3)
    assert map_y[1, 1] == pytest.approx(5999.5, abs=1e-3)
    assert np.allclose(rays_pano[1, 1], [0, 0, 1])


def test_yaw_90_center_points_to_positive_x():
    view = overlapping_virtual_views()[5]
    intrinsics = perspective_intrinsics(3, 3, 90.0)
    rays = perspective_rays(3, 3, intrinsics)
    map_x, map_y, rays_pano = panorama_maps(24000, 12000, rays, view)
    assert map_x[1, 1] == pytest.approx(17999.5, abs=1e-3)
    assert map_y[1, 1] == pytest.approx(5999.5, abs=1e-3)
    assert np.allclose(rays_pano[1, 1], [1, 0, 0], atol=1e-6)


import pytest
