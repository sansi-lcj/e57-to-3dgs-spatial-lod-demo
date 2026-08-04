import numpy as np
import pytest

from e57gs.lod_tiles import nested_lod_indices


def test_nested_lod_indices_are_disjoint_and_complete() -> None:
    axis = np.linspace(0.0, 0.12, 13)
    xyz = np.asarray([(x, y, z) for x in axis for y in axis[:4] for z in axis[:2]])
    base, mid, fine = nested_lod_indices(
        xyz,
        origin=np.zeros(3),
        base_voxel_m=0.06,
        mid_voxel_m=0.02,
    )
    combined = np.concatenate([base, mid, fine])
    assert len(np.unique(combined)) == len(xyz)
    assert sorted(combined.tolist()) == list(range(len(xyz)))
    assert len(base) < len(base) + len(mid) < len(xyz)


def test_nested_lod_indices_validate_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        nested_lod_indices(
            np.zeros((2, 2)),
            origin=np.zeros(3),
            base_voxel_m=0.06,
            mid_voxel_m=0.02,
        )
    with pytest.raises(ValueError, match="mid < base"):
        nested_lod_indices(
            np.zeros((2, 3)),
            origin=np.zeros(3),
            base_voxel_m=0.02,
            mid_voxel_m=0.06,
        )
