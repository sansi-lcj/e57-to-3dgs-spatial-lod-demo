import os
from pathlib import Path

import numpy as np
import pytest

from e57gs.e57io import read_metadata
from e57gs.points import registered_scan, voxel_downsample


SOURCE = Path(os.environ.get("E57GS_TEST_E57", "tests/fixtures/source.e57"))


@pytest.mark.skipif(not SOURCE.exists(), reason="Local E57 fixture is unavailable")
def test_source_metadata_and_associations():
    metadata = read_metadata(SOURCE, include_sha256=False)
    assert metadata.file_header.physical_length == 151_980_032
    assert [scan.point_count for scan in metadata.scans] == [488_212, 615_866]
    assert len(metadata.images) == 2
    assert all((image.width, image.height) == (24_000, 12_000) for image in metadata.images)
    assert {image.associated_data3d_guid for image in metadata.images} == {
        scan.guid for scan in metadata.scans
    }


@pytest.mark.skipif(not SOURCE.exists(), reason="Local E57 fixture is unavailable")
def test_normal_extension_and_registration():
    import pye57

    with pye57.E57(str(SOURCE)) as e57:
        cloud = registered_scan(e57, 0)
    assert len(cloud.xyz) == 488_212
    assert cloud.normals is not None
    assert np.all(np.isfinite(cloud.xyz))
    normal_lengths = np.linalg.norm(cloud.normals, axis=1)
    assert np.quantile(normal_lengths, 0.01) == pytest.approx(1.0, abs=1e-5)
    reduced = voxel_downsample(cloud, 0.04)
    assert 0 < len(reduced.xyz) < len(cloud.xyz)
