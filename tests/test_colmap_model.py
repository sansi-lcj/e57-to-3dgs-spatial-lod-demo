from pathlib import Path

import numpy as np

from e57gs.colmap_model import ColmapImage, validate_with_pycolmap, write_colmap_model
from e57gs.points import PointCloud


def test_binary_colmap_model_round_trip(tmp_path: Path):
    cloud = PointCloud(
        xyz=np.array([[0, 0, 1], [1, 2, 3]], dtype=np.float32),
        rgb=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
    )
    image = ColmapImage(
        image_id=1,
        image_name="pano_camera00/example.jpg",
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 0.0),
    )
    binary_dir = tmp_path / "sparse" / "0"
    text_dir = tmp_path / "sparse_txt" / "0"
    write_colmap_model(
        binary_dir,
        text_dir,
        width=2048,
        height=2048,
        focal=1024.0,
        images=[image],
        cloud=cloud,
    )
    result = validate_with_pycolmap(binary_dir)
    assert result["num_cameras"] == 1
    assert result["num_registered_images"] == 1
    assert result["num_points3d"] == 2
    assert np.allclose(result["image_centers"][image.image_name], [0, 0, 0])
