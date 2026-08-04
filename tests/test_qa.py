import numpy as np

from e57gs.qa import _nearest_panorama_x


def test_rounded_panorama_longitude_wraps_to_first_column():
    width = 24000
    source_x = np.mod(np.array([width - 0.49]), width)

    pixel_x = _nearest_panorama_x(source_x, width)

    assert pixel_x.tolist() == [0]
