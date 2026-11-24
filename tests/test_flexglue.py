from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio

from bundle_adjust.feature_tracks.ft_flexglue import PairMatches, index_pair_matches, load_rgb, native_pixels
from bundle_adjust.feature_tracks.ft_lightglue import affine_image_rotation


def test_flexglue_coordinates_preserve_native_pixels_for_all_quarter_turns():
    xy = np.array([[0.25, 3.1], [13.0, 5.0], [7.25, 8.5]])
    for angle in (0, 90, 180, 270):
        affine, _ = affine_image_rotation(23, 17, angle)
        corner_coordinates = xy @ affine[:, :2].T + affine[:, 2] + 0.5
        np.testing.assert_allclose(native_pixels(corner_coordinates, 23, 17, angle), xy, atol=1e-10)


def test_coordinate_pair_adapter_shares_nearby_observations(tmp_path):
    pairs = [
        PairMatches(0, 1, np.array([[1, 2], [10, 11]]), np.array([[3, 4], [12, 13]]), np.array([0.8, 0.9])),
        PairMatches(0, 2, np.array([[1.2, 2.1]]), np.array([[5, 6]]), np.array([0.95])),
    ]
    paths = [tmp_path / "features" / f"image{i}.npy" for i in range(3)]
    indexed = index_pair_matches(pairs, paths, tmp_path)
    assert indexed.shape == (3, 4)
    assert indexed[0, 2] == indexed[2, 2] == 0
    first_id = np.argmin(np.linalg.norm(np.load(paths[0])[:, :2] - [1.2, 2.1], axis=1))
    assert indexed[2, 0] == first_id
    assert first_id in indexed[:2, 0]
    np.testing.assert_array_equal(pairs[0].xy0[0], [1, 2])
    assert all(path.is_file() for path in paths)


def test_flexglue_raster_reader_supports_pan_and_msi(tmp_path):
    for count in (1, 3, 8):
        path = Path(tmp_path) / f"image{count}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=12, height=8, count=count, dtype="uint16") as dst:
            dst.write(np.arange(count * 8 * 12, dtype=np.uint16).reshape(count, 8, 12))
        image = SimpleNamespace(geotiff_path=str(path), offset={"col0": 2, "row0": 1, "width": 8, "height": 6})
        rgb = load_rgb(image)
        assert rgb.shape == (3, 6, 8)
        assert rgb.dtype == np.float32
        assert rgb.min() >= 0 and rgb.max() <= 1
