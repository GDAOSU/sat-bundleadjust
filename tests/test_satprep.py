from types import SimpleNamespace

import numpy as np

from bundle_adjust.satprep import RegistrationPipeline


def test_outlier_cleanup_preserves_fixed_controls_and_initial_residuals():
    pipeline = RegistrationPipeline.__new__(RegistrationPipeline)
    pipeline.images = [
        SimpleNamespace(geotiff_path="image{}.tif".format(index), center=np.array([7e6, 0, index]))
        for index in range(3)
    ]
    pipeline.cameras = [None] * 3
    pipeline.cam_model = "rpc"
    pipeline.correction_params = ["R"]
    pipeline.n_adj = 0
    pipeline.n_pts_fix = 1
    pipeline.ref_cam_weight = 1.0
    pipeline.sigma_px = 0.5
    pipeline.max_tie_error_px = 3.0
    pipeline.pairs_to_triangulate = [(0, 1), (0, 2), (1, 2)]

    pipeline.C = np.ones((6, 13))
    pipeline.C_v2 = np.full((3, 13), np.nan)
    # The fixed GCP has only one measured pixel. This is a valid constraint.
    pipeline.C[2:, 0] = np.nan
    pipeline.pts3d = np.tile([6378137.0, 0.0, 0.0], (13, 1))
    pipeline.define_ba_parameters(verbose=False)
    p = pipeline.ba_params
    pipeline.ba_sol = p.params_opt.copy()
    pipeline.init_e = np.arange(p.n_obs, dtype=float)
    pipeline.ba_e = np.full(p.n_obs, 0.5)
    pipeline.ba_e[p.pts_ind == 0] = 500.0
    pipeline.ba_e[(p.pts_ind == 1) & (p.cam_ind == 1)] = 30.0
    pipeline.ba_e[(p.pts_ind == 2) & (p.cam_ind != 0)] = 30.0

    pipeline.clean_outlier_observations()
    new = pipeline.ba_params
    assert new.n_pts_fix == 1
    assert new.n_pts == 12
    assert new.n_cam == 3
    assert new.n_obs == p.n_obs - 4
    np.testing.assert_array_equal(new.C[:, 0], p.C[:, 0])
    np.testing.assert_array_equal(new.pts3d[0], p.pts3d[0])
    np.testing.assert_array_equal(new.pts2d_w[new.pts_ind == 0], [2.0])
    retained = (p.pts_ind != 2) & ~((p.pts_ind == 1) & (p.cam_ind == 1))
    np.testing.assert_array_equal(pipeline.initial_inlier_errors, pipeline.init_e[retained])
    assert pipeline.outlier_statistics["removed_by_error"] == 3
    assert pipeline.outlier_statistics["gcp_observations"] == 1

    # L2 for the tie observations, soft-L1 for manual control residuals.
    calls = []
    pipeline._optimize = lambda loss, stage: calls.append((loss, stage))
    pipeline.run_ba_L2()
    loss, stage = calls[0]
    assert stage == "refine"
    rho = loss(np.full(new.n_obs * 2, 100.0))
    assert np.all(rho[0, :2] < 100)
    np.testing.assert_array_equal(rho[0, 2:], 100)


def test_superpoint_rotation_extracts_aligned_descriptors_and_preserves_pixels(monkeypatch):
    import lightglue
    import torch
    from bundle_adjust.feature_tracks import ft_lightglue

    image = np.zeros((6, 9), dtype=np.uint8)
    image[2, 7] = 255
    observed_shapes = []

    class Extractor:
        def __init__(self, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

        def extract(self, tensor, resize=None):
            observed_shapes.append(tuple(tensor.shape[-2:]))
            row, col = torch.nonzero(tensor.squeeze() > 0.9)[0]
            return {
                "keypoints": torch.stack((col, row)).float().reshape(1, 1, 2),
                "keypoint_scores": torch.ones((1, 1)),
                "descriptors": torch.ones((1, 1, 256)),
            }

    monkeypatch.setattr(ft_lightglue.loader, "load_image", lambda *args, **kwargs: image)
    monkeypatch.setattr(lightglue, "SuperPoint", Extractor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for angle in (0, 90, 180, 270):
        features, count = ft_lightglue.superpoint_detect(
            "fixture.tif", tracks_config={"FT_kp_max": 1, "FT_save": False}, rotation_deg=angle
        )
        assert count == 1
        np.testing.assert_allclose(features[0, :2], [7, 2], atol=1e-5)
    assert observed_shapes == [(6, 9), (9, 6), (6, 9), (9, 6)]
