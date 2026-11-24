"""Offline registration extensions for SAT-BA v2."""

from itertools import combinations
from pathlib import Path

import numpy as np

from . import ba_core, ba_rpcfit, geo_utils, loader
from .ba_pipeline import BundleAdjustmentPipeline
from .feature_tracks.ft_match import compute_pairs_to_match
from .feature_tracks.ft_triangulate import init_pts3d


class RegistrationPipeline(BundleAdjustmentPipeline):
    """Adjust RPCs against reusable tie tracks and optional fixed ECEF controls."""

    def __init__(self, ba_data, correspondence, altitude, controls=None, sigma_px=1.0, max_iter=100):
        self.correspondence = np.asarray(correspondence, dtype=np.float64)
        self.reference_altitude = altitude
        self.controls = controls
        self.sigma_px = sigma_px
        self.max_iter = max_iter
        super().__init__(
            ba_data,
            tracks_config={"FT_K": 0, "FT_save": True},
            extra_ba_config={
                "cam_model": "rpc",
                "save_figures": False,
                "clean_outliers": False,
                "fix_ref_cam": controls is None,
                "correction_params": ["R"],
            },
        )

    def set_footprints(self):
        for image in self.images:
            image.set_footprint(alt=self.reference_altitude)

    def compute_feature_tracks(self):
        self.C = self.correspondence.copy()
        seen = np.isfinite(self.C[::2])
        pairs = [(a, b) for a, b in combinations(range(len(self.images)), 2) if np.any(seen[a] & seen[b])]
        footprints = [{"geojson": geo_utils.utm_geojson_from_lonlat_geojson(image.lonlat_geojson), "z": image.alt}
                      for image in self.images]
        _, self.pairs_to_triangulate = compute_pairs_to_match(pairs, footprints, [image.center for image in self.images])
        self.feature_tracks_running_time = 0.0

    def initialize_pts3d(self):
        # A weak baseline can support pixel residuals even when depth is underdetermined.
        triangulated = init_pts3d(self.C, self.cameras, "rpc", self.pairs_to_triangulate).astype(np.float64)
        valid = np.isfinite(triangulated).all(axis=1) & (np.linalg.norm(triangulated, axis=1) > 6e6)
        points = np.where(valid[:, None], triangulated, 0.0)
        counts = valid.astype(int)
        for index, camera in enumerate(self.cameras):
            xy = self.C[2 * index:2 * index + 2]
            missing = ~valid & np.isfinite(xy[0])
            if not missing.any():
                continue
            heights = np.full(missing.sum(), self.reference_altitude)
            lon, lat = camera.localization(*xy[:, missing], heights)
            xyz = np.column_stack(geo_utils.latlon_to_ecef_custom(lat, lon, heights))
            finite = np.isfinite(xyz).all(axis=1)
            selected = np.flatnonzero(missing)[finite]
            points[selected] += xyz[finite]
            counts[selected] += 1
        self.C = self.C[:, counts > 0]
        self.pts3d = points[counts > 0] / counts[counts > 0, None]
        if not self.C.shape[1] or np.any(np.sum(np.isfinite(self.C[::2]), axis=1) < 5):
            raise ValueError("Triangulation left fewer than five usable observations for at least one camera")
        if self.controls is not None:
            coords, observations = self.controls
            self.C = np.hstack((observations, self.C))
            self.pts3d = np.vstack((coords, self.pts3d))
            self.n_pts_fix = len(coords)
        # BundleAdjustmentPipeline also permutes feature bookkeeping when it fixes a camera.
        self.C_v2 = np.full((len(self.images), self.C.shape[1]), np.nan)
        self.features = [None] * len(self.images)

    def define_ba_parameters(self, freeze_all_cams=False, verbose=True):
        super().define_ba_parameters(freeze_all_cams, verbose)
        if self.n_pts_fix:
            fixed_observations = self.ba_params.pts_ind < self.ba_params.n_pts_fix
            self.ba_params.pts2d_w[fixed_observations] = 1.0 / self.sigma_px

    def run_ba_L2(self):
        # Retain robust residuals throughout the solve, including manual observations.
        settings = {"loss": "soft_l1", "f_scale": 1.0, "max_iter": self.max_iter}
        if self.n_adj:
            self.ba_params.rotation_prior_weight = 5e6
        _, self.ba_sol, self.init_e, self.ba_e, iters = ba_core.run_ba_optimization(
            self.ba_params, settings, True, False
        )
        self.ba_iters += iters

    def save_corrected_rpcs(self):
        super().save_corrected_rpcs()
        if self.n_adj and self.global_transform is not None:
            reference = self.images[0]
            seen = np.isfinite(self.ba_params.C[0])
            shifted, _, _ = ba_rpcfit.fit_Rt_corrected_rpc(
                self.corrected_cameras[0].reshape(1, 9),
                self.global_transform,
                reference.rpc,
                reference.offset,
                self.ba_params.pts3d_ba[seen],
            )
            shifted.write_to_file(str(Path(self.out_dir) / "rpcs_adj" / (loader.get_id(reference.geotiff_path) + ".rpc_adj")))

    def correct_drift_object_space(self):
        if self.n_pts_fix:
            self.global_transform = None
            loader.flush_print("The fixed ground control coordinates define the output datum.")
        else:
            super().correct_drift_object_space()
