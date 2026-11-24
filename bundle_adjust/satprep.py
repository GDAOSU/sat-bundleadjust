"""Offline registration extensions for SAT-BA v2."""

from itertools import combinations
from pathlib import Path

import numpy as np

from . import ba_core, ba_outliers, ba_rpcfit, geo_utils, loader
from .ba_pipeline import BundleAdjustmentPipeline
from .feature_tracks.ft_match import compute_pairs_to_match
from .feature_tracks.ft_triangulate import init_pts3d
from .feature_tracks.ft_utils import filter_C_using_pairs_to_triangulate
from scipy.sparse.csgraph import connected_components


class RegistrationPipeline(BundleAdjustmentPipeline):
    """Adjust RPCs against reusable tie tracks and optional fixed ECEF controls."""

    API_VERSION = 5

    def __init__(self, ba_data, correspondence, altitude, controls=None, sigma_px=1.0,
                 max_iter=100, max_tie_error_px=3.0, initial_tie_points=None):
        self.correspondence = np.asarray(correspondence, dtype=np.float64)
        self.reference_altitude = altitude
        self.controls = controls
        self.sigma_px = sigma_px
        self.max_iter = max_iter
        self.max_tie_error_px = max_tie_error_px
        self.initial_tie_points = initial_tie_points
        self.stage_iterations = {}
        self.outlier_statistics = {}
        super().__init__(
            ba_data,
            tracks_config={"FT_K": 0, "FT_save": True},
            extra_ba_config={
                "cam_model": "rpc",
                "save_figures": False,
                "clean_outliers": True,
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
        if self.initial_tie_points is None:
            triangulated = init_pts3d(self.C, self.cameras, "rpc", self.pairs_to_triangulate).astype(np.float64)
        else:
            triangulated = np.asarray(self.initial_tie_points, dtype=np.float64)
            if triangulated.shape != (self.C.shape[1], 3):
                raise ValueError("Initial points must correspond to the input tie tracks")
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
        self.track_source_indices = np.flatnonzero(counts > 0)
        self.pts3d = points[counts > 0] / counts[counts > 0, None]
        if not self.C.shape[1] or np.any(np.sum(np.isfinite(self.C[::2]), axis=1) < 5):
            raise ValueError("Triangulation left fewer than five usable observations for at least one camera")
        if self.controls is not None:
            coords, observations = self.controls
            self.C = np.hstack((observations, self.C))
            self.pts3d = np.vstack((coords, self.pts3d))
            self.n_pts_fix = len(coords)
            self.track_source_indices = np.concatenate((-1 - np.arange(len(coords)), self.track_source_indices))
        # BundleAdjustmentPipeline also permutes feature bookkeeping when it fixes a camera.
        self.C_v2 = np.full((len(self.images), self.C.shape[1]), np.nan)
        self.features = [None] * len(self.images)

    def define_ba_parameters(self, freeze_all_cams=False, verbose=True):
        super().define_ba_parameters(freeze_all_cams, verbose)
        if self.n_pts_fix:
            fixed_observations = self.ba_params.pts_ind < self.ba_params.n_pts_fix
            self.ba_params.pts2d_w[fixed_observations] = 1.0 / self.sigma_px
        if self.n_adj:
            self.ba_params.rotation_prior_weight = 5e6

    def run_ba_softL1(self):
        self._optimize("soft_l1", "robust")

    def clean_outlier_observations(self):
        """Prune automatic observations, retaining every measured GCP."""
        p = self.ba_params
        fixed = p.n_pts_fix
        automatic = p.pts_ind >= fixed
        C = p.C.copy()
        thresholds = {}
        removed = 0
        for cam in range(p.n_cam):
            observed = automatic & (p.cam_ind == cam)
            errors = self.ba_e[observed]
            if not len(errors):
                continue
            # Use the upstream per-camera elbow rule, with an absolute cap.
            # The cap also applies when the residual distribution has no clear elbow.
            elbow, success = ba_outliers.get_elbow_value(errors) if len(errors) > 1 else (errors[0], False)
            threshold = max(elbow, 1.0) if success else errors.max()
            threshold = min(threshold, self.max_tie_error_px)
            bad = observed & (self.ba_e > threshold)
            C[2 * cam:2 * cam + 2, p.pts_ind[bad]] = np.nan
            removed += int(bad.sum())
            image = self.images[p.cam_prev_indices[cam]]
            thresholds[Path(image.geotiff_path).stem] = float(threshold)

        # Apply the native track cleanup only to triangulated tie points.
        # A manually measured ground point can have a single observation.
        ties = C[:, fixed:]
        keep = np.concatenate((np.arange(fixed), fixed + filter_C_using_pairs_to_triangulate(
            ties, p.pairs_to_triangulate
        )))
        retained = np.isin(p.pts_ind, keep) & np.isfinite(C[2 * p.cam_ind, p.pts_ind])
        if len(keep) <= fixed:
            raise ValueError("Outlier removal left no triangulatable tie tracks")

        # Warm-start the second pass from the robust solution. Keep the
        # original point initialization for diagnostics and datum handling.
        optimized_points, optimized_cameras = p.get_vars_ready_for_fun(self.ba_sol.copy())
        self.C = C[:, keep]
        self.track_source_indices = self.track_source_indices[p.pts_prev_indices[keep]]
        self.C_v2 = self.C_v2[:, p.pts_prev_indices[keep]]
        self.pts3d = p.pts3d[keep]
        incidence = np.isfinite(self.C[::2]).astype(np.int32)
        if connected_components((incidence @ incidence.T) >= 5, directed=False, return_labels=False) != 1:
            raise ValueError("Outlier removal disconnected the camera network. Add reliable matches or adjust the threshold.")
        self.check_connectivity_graph(min_matches=4)
        self.define_ba_parameters(verbose=False)
        new = self.ba_params
        if new.n_cam != p.n_cam or new.n_pts_fix != fixed or new.n_obs != int(retained.sum()):
            raise ValueError("Outlier removal changed the camera or ground-control constraints")
        new.params_opt = np.hstack((optimized_cameras[:, :new.n_params].ravel(), optimized_points[keep].ravel()))
        self.initial_inlier_errors = self.init_e[retained]
        self.outlier_statistics = {
            "max_tie_error_px": float(self.max_tie_error_px),
            "tie_tracks_before": int(p.n_pts - fixed),
            "tie_tracks_after": int(new.n_pts - fixed),
            "tie_observations_before": int(automatic.sum()),
            "tie_observations_after": int((new.pts_ind >= fixed).sum()),
            "removed_by_error": removed,
            "gcp_observations": int((new.pts_ind < fixed).sum()),
            "thresholds_px": thresholds,
        }
        loader.flush_print("Reprojection error thresholds per image: {}".format(thresholds))
        loader.flush_print("Deleted {} tie observations and {} tie tracks. Preserved {} GCP observations.".format(
            self.outlier_statistics["tie_observations_before"] - self.outlier_statistics["tie_observations_after"],
            p.n_pts - new.n_pts, self.outlier_statistics["gcp_observations"]
        ))

    def run_ba_L2(self):
        # Refine the cleaned tie network with L2. Manual controls retain their
        # pixel sigma and robust loss, and their ECEF locations stay fixed.
        gcp_residuals = np.flatnonzero(np.repeat(self.ba_params.pts_ind < self.ba_params.n_pts_fix, 2))

        def control_loss(z):
            rho = np.vstack((z, np.ones_like(z), np.zeros_like(z)))
            t = 1 + z[gcp_residuals]
            rho[0, gcp_residuals] = 2 * (np.sqrt(t) - 1)
            rho[1, gcp_residuals] = t ** -0.5
            rho[2, gcp_residuals] = -0.5 * t ** -1.5
            return rho

        self._optimize(control_loss if len(gcp_residuals) else "linear", "refine")
        self.init_e = self.initial_inlier_errors

    def _optimize(self, loss, stage):
        settings = {"loss": loss, "f_scale": 1.0, "max_iter": self.max_iter}
        _, self.ba_sol, self.init_e, self.ba_e, iters = ba_core.run_ba_optimization(
            self.ba_params, settings, True, False
        )
        self.ba_iters += iters
        self.stage_iterations[stage] = iters

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
