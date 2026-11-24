"""Detector-free FlexGlue matching using the official RoMa v2 refiners.

The matcher works in native cropped-image pixels. FlexGlue's corner-coordinate
convention and optional quarter turns are converted back before RPC geometry
or feature-track construction.
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .ft_lightglue import affine_image_rotation, common_superpoint_orientations

FLEXGLUE_REVISION = "99920296d082c9405ebfd612d674efe288e74881"


@dataclass
class PairMatches:
    first: int
    second: int
    xy0: np.ndarray
    xy1: np.ndarray
    confidence: np.ndarray


@dataclass
class PreparedImage:
    data: dict
    width: int
    height: int
    rotation_deg: int


def load_rgb(image, bands=None):
    """Read a native-grid RGB composite, supporting PAN and multispectral TIFFs."""
    offset = image.offset
    with rasterio.open(image.geotiff_path) as src:
        if bands is None:
            bands = (5, 3, 2) if src.count == 8 else (1, 2, 3) if src.count >= 3 else (1,)
        values = src.read(
            list(bands),
            window=Window(offset["col0"], offset["row0"], offset["width"], offset["height"]),
            masked=True,
            out_dtype="float32",
        ).filled(np.nan)
    channels = []
    for band in values:
        valid = np.isfinite(band)
        if valid.any():
            lo, hi = np.percentile(band[valid], [1, 99])
            scaled = np.where(valid, np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1), 0)
        else:
            scaled = np.zeros_like(band)
        channels.append(scaled.astype(np.float32))
    rgb = np.stack(channels)
    if len(rgb) == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    if rgb.shape[0] != 3:
        raise ValueError("FlexGlue expects one grayscale band or three visible bands")
    return rgb


def native_pixels(corner_xy, width, height, rotation_deg):
    """Convert FlexGlue corner coordinates to native zero-based pixel centers."""
    pixels = np.asarray(corner_xy, dtype=np.float64) - 0.5
    if rotation_deg:
        affine, _ = affine_image_rotation(width, height, rotation_deg)
        pixels = (pixels - affine[:, 2]) @ affine[:, :2]
    return pixels


class FlexGlueMatcher:
    def __init__(self, max_samples=4096, resolution=512, device=None):
        import torch
        from flexglue import FlexGlue

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if resolution < 32 or resolution % 32:
            raise ValueError("FlexGlue resolution must be a positive multiple of 32")
        if max_samples < 8:
            raise ValueError("FlexGlue requires at least eight samples")
        self.model = FlexGlue(
            refine=True,
            describe_resize=(resolution, resolution),
            refine_resize=(resolution, resolution),
            max_num_samples=max_samples,
        ).eval().to(self.device)

    def prepare(self, rgb, rotation_deg=0):
        import torch

        if rotation_deg not in (0, 90, 180, 270):
            raise ValueError("Expected a quarter-turn rotation")
        rgb = np.asarray(rgb, dtype=np.float32)
        height, width = rgb.shape[-2:]
        if rotation_deg:
            rgb = np.rot90(rgb, k=rotation_deg // 90, axes=(-2, -1)).copy()
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        with torch.inference_mode():
            data = self.model.descriptor.describe(tensor, resize=self.model.conf.describe_resize)
        return PreparedImage(data, width, height, rotation_deg)

    def match(self, first, second):
        import torch
        from flexglue.utils import pack_matches, rbd

        with torch.inference_mode():
            pred = rbd(pack_matches(self.model.match(first.data, second.data)))
            ids = pred["matches"]
            xy0 = pred["keypoints0"][ids[:, 0]].cpu().numpy()
            xy1 = pred["keypoints1"][ids[:, 1]].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
        xy0 = native_pixels(xy0, first.width, first.height, first.rotation_deg)
        xy1 = native_pixels(xy1, second.width, second.height, second.rotation_deg)
        valid = np.isfinite(scores) & np.isfinite(xy0).all(axis=1) & np.isfinite(xy1).all(axis=1)
        for xy, image in ((xy0, first), (xy1, second)):
            valid &= ((xy >= -0.5) & (xy < [image.width - 0.5, image.height - 0.5])).all(axis=1)
        return xy0[valid], xy1[valid], scores[valid]


def filter_pair(xy0, xy1, confidence, ransac_threshold, max_matches):
    """Apply pair geometry before limiting the confidence-ranked observations."""
    import cv2

    if len(xy0) < 8:
        return xy0[:0], xy1[:0], confidence[:0]
    fundamental, mask = cv2.findFundamentalMat(xy0, xy1, cv2.FM_RANSAC, ransac_threshold)
    if fundamental is None or mask is None:
        return xy0[:0], xy1[:0], confidence[:0]
    indices = np.flatnonzero(mask.ravel())
    indices = indices[np.argsort(-confidence[indices], kind="stable")[:max_matches]]
    return xy0[indices], xy1[indices], confidence[indices]


def match_satellite_pairs(images, pairs, config, masks=None):
    """Match overlapping image pairs, caching the dense image descriptions."""
    orientation = config.get("FT_flexglue_orientation", "native")
    if orientation not in ("native", "rpc"):
        raise ValueError("FlexGlue orientation must be native or rpc")
    rotations = common_superpoint_orientations(images) if orientation == "rpc" else None
    bands = config.get("FT_flexglue_bands")
    matcher = FlexGlueMatcher(config["FT_kp_max"], config.get("FT_flexglue_resolution", 512))
    cache = OrderedDict()

    def prepared(index):
        if index not in cache:
            if len(cache) >= 8:
                cache.popitem(last=False)
            rgb = load_rgb(images[index], bands[index] if bands else None)
            angle = rotations[index][2] if rotations else 0
            cache[index] = matcher.prepare(rgb, angle)
        cache.move_to_end(index)
        return cache[index]

    for a, b in pairs:
        xy0, xy1, scores = matcher.match(prepared(a), prepared(b))
        before = len(scores)
        if masks is not None:
            accepted = np.ones(before, dtype=bool)
            for index, xy in ((a, xy0), (b, xy1)):
                mask = np.load(masks[index], allow_pickle=False)
                pixels = np.floor(xy + 0.5).astype(int)
                accepted &= mask[pixels[:, 1], pixels[:, 0]] > 0
            xy0, xy1, scores = xy0[accepted], xy1[accepted], scores[accepted]
        xy0, xy1, scores = filter_pair(xy0, xy1, scores, config["FT_ransac"], config["FT_matches_max"])
        print(f"{len(scores):4} FlexGlue matches (confidence/cycle: {before}, geometry: {len(scores)}) in pair ({a}, {b})", flush=True)
        yield PairMatches(a, b, xy0, xy1, scores)


def _merge_pixels(xy, confidence, radius=0.75):
    """Assign a stable feature index to nearby coordinates without transitive merging."""
    representatives = []
    cells = defaultdict(list)
    indices = np.empty(len(xy), dtype=int)
    for i in np.argsort(-confidence, kind="stable"):
        point = xy[i]
        cell = tuple(np.floor(point / radius).astype(int))
        nearby = [
            index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for index in cells[(cell[0] + dx, cell[1] + dy)]
        ]
        if nearby:
            distances = np.linalg.norm(np.asarray([representatives[index] for index in nearby]) - point, axis=1)
            best = int(np.argmin(distances))
            if distances[best] <= radius:
                indices[i] = nearby[best]
                continue
        indices[i] = len(representatives)
        cells[cell].append(indices[i])
        representatives.append(point)
    return np.asarray(representatives, dtype=float).reshape(-1, 2), indices


def index_pair_matches(pairs, feature_paths, output_dir):
    """Adapt coordinate pairs to SAT-BA's per-image feature arrays and indices.

    Exact pair coordinates remain available to consumers of the pair results.
    """
    per_image = [[] for _ in feature_paths]
    locations = []
    lengths = np.zeros(len(feature_paths), dtype=int)
    for pair in pairs:
        sides = []
        for index, xy in ((pair.first, pair.xy0), (pair.second, pair.xy1)):
            sides.append(slice(lengths[index], lengths[index] + len(xy)))
            lengths[index] += len(xy)
            per_image[index].append((xy, pair.confidence))
        locations.append(sides)

    representatives, mappings = [], []
    for entries in per_image:
        xy = np.concatenate([entry[0] for entry in entries]) if entries else np.empty((0, 2))
        scores = np.concatenate([entry[1] for entry in entries]) if entries else np.empty(0)
        points, mapping = _merge_pixels(xy, scores)
        representatives.append(points)
        mappings.append(mapping)

    size = max(len(points) for points in representatives)
    for path, points in zip(feature_paths, representatives):
        features = np.full((size, 4), np.nan, dtype=np.float32)
        features[:len(points), :2] = points
        features[:len(points), 2:] = [1, 0]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)

    directory = Path(output_dir) / "pairwise_matches"
    directory.mkdir(parents=True, exist_ok=True)
    indexed = []
    for pair, (first, second) in zip(pairs, locations):
        match = np.unique(np.column_stack((mappings[pair.first][first], mappings[pair.second][second])), axis=0)
        name = Path(feature_paths[pair.first]).stem + "_" + Path(feature_paths[pair.second]).stem
        np.save(directory / (name + ".npy"), match)
        if len(match):
            indexed.append(np.column_stack((match, np.tile([pair.first, pair.second], (len(match), 1)))))
    return np.concatenate(indexed) if indexed else np.empty((0, 4), dtype=int)
