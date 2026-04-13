"""
Satellite Ingestion Tests



TESTING STRATEGY
────────────────
The Sentinel Hub API requires real credentials and makes network calls.
We never make real API calls in unit tests — that would:
  1. Fail in CI environments (no credentials)
  2. Burn through the free-tier quota
  3. Make tests slow (seconds per call) and non-deterministic (cloud cover varies)

Instead, we use unittest.mock to replace the API call with a function that
returns a synthetic tile array. This lets us test all the logic around the
API call — caching, array normalisation, shape validation, error handling —
without touching the network.

Think of it like testing a librarian's checkout procedure using fake books.
You don't need real books to verify the checkout stamp works correctly.

Tests are organized into:
  1. Cache utilities — key generation, read/write/clear
  2. get_tile() logic — cache hits, normalization, shape handling
  3. get_tile_pair() — mismatched shape handling, None propagation
  4. Region-level wrappers — region_id validation
  5. Persistence — save_tile_to_disk (npy format, no rasterio needed)
  6. Cache stats and listing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ingestion.satellite import (
    _make_cache_key,
    _cache_path,
    _read_from_cache,
    _write_to_cache,
    get_tile,
    get_tile_pair,
    fetch_tile_for_region,
    fetch_tile_pair_for_region,
    save_tile_to_disk,
    list_cached_tiles,
    get_cache_stats,
    clear_cache,
    BAND_NIR, BAND_RED, BAND_GREEN, BAND_SWIR1, BAND_SWIR2,
)
from scripts.visualize_tiles import compute_ndvi_display, make_true_color


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """
    Redirects TILE_CACHE_DIR to a temporary directory for every test.
    This ensures tests don't pollute the real cache and don't depend on
    tiles from previous test runs.

    autouse=True means this runs automatically for EVERY test in this file.
    """
    monkeypatch.setattr("ingestion.satellite.TILE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("ingestion.satellite.TILE_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_bbox():
    """Tigray bounding box — used across most tests."""
    return [36.45, 12.30, 40.00, 15.00]


@pytest.fixture
def sample_date():
    return date(2021, 3, 15)


@pytest.fixture
def synthetic_tile():
    """
    A realistic synthetic tile: shape (64, 64, 6) float32 in [0, 1].

    64×64 pixels is small enough for fast tests but large enough to
    run real NDVI and change detection computations on.

    The 6 channels correspond to B02, B03, B04, B08, B11, B12.
    We give NIR (channel 3) a higher value than Red (channel 2)
    to simulate healthy vegetation (positive NDVI).
    """
    rng = np.random.default_rng(seed=42)
    tile = rng.uniform(0.05, 0.30, (64, 64, 6)).astype(np.float32)
    # Make NIR (ch3) > Red (ch2) so NDVI is positive — simulates vegetation
    tile[:, :, BAND_NIR] = rng.uniform(0.35, 0.65, (64, 64)).astype(np.float32)
    tile[:, :, BAND_RED] = rng.uniform(0.05, 0.15, (64, 64)).astype(np.float32)
    return tile


@pytest.fixture
def synthetic_tile_changed(synthetic_tile):
    """
    A modified tile simulating land cover change:
    NIR drops significantly (vegetation removed), Red increases (bare soil).
    """
    tile = synthetic_tile.copy()
    tile[:, :, BAND_NIR] = tile[:, :, BAND_NIR] * 0.35   # sharp NIR drop
    tile[:, :, BAND_RED] = tile[:, :, BAND_RED] * 2.0    # red increases (soil)
    tile = np.clip(tile, 0, 1)
    return tile


# ─────────────────────────────────────────────
# CACHE KEY TESTS
# ─────────────────────────────────────────────

class TestCacheKey:

    def test_same_inputs_produce_same_key(self, sample_bbox, sample_date):
        key1 = _make_cache_key(sample_bbox, sample_date, "analysis")
        key2 = _make_cache_key(sample_bbox, sample_date, "analysis")
        assert key1 == key2

    def test_different_dates_produce_different_keys(self, sample_bbox):
        key1 = _make_cache_key(sample_bbox, date(2021, 1, 1), "analysis")
        key2 = _make_cache_key(sample_bbox, date(2021, 6, 1), "analysis")
        assert key1 != key2

    def test_different_evalscripts_produce_different_keys(self, sample_bbox, sample_date):
        key1 = _make_cache_key(sample_bbox, sample_date, "analysis")
        key2 = _make_cache_key(sample_bbox, sample_date, "true_color")
        assert key1 != key2

    def test_different_bboxes_produce_different_keys(self, sample_date):
        key1 = _make_cache_key([36.0, 12.0, 40.0, 15.0], sample_date, "analysis")
        key2 = _make_cache_key([73.0, 34.0, 96.0, 49.0], sample_date, "analysis")
        assert key1 != key2

    def test_key_contains_date_prefix(self, sample_bbox, sample_date):
        """Keys should have date prefix for human readability."""
        key = _make_cache_key(sample_bbox, sample_date, "analysis")
        assert "20210315" in key

    def test_key_is_filesystem_safe(self, sample_bbox, sample_date):
        """Cache keys must be valid filenames — no slashes, spaces, or colons."""
        key = _make_cache_key(sample_bbox, sample_date, "analysis")
        invalid_chars = set('/\\:*?"<>| ')
        assert not any(c in key for c in invalid_chars)

    def test_float_precision_doesnt_affect_key(self, sample_date):
        """Tiny float differences from IEEE 754 representation should not change the key."""
        bbox1 = [36.45, 12.30, 40.00, 15.00]
        bbox2 = [36.4500000001, 12.3000000001, 40.0000000001, 15.0000000001]
        key1 = _make_cache_key(bbox1, sample_date, "analysis")
        key2 = _make_cache_key(bbox2, sample_date, "analysis")
        assert key1 == key2


# ─────────────────────────────────────────────
# CACHE READ/WRITE TESTS
# ─────────────────────────────────────────────

class TestCacheReadWrite:

    def test_write_then_read_roundtrip(self, synthetic_tile, isolated_cache):
        key = "test_tile_001"
        meta = {"target_date": "2021-03-15", "resolution_m": 10}
        _write_to_cache(key, synthetic_tile, meta)

        result = _read_from_cache(key)
        assert result is not None
        arr, recovered_meta = result
        np.testing.assert_array_equal(arr, synthetic_tile)
        assert recovered_meta["target_date"] == "2021-03-15"

    def test_read_nonexistent_returns_none(self, isolated_cache):
        result = _read_from_cache("this_key_does_not_exist")
        assert result is None

    def test_write_creates_both_files(self, synthetic_tile, isolated_cache):
        key = "test_tile_002"
        _write_to_cache(key, synthetic_tile, {"date": "2021-01-01"})
        arr_path, meta_path = _cache_path(key)
        assert arr_path.exists(), ".npy file should exist after write"
        assert meta_path.exists(), ".meta.json file should exist after write"

    def test_corrupted_npy_returns_none_and_deletes(self, isolated_cache):
        """A corrupted cache file should return None, not raise an exception."""
        key = "corrupted_tile"
        arr_path, meta_path = _cache_path(key)
        arr_path.write_bytes(b"not a numpy file")  # write garbage
        meta_path.write_text('{"date": "2021-01-01"}')

        result = _read_from_cache(key)
        assert result is None
        # Files should be cleaned up
        assert not arr_path.exists(), "Corrupted .npy should be deleted"

    def test_array_shape_preserved(self, isolated_cache):
        """Float32 arrays with shape (64, 64, 6) should round-trip exactly."""
        arr = np.zeros((64, 64, 6), dtype=np.float32)
        arr[10, 10, :] = [0.1, 0.2, 0.3, 0.7, 0.05, 0.04]
        _write_to_cache("shape_test", arr, {})
        result = _read_from_cache("shape_test")
        assert result is not None
        recovered, _ = result
        assert recovered.shape == (64, 64, 6)
        assert recovered.dtype == np.float32
        np.testing.assert_array_almost_equal(recovered[10, 10, :], arr[10, 10, :])


# ─────────────────────────────────────────────
# get_tile() TESTS
# ─────────────────────────────────────────────

class TestGetTile:

    def _mock_sh_request(self, tile: np.ndarray):
        """
        Returns a mock SentinelHubRequest whose get_data() returns our
        synthetic tile scaled to Sentinel Hub's [0, 10000] range.
        """
        mock_request = MagicMock()
        # Sentinel Hub returns values in 0–10000 range; our code normalises to 0–1.
        mock_request.get_data.return_value = [tile * 10000.0]
        return mock_request

    def test_returns_cached_tile_without_api_call(
        self, synthetic_tile, sample_bbox, sample_date, isolated_cache
    ):
        """If the tile is cached, get_tile() must not touch the API."""
        cache_key = _make_cache_key(sample_bbox, sample_date, "analysis")
        _write_to_cache(cache_key, synthetic_tile, {
            "target_date": sample_date.isoformat(),
            "resolution_m": 10,
        })

        # Patch the API class — if it's called, the test fails
        with patch("ingestion.satellite.SentinelHubRequest") as mock_api:
            result = get_tile(sample_bbox, sample_date)
            mock_api.assert_not_called()

        assert result is not None
        arr, meta = result
        assert meta["cache_hit"] is True
        np.testing.assert_array_equal(arr, synthetic_tile)

    def test_normalises_values_to_0_1(
        self, synthetic_tile, sample_bbox, sample_date, isolated_cache
    ):
        """Raw Sentinel Hub values (0–10000) must be normalised to [0, 1]."""
        mock_req = self._mock_sh_request(synthetic_tile)

        with patch("ingestion.satellite.SENTINELHUB_AVAILABLE", True), \
             patch("ingestion.satellite._get_sh_config", return_value=MagicMock()), \
             patch("ingestion.satellite.SentinelHubRequest", return_value=mock_req), \
             patch("ingestion.satellite.BBox", return_value=MagicMock()), \
             patch("ingestion.satellite.CRS", MagicMock()), \
             patch("ingestion.satellite.DataCollection", MagicMock()), \
             patch("ingestion.satellite.MosaickingOrder", MagicMock()), \
             patch("ingestion.satellite.MimeType", MagicMock()), \
             patch("ingestion.satellite.bbox_to_dimensions", return_value=(64, 64)):

            result = get_tile(sample_bbox, sample_date, force_refresh=True)

        assert result is not None
        arr, meta = result
        assert arr.max() <= 1.0, f"Max value should be ≤ 1.0, got {arr.max()}"
        assert arr.min() >= 0.0, f"Min value should be ≥ 0.0, got {arr.min()}"

    def test_returns_none_when_api_returns_empty(
        self, sample_bbox, sample_date, isolated_cache
    ):
        """If Sentinel Hub returns no data (fully cloudy), return None gracefully."""
        mock_req = MagicMock()
        mock_req.get_data.return_value = []

        with patch("ingestion.satellite.SENTINELHUB_AVAILABLE", True), \
             patch("ingestion.satellite._get_sh_config", return_value=MagicMock()), \
             patch("ingestion.satellite.SentinelHubRequest", return_value=mock_req), \
             patch("ingestion.satellite.BBox", return_value=MagicMock()), \
             patch("ingestion.satellite.CRS", MagicMock()), \
             patch("ingestion.satellite.DataCollection", MagicMock()), \
             patch("ingestion.satellite.MosaickingOrder", MagicMock()), \
             patch("ingestion.satellite.MimeType", MagicMock()), \
             patch("ingestion.satellite.bbox_to_dimensions", return_value=(64, 64)):

            result = get_tile(sample_bbox, sample_date, force_refresh=True)

        assert result is None

    def test_cache_miss_writes_to_cache(
        self, synthetic_tile, sample_bbox, sample_date, isolated_cache
    ):
        """After a successful API fetch, the tile should be saved to cache."""
        mock_req = self._mock_sh_request(synthetic_tile)

        with patch("ingestion.satellite.SENTINELHUB_AVAILABLE", True), \
             patch("ingestion.satellite._get_sh_config", return_value=MagicMock()), \
             patch("ingestion.satellite.SentinelHubRequest", return_value=mock_req), \
             patch("ingestion.satellite.BBox", return_value=MagicMock()), \
             patch("ingestion.satellite.CRS", MagicMock()), \
             patch("ingestion.satellite.DataCollection", MagicMock()), \
             patch("ingestion.satellite.MosaickingOrder", MagicMock()), \
             patch("ingestion.satellite.MimeType", MagicMock()), \
             patch("ingestion.satellite.bbox_to_dimensions", return_value=(64, 64)):

            result = get_tile(sample_bbox, sample_date, force_refresh=True)

        assert result is not None
        cache_key = _make_cache_key(sample_bbox, sample_date, "analysis")
        cached = _read_from_cache(cache_key)
        assert cached is not None, "Tile should be in cache after a successful fetch"

    def test_force_refresh_bypasses_cache(
        self, synthetic_tile, sample_bbox, sample_date, isolated_cache
    ):
        """force_refresh=True should call the API even if a cache entry exists."""
        cache_key = _make_cache_key(sample_bbox, sample_date, "analysis")
        stale_tile = np.zeros_like(synthetic_tile)
        _write_to_cache(cache_key, stale_tile, {"target_date": sample_date.isoformat()})

        mock_req = self._mock_sh_request(synthetic_tile)

        with patch("ingestion.satellite.SENTINELHUB_AVAILABLE", True), \
             patch("ingestion.satellite._get_sh_config", return_value=MagicMock()), \
             patch("ingestion.satellite.SentinelHubRequest", return_value=mock_req), \
             patch("ingestion.satellite.BBox", return_value=MagicMock()), \
             patch("ingestion.satellite.CRS", MagicMock()), \
             patch("ingestion.satellite.DataCollection", MagicMock()), \
             patch("ingestion.satellite.MosaickingOrder", MagicMock()), \
             patch("ingestion.satellite.MimeType", MagicMock()), \
             patch("ingestion.satellite.bbox_to_dimensions", return_value=(64, 64)):

            result = get_tile(sample_bbox, sample_date, force_refresh=True)

        assert result is not None
        arr, _ = result
        # Should return the fresh tile (not all zeros)
        assert arr.sum() > 0

    def test_metadata_contains_required_fields(
        self, synthetic_tile, sample_bbox, sample_date, isolated_cache
    ):
        """tile metadata must contain the fields downstream modules depend on."""
        mock_req = self._mock_sh_request(synthetic_tile)

        with patch("ingestion.satellite.SENTINELHUB_AVAILABLE", True), \
             patch("ingestion.satellite._get_sh_config", return_value=MagicMock()), \
             patch("ingestion.satellite.SentinelHubRequest", return_value=mock_req), \
             patch("ingestion.satellite.BBox", return_value=MagicMock()), \
             patch("ingestion.satellite.CRS", MagicMock()), \
             patch("ingestion.satellite.DataCollection", MagicMock()), \
             patch("ingestion.satellite.MosaickingOrder", MagicMock()), \
             patch("ingestion.satellite.MimeType", MagicMock()), \
             patch("ingestion.satellite.bbox_to_dimensions", return_value=(64, 64)):

            result = get_tile(sample_bbox, sample_date, force_refresh=True)

        _, meta = result
        required_fields = {"bbox", "target_date", "resolution_m", "shape", "cache_hit"}
        assert required_fields.issubset(meta.keys()), (
            f"Missing metadata fields: {required_fields - meta.keys()}"
        )


# ─────────────────────────────────────────────
# get_tile_pair() TESTS
# ─────────────────────────────────────────────

class TestGetTilePair:

    def test_returns_none_if_before_tile_unavailable(
        self, sample_bbox, isolated_cache
    ):
        """If either tile is missing (cloudy), the pair should return None."""
        with patch("ingestion.satellite.get_tile", side_effect=[None, MagicMock()]):
            result = get_tile_pair(
                sample_bbox, date(2021, 1, 1), date(2021, 4, 1)
            )
        assert result is None

    def test_returns_none_if_after_tile_unavailable(
        self, synthetic_tile, sample_bbox, isolated_cache
    ):
        meta = {"target_date": "2021-01-01", "cache_hit": False}
        with patch("ingestion.satellite.get_tile", side_effect=[
            (synthetic_tile, meta),
            None,
        ]):
            result = get_tile_pair(
                sample_bbox, date(2021, 1, 1), date(2021, 4, 1)
            )
        assert result is None

    def test_returns_four_tuple_on_success(
        self, synthetic_tile, sample_bbox, isolated_cache
    ):
        meta_b = {"target_date": "2021-01-01", "cache_hit": True, "resolution_m": 10}
        meta_a = {"target_date": "2021-04-01", "cache_hit": True, "resolution_m": 10}

        with patch("ingestion.satellite.get_tile", side_effect=[
            (synthetic_tile, meta_b),
            (synthetic_tile, meta_a),
        ]):
            result = get_tile_pair(
                sample_bbox, date(2021, 1, 1), date(2021, 4, 1)
            )

        assert result is not None
        assert len(result) == 4
        tile_before, tile_after, meta_before, meta_after = result
        assert tile_before.shape == tile_after.shape

    def test_both_tiles_share_same_shape(
        self, synthetic_tile, sample_bbox, isolated_cache
    ):
        meta_b = {"target_date": "2021-01-01", "cache_hit": True}
        meta_a = {"target_date": "2021-04-01", "cache_hit": True}

        with patch("ingestion.satellite.get_tile", side_effect=[
            (synthetic_tile, meta_b),
            (synthetic_tile.copy(), meta_a),
        ]):
            result = get_tile_pair(
                sample_bbox, date(2021, 1, 1), date(2021, 4, 1)
            )

        tile_before, tile_after, _, _ = result
        assert tile_before.shape == tile_after.shape, (
            "Before and after tiles must have identical shapes for pixel-wise comparison"
        )


# ─────────────────────────────────────────────
# REGION WRAPPER TESTS
# ─────────────────────────────────────────────

class TestRegionWrappers:

    def test_fetch_for_valid_region(self, synthetic_tile, isolated_cache):
        meta = {"target_date": "2021-03-15", "cache_hit": False}
        with patch("ingestion.satellite.get_tile", return_value=(synthetic_tile, meta)):
            result = fetch_tile_for_region("eth_tigray", date(2021, 3, 15))
        assert result is not None

    def test_fetch_for_unknown_region_raises(self, isolated_cache):
        with pytest.raises(ValueError, match="Unknown region_id"):
            fetch_tile_for_region("not_a_real_region", date(2021, 3, 15))

    def test_fetch_pair_for_valid_region(self, synthetic_tile, isolated_cache):
        meta = {"target_date": "2021-01-01", "cache_hit": False}
        with patch("ingestion.satellite.get_tile", return_value=(synthetic_tile, meta)):
            result = fetch_tile_pair_for_region(
                "ukr_mariupol", date(2022, 1, 1), date(2022, 4, 1)
            )
        assert result is not None

    def test_all_configured_regions_have_valid_bboxes(self):
        """Smoke test: every region in config.py should work as a region_id."""
        from config import MONITORED_REGIONS
        meta = {"target_date": "2021-01-01", "cache_hit": True}
        synthetic = np.zeros((32, 32, 6), dtype=np.float32)

        for region in MONITORED_REGIONS:
            with patch("ingestion.satellite.get_tile", return_value=(synthetic, meta)):
                result = fetch_tile_for_region(region.region_id, date(2021, 1, 1))
            assert result is not None, f"Failed for region: {region.region_id}"


# ─────────────────────────────────────────────
# PERSISTENCE TESTS
# ─────────────────────────────────────────────

class TestSaveTileToDisk:

    def test_save_as_npy(self, synthetic_tile, tmp_path):
        path = str(tmp_path / "test_tile")
        saved_path = save_tile_to_disk(synthetic_tile, path, as_geotiff=False)
        assert saved_path.endswith(".npy")
        assert Path(saved_path).exists()

    def test_npy_roundtrip(self, synthetic_tile, tmp_path):
        path = str(tmp_path / "roundtrip")
        saved_path = save_tile_to_disk(synthetic_tile, path, as_geotiff=False)
        loaded = np.load(saved_path)
        np.testing.assert_array_equal(loaded, synthetic_tile)

    def test_creates_parent_directories(self, synthetic_tile, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "tile")
        save_tile_to_disk(synthetic_tile, deep_path, as_geotiff=False)
        assert Path(deep_path).with_suffix(".npy").exists()

    def test_geotiff_requires_bbox(self, synthetic_tile, tmp_path):
        path = str(tmp_path / "test.tif")
        with pytest.raises(ValueError, match="bbox must be provided"):
            save_tile_to_disk(synthetic_tile, path, as_geotiff=True, bbox=None)


# ─────────────────────────────────────────────
# CACHE STATS AND MANAGEMENT TESTS
# ─────────────────────────────────────────────

class TestCacheManagement:

    def test_empty_cache_stats(self, isolated_cache):
        stats = get_cache_stats()
        assert stats["tile_count"] == 0
        assert stats["total_size_mb"] == 0

    def test_cache_stats_after_writes(self, synthetic_tile, isolated_cache):
        _write_to_cache("tile_1", synthetic_tile, {"bbox": [1, 2, 3, 4]})
        _write_to_cache("tile_2", synthetic_tile, {"bbox": [5, 6, 7, 8]})
        stats = get_cache_stats()
        assert stats["tile_count"] == 2

    def test_list_cached_tiles_empty(self, isolated_cache):
        tiles = list_cached_tiles()
        assert tiles == []

    def test_list_cached_tiles_returns_metadata(self, synthetic_tile, isolated_cache):
        _write_to_cache("my_tile", synthetic_tile, {
            "target_date": "2021-03-15",
            "resolution_m": 10,
            "bbox": [36.45, 12.30, 40.00, 15.00],
        })
        tiles = list_cached_tiles()
        assert len(tiles) == 1
        assert tiles[0]["target_date"] == "2021-03-15"

    def test_clear_cache_deletes_all(self, synthetic_tile, isolated_cache):
        _write_to_cache("tile_a", synthetic_tile, {})
        _write_to_cache("tile_b", synthetic_tile, {})
        deleted = clear_cache()
        assert deleted == 2
        assert get_cache_stats()["tile_count"] == 0


# ─────────────────────────────────────────────
# NDVI COMPUTATION TESTS (from visualize_tiles.py)
# ─────────────────────────────────────────────

class TestNDVIComputation:

    def test_ndvi_range(self, synthetic_tile):
        """NDVI must always be in [-1, 1]."""
        ndvi = compute_ndvi_display(synthetic_tile)
        assert ndvi.min() >= -1.0, f"NDVI below -1: {ndvi.min()}"
        assert ndvi.max() <= 1.0,  f"NDVI above  1: {ndvi.max()}"

    def test_vegetation_has_positive_ndvi(self, synthetic_tile):
        """Tiles with NIR > Red should have predominantly positive NDVI."""
        # synthetic_tile fixture sets NIR > Red to simulate vegetation
        ndvi = compute_ndvi_display(synthetic_tile)
        positive_fraction = (ndvi > 0).mean()
        assert positive_fraction > 0.8, (
            f"Expected >80% positive NDVI for vegetation tile, got {positive_fraction:.0%}"
        )

    def test_changed_tile_has_lower_ndvi(self, synthetic_tile, synthetic_tile_changed):
        """After land cover change (NIR drop), mean NDVI should decrease."""
        ndvi_before = compute_ndvi_display(synthetic_tile)
        ndvi_after  = compute_ndvi_display(synthetic_tile_changed)
        assert ndvi_after.mean() < ndvi_before.mean(), (
            "Changed tile (vegetation removed) should have lower NDVI than before"
        )

    def test_ndvi_shape_matches_input(self, synthetic_tile):
        """NDVI output should have same (H, W) spatial shape as input."""
        ndvi = compute_ndvi_display(synthetic_tile)
        assert ndvi.shape == (64, 64)

    def test_zero_denominator_handled(self):
        """Zero NIR + Red (no-data pixels) should produce NDVI = 0, not NaN."""
        arr = np.zeros((10, 10, 6), dtype=np.float32)  # all zeros
        ndvi = compute_ndvi_display(arr)
        assert not np.any(np.isnan(ndvi)), "NDVI should not contain NaN values"
        assert not np.any(np.isinf(ndvi)), "NDVI should not contain Inf values"

    def test_true_color_shape_and_dtype(self, synthetic_tile):
        """True color output must be (H, W, 3) uint8 in [0, 255]."""
        tc = make_true_color(synthetic_tile)
        assert tc.shape == (64, 64, 3)
        assert tc.dtype == np.uint8
        assert tc.min() >= 0
        assert tc.max() <= 255