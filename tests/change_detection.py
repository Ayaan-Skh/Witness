"""
tests/test_day5_6.py — Change Detection Tests

Run with: pytest tests/test_day5_6.py -v

TESTING PHILOSOPHY
──────────────────
Change detection is pure deterministic math — no API calls, no LLM,
no randomness. This means we can test it exhaustively with synthetic
tiles that have known, controlled properties.

The key insight: we don't need real satellite imagery to test the algorithm.
We need tiles whose spectral properties match the scenarios we care about:

  Scenario A — "Healthy forest, no change"
    Both tiles: high NIR, low Red (high NDVI), stable SWIR.
    Expected: near-zero composite score.

  Scenario B — "Deforestation"
    Before: high NIR, low Red (vegetation).
    After:  low NIR, high Red (bare soil).
    Expected: high NDVI drop score, VEGETATION_LOSS signal type.

  Scenario C — "Active fire"
    After tile: very high SWIR1 values in a patch.
    Expected: THERMAL_ANOMALY detected in standalone check + high composite.

  Scenario D — "New construction"
    Before: moderate SWIR, high NIR (mixed vegetation).
    After:  high SWIR, low NIR (concrete/asphalt).
    Expected: NDBI increase, STRUCTURE_CHANGE signal.

  Scenario E — "No change, random noise"
    Both tiles: identical content with tiny random perturbations (<0.01).
    Expected: composite score below threshold (no false positive).

Each test verifies both the numeric score AND the downstream behavior
(whether an AnomalyEvent is produced, what signal_type it has).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from detection.change_detection import (
    compute_ndvi,
    compute_ndbi,
    compute_change_score,
    detect_thermal_anomaly,
    run_change_detection,
    _ndvi_drop_score,
    _spectral_change_score,
    _thermal_score,
    _structure_change_score,
    WEIGHT_NDVI_DROP,
    WEIGHT_SPECTRAL,
    WEIGHT_THERMAL,
    WEIGHT_STRUCTURE,
)
from normalization.schema import AnomalyEvent, AnomalySource, SignalType


# ─────────────────────────────────────────────
# SYNTHETIC TILE FACTORIES
#
# These functions produce tiles with precisely controlled spectral properties.
# Each factory documents which real-world surface type it simulates.
# ─────────────────────────────────────────────

def make_tile(
    h: int = 64,
    w: int = 64,
    blue: float = 0.05,
    green: float = 0.08,
    red: float = 0.07,
    nir: float = 0.40,
    swir1: float = 0.10,
    swir2: float = 0.08,
    noise: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """
    Creates a uniform tile with optional small random noise.
    Band order: [Blue, Green, Red, NIR, SWIR1, SWIR2].
    """
    rng = np.random.default_rng(seed=seed)
    tile = np.stack([
        np.full((h, w), blue,  dtype=np.float32),
        np.full((h, w), green, dtype=np.float32),
        np.full((h, w), red,   dtype=np.float32),
        np.full((h, w), nir,   dtype=np.float32),
        np.full((h, w), swir1, dtype=np.float32),
        np.full((h, w), swir2, dtype=np.float32),
    ], axis=2)
    if noise > 0:
        tile += rng.uniform(-noise, noise, tile.shape).astype(np.float32)
        tile = np.clip(tile, 0.0, 1.0)
    return tile


def make_forest_tile(**kwargs) -> np.ndarray:
    """Dense vegetation: high NIR, low Red → NDVI ≈ 0.70"""
    return make_tile(red=0.06, nir=0.45, swir1=0.08, **kwargs)


def make_bare_soil_tile(**kwargs) -> np.ndarray:
    """Bare soil / deforested: low NIR, moderate Red → NDVI ≈ 0.05"""
    return make_tile(red=0.18, nir=0.22, swir1=0.20, **kwargs)


def make_concrete_tile(**kwargs) -> np.ndarray:
    """
    Urban / built-up: high SWIR, moderate-low NIR → NDBI positive.
    Represents new construction or paved areas.
    """
    return make_tile(red=0.15, nir=0.20, swir1=0.35, swir2=0.28, **kwargs)


def make_fire_tile(**kwargs) -> np.ndarray:
    """
    Active fire: SWIR saturates (heat signature) while NIR stays moderate
    (canopy still present). This separates THERMAL_ANOMALY from VEGETATION_LOSS —
    the dominant signal is the SWIR spike, not an NDVI drop.
    Accepts swir1/swir2 kwargs to allow override in tests.
    """
    defaults = dict(red=0.10, nir=0.38, swir1=0.65, swir2=0.55)
    defaults.update(kwargs)
    return make_tile(**defaults)


def make_water_tile(**kwargs) -> np.ndarray:
    """
    Open water: high Blue/Green, very low NIR/SWIR → NDVI strongly negative.
    """
    return make_tile(blue=0.12, green=0.10, red=0.05, nir=0.02, swir1=0.01, **kwargs)


# ─────────────────────────────────────────────
# NDVI TESTS
# ─────────────────────────────────────────────

class TestComputeNDVI:

    def test_output_range_is_minus1_to_1(self):
        tile = make_forest_tile()
        ndvi = compute_ndvi(tile)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_forest_has_high_positive_ndvi(self):
        tile = make_forest_tile()
        ndvi = compute_ndvi(tile)
        assert ndvi.mean() > 0.5, f"Forest NDVI should be >0.5, got {ndvi.mean():.3f}"

    def test_bare_soil_has_low_ndvi(self):
        tile = make_bare_soil_tile()
        ndvi = compute_ndvi(tile)
        assert ndvi.mean() < 0.25, f"Bare soil NDVI should be <0.25, got {ndvi.mean():.3f}"

    def test_water_has_negative_ndvi(self):
        tile = make_water_tile()
        ndvi = compute_ndvi(tile)
        assert ndvi.mean() < 0.0, f"Water NDVI should be negative, got {ndvi.mean():.3f}"

    def test_output_shape_matches_input_spatial(self):
        tile = make_forest_tile(h=32, w=48)
        ndvi = compute_ndvi(tile)
        assert ndvi.shape == (32, 48)

    def test_zero_denominator_produces_no_nan(self):
        """Pixels where NIR + Red == 0 should give NDVI = 0, not NaN or Inf."""
        tile = np.zeros((10, 10, 6), dtype=np.float32)
        ndvi = compute_ndvi(tile)
        assert not np.any(np.isnan(ndvi))
        assert not np.any(np.isinf(ndvi))
        assert np.all(ndvi == 0.0)

    def test_formula_correctness(self):
        """
        Verify NDVI formula against manually computed values.
        NIR=0.4, Red=0.1 → NDVI = (0.4-0.1)/(0.4+0.1) = 0.6
        """
        tile = make_tile(red=0.1, nir=0.4, noise=0)
        ndvi = compute_ndvi(tile)
        expected = (0.4 - 0.1) / (0.4 + 0.1)
        np.testing.assert_allclose(ndvi, expected, atol=1e-5)

    def test_output_dtype_is_float32(self):
        ndvi = compute_ndvi(make_forest_tile())
        assert ndvi.dtype == np.float32


# ─────────────────────────────────────────────
# NDBI TESTS
# ─────────────────────────────────────────────

class TestComputeNDBI:

    def test_output_range(self):
        ndbi = compute_ndbi(make_concrete_tile())
        assert ndbi.min() >= -1.0
        assert ndbi.max() <= 1.0

    def test_concrete_has_positive_ndbi(self):
        tile = make_concrete_tile()
        ndbi = compute_ndbi(tile)
        assert ndbi.mean() > 0.0, f"Built-up surface should have positive NDBI, got {ndbi.mean():.3f}"

    def test_forest_has_negative_ndbi(self):
        tile = make_forest_tile()
        ndbi = compute_ndbi(tile)
        assert ndbi.mean() < 0.0, f"Vegetation should have negative NDBI, got {ndbi.mean():.3f}"

    def test_formula_correctness(self):
        """
        SWIR1=0.35, NIR=0.20 → NDBI = (0.35-0.20)/(0.35+0.20) ≈ 0.273
        """
        tile = make_tile(nir=0.20, swir1=0.35, noise=0)
        ndbi = compute_ndbi(tile)
        expected = (0.35 - 0.20) / (0.35 + 0.20)
        np.testing.assert_allclose(ndbi, expected, atol=1e-5)


# ─────────────────────────────────────────────
# SUB-SCORE COMPONENT TESTS
# ─────────────────────────────────────────────

class TestNDVIDropScore:

    def test_no_change_gives_zero_score(self):
        tile = make_forest_tile(seed=1)
        score, detail = _ndvi_drop_score(tile, tile.copy())
        assert score < 0.05, f"Identical tiles should give near-zero NDVI drop score, got {score:.3f}"

    def test_deforestation_gives_high_score(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        score, detail = _ndvi_drop_score(before, after)
        assert score > 0.4, f"Deforestation should give high NDVI drop score, got {score:.3f}"

    def test_score_in_0_1_range(self):
        before = make_forest_tile()
        after  = make_bare_soil_tile()
        score, _ = _ndvi_drop_score(before, after)
        assert 0.0 <= score <= 1.0

    def test_detail_dict_has_required_keys(self):
        before = make_forest_tile()
        after  = make_bare_soil_tile()
        _, detail = _ndvi_drop_score(before, after)
        required = {"ndvi_before_mean", "ndvi_after_mean", "ndvi_drop_score", "loss_pixel_fraction"}
        assert required.issubset(detail.keys())

    def test_regrowth_does_not_increase_score(self):
        """NDVI going UP (regrowth) should not inflate the loss score."""
        before = make_bare_soil_tile(noise=0)
        after  = make_forest_tile(noise=0)
        score, _ = _ndvi_drop_score(before, after)
        assert score < 0.1, "Vegetation regrowth should give low NDVI drop score"


class TestSpectralChangeScore:

    def test_identical_tiles_give_near_zero(self):
        tile = make_forest_tile()
        score, _ = _spectral_change_score(tile, tile.copy())
        assert score < 0.05

    def test_large_change_gives_high_score(self):
        score, _ = _spectral_change_score(make_forest_tile(noise=0), make_fire_tile(noise=0))
        assert score > 0.3, f"Forest→fire should give high spectral score, got {score:.3f}"

    def test_score_in_range(self):
        score, _ = _spectral_change_score(make_forest_tile(), make_concrete_tile())
        assert 0.0 <= score <= 1.0

    def test_detail_keys(self):
        _, detail = _spectral_change_score(make_forest_tile(), make_bare_soil_tile())
        assert "spectral_mad_mean" in detail
        assert "spectral_mad_p90" in detail
        assert "spectral_score" in detail


class TestThermalScore:

    def test_no_thermal_change_gives_low_score(self):
        tile = make_forest_tile()
        score, _ = _thermal_score(tile, tile.copy())
        assert score < 0.05

    def test_fire_gives_high_thermal_score(self):
        before = make_forest_tile(noise=0)
        after  = make_fire_tile(noise=0)
        score, _ = _thermal_score(before, after)
        assert score > 0.7, f"Fire should give high thermal score, got {score:.3f}"

    def test_detail_keys(self):
        _, detail = _thermal_score(make_forest_tile(), make_fire_tile())
        assert "swir1_before_mean" in detail
        assert "thermal_score" in detail
        assert "hot_pixel_fraction" in detail


class TestStructureScore:

    def test_no_construction_gives_low_score(self):
        tile = make_forest_tile()
        score, _ = _structure_change_score(tile, tile.copy())
        assert score < 0.05

    def test_new_construction_gives_positive_score(self):
        before = make_forest_tile(noise=0)
        after  = make_concrete_tile(noise=0)
        score, _ = _structure_change_score(before, after)
        assert score > 0.2, f"Forest→concrete should give positive structure score, got {score:.3f}"

    def test_score_in_range(self):
        score, _ = _structure_change_score(make_forest_tile(), make_concrete_tile())
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────
# COMPOSITE SCORE TESTS
# ─────────────────────────────────────────────

class TestComputeChangeScore:

    def test_identical_tiles_score_near_zero(self):
        tile = make_forest_tile(seed=42)
        score, signal, detail = compute_change_score(tile, tile.copy())
        assert score < 0.08, f"Identical tiles should score near zero, got {score:.3f}"

    def test_noisy_identical_tiles_below_threshold(self):
        """
        Two tiles of the same surface type with realistic sensor noise (<0.01)
        should not trigger a false positive above the detection threshold.
        This is the core false-positive guard.
        """
        from config import SATELLITE_CHANGE_THRESHOLD
        tile_before = make_forest_tile(noise=0.008, seed=1)
        tile_after  = make_forest_tile(noise=0.008, seed=2)
        score, _, _ = compute_change_score(tile_before, tile_after)
        assert score < SATELLITE_CHANGE_THRESHOLD, (
            f"Sensor noise alone should not trigger change detection threshold "
            f"({SATELLITE_CHANGE_THRESHOLD}), got score={score:.3f}"
        )

    def test_deforestation_exceeds_threshold(self):
        from config import SATELLITE_CHANGE_THRESHOLD
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        score, _, _ = compute_change_score(before, after)
        assert score >= SATELLITE_CHANGE_THRESHOLD, (
            f"Deforestation should exceed threshold, got {score:.3f}"
        )

    def test_deforestation_gives_vegetation_loss_signal(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        _, signal, _ = compute_change_score(before, after)
        assert signal == SignalType.VEGETATION_LOSS, (
            f"Deforestation should be VEGETATION_LOSS, got {signal}"
        )

    def test_fire_gives_thermal_anomaly_signal(self):
        before = make_forest_tile(noise=0)
        after  = make_fire_tile(noise=0)
        _, signal, _ = compute_change_score(before, after)
        assert signal == SignalType.THERMAL_ANOMALY, (
            f"Fire should be THERMAL_ANOMALY, got {signal}"
        )

    def test_score_in_0_1_range(self):
        for before, after in [
            (make_forest_tile(),  make_bare_soil_tile()),
            (make_forest_tile(),  make_fire_tile()),
            (make_forest_tile(),  make_concrete_tile()),
            (make_water_tile(),   make_forest_tile()),
        ]:
            score, _, _ = compute_change_score(before, after)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_detail_dict_structure(self):
        before = make_forest_tile()
        after  = make_bare_soil_tile()
        _, _, detail = compute_change_score(before, after)
        assert "composite_score"  in detail
        assert "sub_scores"       in detail
        assert "weights"          in detail
        assert "dominant_signal"  in detail
        assert "ndvi_drop" in detail["sub_scores"]
        assert "spectral"  in detail["sub_scores"]
        assert "thermal"   in detail["sub_scores"]
        assert "structure" in detail["sub_scores"]

    def test_weights_sum_to_one(self):
        total = WEIGHT_NDVI_DROP + WEIGHT_SPECTRAL + WEIGHT_THERMAL + WEIGHT_STRUCTURE
        assert abs(total - 1.0) < 1e-6

    def test_composite_matches_manual_calculation(self):
        """
        With noise=0 (uniform tiles), verify composite score matches
        the manual weighted-sum of sub-scores.
        """
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)

        ndvi_s,  _ = _ndvi_drop_score(before, after)
        spec_s,  _ = _spectral_change_score(before, after)
        therm_s, _ = _thermal_score(before, after)
        struct_s,_ = _structure_change_score(before, after)

        expected = (
            WEIGHT_NDVI_DROP  * ndvi_s  +
            WEIGHT_SPECTRAL   * spec_s  +
            WEIGHT_THERMAL    * therm_s +
            WEIGHT_STRUCTURE  * struct_s
        )
        score, _, _ = compute_change_score(before, after)
        assert abs(score - expected) < 1e-5, (
            f"Composite score {score:.6f} should match manual calculation {expected:.6f}"
        )

    def test_multi_signal_convergence_scores_higher_than_single(self):
        """
        A tile whose changes span more sub-signals should score higher than
        one with a single dominant signal. We construct this directly by
        comparing a tile with only a spectral shift vs one with spectral
        shift + strong SWIR increase.
        """
        # Weak change: only slight spectral shift, all weights pulling low
        before = make_tile(red=0.10, nir=0.40, swir1=0.10, noise=0)
        after_mild = make_tile(red=0.12, nir=0.38, swir1=0.12, noise=0)
        score_mild, _, _ = compute_change_score(before, after_mild)

        # Stronger multi-signal: meaningful NDVI drop AND thermal spike
        after_strong = make_tile(red=0.18, nir=0.22, swir1=0.55, noise=0)
        score_strong, _, _ = compute_change_score(before, after_strong)

        assert score_strong > score_mild, (
            f"Multi-signal change ({score_strong:.3f}) should score higher than "
            f"weak single-signal ({score_mild:.3f})"
        )


# ─────────────────────────────────────────────
# THERMAL ANOMALY STANDALONE DETECTION TESTS
# ─────────────────────────────────────────────

class TestDetectThermalAnomaly:

    def test_fire_tile_is_flagged(self):
        tile = make_fire_tile()
        is_anomaly, p99 = detect_thermal_anomaly(tile)
        assert is_anomaly is True, f"Fire tile should be flagged, p99={p99:.3f}"

    def test_forest_tile_not_flagged(self):
        tile = make_forest_tile()
        is_anomaly, p99 = detect_thermal_anomaly(tile)
        assert is_anomaly is False, f"Forest tile should not be flagged, p99={p99:.3f}"

    def test_returns_p99_value(self):
        tile = make_fire_tile(swir1=0.72, noise=0)  # override accepted via defaults.update()
        is_anomaly, p99 = detect_thermal_anomaly(tile)
        assert p99 > 0.0
        assert isinstance(p99, float)

    def test_bare_soil_not_flagged(self):
        tile = make_bare_soil_tile()
        is_anomaly, _ = detect_thermal_anomaly(tile)
        assert is_anomaly is False


# ─────────────────────────────────────────────
# ORCHESTRATION TESTS (run_change_detection)
# ─────────────────────────────────────────────

class TestRunChangeDetection:

    def _pair_result(self, tile_before, tile_after):
        """Helper: returns what fetch_tile_pair_for_region would return."""
        meta = {"target_date": "2021-01-01", "cache_hit": True, "resolution_m": 60}
        return (tile_before, tile_after, meta, meta)

    def test_returns_none_when_tiles_unavailable(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region", return_value=None):
            result = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert result is None

    def test_returns_none_when_below_threshold(self):
        """Identical tiles → score near zero → no event."""
        tile = make_forest_tile()
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(tile, tile.copy()),
        ):
            result = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert result is None

    def test_returns_anomaly_event_on_significant_change(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            result = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert result is not None
        assert isinstance(result, AnomalyEvent)

    def test_event_has_correct_source(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert event.source == AnomalySource.SATELLITE

    def test_event_region_matches_input(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert event.region_id == "eth_tigray"

    def test_event_timestamp_is_after_date(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        target_date = date(2021, 4, 1)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), target_date)
        assert event.timestamp.year  == target_date.year
        assert event.timestamp.month == target_date.month
        assert event.timestamp.day   == target_date.day

    def test_event_intensity_score_in_range(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert 0.0 <= event.intensity_score <= 1.0

    def test_event_raw_data_contains_sub_scores(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert "composite_score" in event.raw_data
        assert "sub_scores"      in event.raw_data
        assert "date_before"     in event.raw_data
        assert "date_after"      in event.raw_data

    def test_event_country_code_matches_region(self):
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert event.country_code == "ET"

    def test_fire_event_is_thermal_anomaly(self):
        before = make_forest_tile(noise=0)
        after  = make_fire_tile(noise=0)
        with patch(
            "detection.change_detection.fetch_tile_pair_for_region",
            return_value=self._pair_result(before, after),
        ):
            event = run_change_detection("eth_tigray", date(2021, 1, 1), date(2021, 4, 1))
        assert event is not None
        assert event.signal_type == SignalType.THERMAL_ANOMALY

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region_id"):
            run_change_detection("not_real", date(2021, 1, 1), date(2021, 4, 1))

    def test_all_monitored_regions_produce_valid_events(self):
        """
        Smoke test: run_change_detection on every configured region should
        either return a valid AnomalyEvent or None — never crash.
        """
        from config import MONITORED_REGIONS
        before = make_forest_tile(noise=0)
        after  = make_bare_soil_tile(noise=0)
        meta   = {"target_date": "2021-01-01", "cache_hit": True, "resolution_m": 60}

        for region in MONITORED_REGIONS:
            with patch(
                "detection.change_detection.fetch_tile_pair_for_region",
                return_value=(before, after, meta, meta),
            ):
                event = run_change_detection(
                    region.region_id, date(2021, 1, 1), date(2021, 4, 1)
                )
            assert event is not None
            assert isinstance(event, AnomalyEvent)
            assert event.region_id == region.region_id
            assert event.country_code == region.country_code


# ─────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_all_zero_tile_does_not_crash(self):
        tile = np.zeros((32, 32, 6), dtype=np.float32)
        score, signal, detail = compute_change_score(tile, tile.copy())
        assert score == 0.0
        assert not np.isnan(score)

    def test_all_ones_tile_does_not_crash(self):
        tile = np.ones((32, 32, 6), dtype=np.float32)
        score, _, _ = compute_change_score(tile, tile.copy())
        assert 0.0 <= score <= 1.0
        assert not np.isnan(score)

    def test_single_pixel_tile(self):
        """Degenerate 1×1 tile should not crash (though scores may be trivial)."""
        before = make_forest_tile(h=1, w=1, noise=0)
        after  = make_bare_soil_tile(h=1, w=1, noise=0)
        score, _, _ = compute_change_score(before, after)
        assert 0.0 <= score <= 1.0

    def test_large_tile_does_not_crash(self):
        """256×256 tile (typical for a 25km² region at 10m) should process fine."""
        before = make_forest_tile(h=256, w=256, noise=0.005)
        after  = make_bare_soil_tile(h=256, w=256, noise=0.005)
        score, _, _ = compute_change_score(before, after)
        assert 0.0 <= score <= 1.0

    def test_score_is_not_nan_for_any_scenario(self):
        scenarios = [
            (make_forest_tile(), make_bare_soil_tile()),
            (make_forest_tile(), make_fire_tile()),
            (make_forest_tile(), make_concrete_tile()),
            (make_water_tile(),  make_forest_tile()),
            (make_fire_tile(),   make_forest_tile()),
        ]
        for before, after in scenarios:
            score, _, _ = compute_change_score(before, after)
            assert not np.isnan(score), f"NaN score for scenario {before[0,0,:]} → {after[0,0,:]}"