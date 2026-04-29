"""
tests/test_day9_10.py — GDELT Ingestion & Anomaly Detection Tests

All BigQuery calls are mocked — no real GCP project required.
Tests cover:
  1. Date format conversion utilities
  2. Timeseries gap-filling
  3. Baseline and z-score math
  4. All three anomaly detectors (tone crash, blackout, volume spike)
  5. run_gdelt_detection orchestration
  6. Intensity score normalisation
  7. Multi-signal precedence and CONFLICT_EVENTS upgrade
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ingestion.gdelt import (
    _gdelt_date,
    _parse_gdelt_date,
    _location_filter,
    fill_missing_dates,
    estimate_query_cost_mb,
    CONCERNING_CAMEO_CODES,
    CONFLICT_CAMEO_CODES,
    DISPLACEMENT_CAMEO_CODES,
)
from detection.gdelt_anomaly import (
    compute_baseline,
    compute_zscore,
    detect_tone_crash,
    detect_communication_blackout,
    detect_volume_spike,
    run_gdelt_detection,
    _zscore_to_intensity,
)
from normalization.schema import AnomalyEvent, AnomalySource, SignalType


# ─────────────────────────────────────────────
# SYNTHETIC TIMESERIES BUILDERS
# ─────────────────────────────────────────────

def make_tone_series(
    start: date,
    days: int,
    baseline_tone: float = -3.0,
    crash_day: int = None,
    crash_value: float = -18.0,
) -> list[dict]:
    """
    Generates a synthetic tone timeseries.
    crash_day: index (0-based) of the day to inject a crash value.
    """
    series = []
    for i in range(days):
        d    = start + timedelta(days=i)
        tone = crash_value if i == crash_day else baseline_tone + np.random.default_rng(i).uniform(-0.5, 0.5)
        series.append({"event_date": d, "avg_tone": tone, "article_count": 100, "event_count": 50})
    return series


def make_volume_series(
    start: date,
    days: int,
    baseline_vol: int = 200,
    blackout_day: int = None,
    blackout_value: int = 5,
    spike_day: int = None,
    spike_value: int = 800,
) -> list[dict]:
    series = []
    for i in range(days):
        d = start + timedelta(days=i)
        if i == blackout_day:
            vol = blackout_value
        elif i == spike_day:
            vol = spike_value
        else:
            vol = baseline_vol + int(np.random.default_rng(i).uniform(-20, 20))
        series.append({"event_date": d, "mention_count": vol, "event_count": vol // 4})
    return series


def make_themes(codes: list[str], concerning_override: bool = None) -> list[dict]:
    return [
        {
            "cameo_code":    c,
            "event_count":   50,
            "mention_count": 200,
            "avg_tone":      -5.0,
            "avg_goldstein": -4.0,
            "is_concerning": (c in CONCERNING_CAMEO_CODES) if concerning_override is None else concerning_override,
        }
        for c in codes
    ]


# ─────────────────────────────────────────────
# DATE FORMAT TESTS
# ─────────────────────────────────────────────

class TestDateFormats:

    def test_gdelt_date_format(self):
        assert _gdelt_date(date(2021, 3, 15)) == 20210315

    def test_gdelt_date_january(self):
        assert _gdelt_date(date(2021, 1, 1)) == 20210101

    def test_parse_gdelt_date_from_int(self):
        assert _parse_gdelt_date(20210315) == date(2021, 3, 15)

    def test_parse_gdelt_date_from_string(self):
        assert _parse_gdelt_date("20210315") == date(2021, 3, 15)

    def test_parse_gdelt_date_roundtrip(self):
        d = date(2022, 11, 30)
        assert _parse_gdelt_date(_gdelt_date(d)) == d

    def test_location_filter_with_admin1(self):
        f = _location_filter("ET", "ET.TI")
        assert "ET.TI" in f
        assert "ADM1Code" in f

    def test_location_filter_country_only(self):
        f = _location_filter("ET", None)
        assert "ET" in f
        assert "CountryCode" in f
        assert "ADM1" not in f


# ─────────────────────────────────────────────
# GAP FILLING TESTS
# ─────────────────────────────────────────────

class TestFillMissingDates:

    def test_no_gaps_unchanged(self):
        start = date(2021, 1, 1)
        series = [{"event_date": start + timedelta(i), "mention_count": 100, "event_count": 25}
                  for i in range(5)]
        filled = fill_missing_dates(series, start, start + timedelta(4))
        assert len(filled) == 5
        assert all(r["mention_count"] == 100 for r in filled)

    def test_gap_filled_with_zero(self):
        start = date(2021, 1, 1)
        # Skip day 2
        series = [
            {"event_date": date(2021, 1, 1), "mention_count": 100, "event_count": 25},
            {"event_date": date(2021, 1, 3), "mention_count": 100, "event_count": 25},
        ]
        filled = fill_missing_dates(series, date(2021, 1, 1), date(2021, 1, 3))
        assert len(filled) == 3
        day2 = next(r for r in filled if r["event_date"] == date(2021, 1, 2))
        assert day2["mention_count"] == 0

    def test_all_missing_returns_zeros(self):
        start = date(2021, 1, 1)
        filled = fill_missing_dates([], start, start + timedelta(4))
        assert len(filled) == 5

    def test_preserves_existing_values(self):
        start = date(2021, 1, 1)
        series = [{"event_date": start, "mention_count": 999, "event_count": 50}]
        filled = fill_missing_dates(series, start, start + timedelta(2))
        assert filled[0]["mention_count"] == 999

    def test_correct_date_count(self):
        """fill_missing_dates should return exactly (end - start + 1) rows."""
        start = date(2021, 1, 1)
        end   = date(2021, 1, 31)
        filled = fill_missing_dates([], start, end)
        assert len(filled) == 31


# ─────────────────────────────────────────────
# BASELINE AND Z-SCORE TESTS
# ─────────────────────────────────────────────

class TestBaseline:

    def test_mean_and_std_correct(self):
        values = [10.0, 12.0, 8.0, 11.0, 9.0]
        mean, std = compute_baseline(values)
        assert abs(mean - 10.0) < 0.01
        assert abs(std  -  1.58) < 0.1

    def test_flat_series_returns_std_one(self):
        """All-identical values should give std=1 to avoid division by zero."""
        mean, std = compute_baseline([5.0, 5.0, 5.0, 5.0])
        assert std == 1.0

    def test_empty_series_safe(self):
        mean, std = compute_baseline([])
        assert mean == 0.0
        assert std  == 1.0

    def test_exclude_zeros_removes_them(self):
        values = [0.0, 10.0, 0.0, 12.0, 0.0]
        mean_with, std_with   = compute_baseline(values, exclude_zeros=False)
        mean_without, _       = compute_baseline(values, exclude_zeros=True)
        assert mean_without > mean_with   # zeros pull mean down

    def test_none_values_excluded(self):
        values = [10.0, None, 12.0, None, 11.0]
        mean, std = compute_baseline(values)
        assert abs(mean - 11.0) < 0.1

    def test_zscore_zero_at_mean(self):
        assert compute_zscore(10.0, 10.0, 2.0) == pytest.approx(0.0)

    def test_zscore_positive_above_mean(self):
        z = compute_zscore(14.0, 10.0, 2.0)
        assert z == pytest.approx(2.0)

    def test_zscore_negative_below_mean(self):
        z = compute_zscore(6.0, 10.0, 2.0)
        assert z == pytest.approx(-2.0)

    def test_zscore_safe_with_zero_std(self):
        z = compute_zscore(5.0, 5.0, 0.0)
        assert z == 0.0

    def test_zscore_formula(self):
        z = compute_zscore(20.0, 10.0, 5.0)
        assert z == pytest.approx((20.0 - 10.0) / 5.0)


# ─────────────────────────────────────────────
# TONE CRASH DETECTOR TESTS
# ─────────────────────────────────────────────

class TestDetectToneCrash:

    def test_normal_tone_no_alarm(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        series = make_tone_series(start, 91, baseline_tone=-3.0)
        _, _, _, zscore = detect_tone_crash("eth_tigray", target, series)
        assert zscore > -2.0, f"Normal tone should not alarm, got z={zscore:.2f}"

    def test_tone_crash_fires(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        # Inject extreme crash on the last day (index 90 = target)
        series = make_tone_series(start, 91, baseline_tone=-3.0,
                                  crash_day=90, crash_value=-22.0)
        tone, mean, std, zscore = detect_tone_crash("eth_tigray", target, series)
        assert zscore <= -2.0, f"Tone crash should give z<=-2, got z={zscore:.2f}"
        assert tone == pytest.approx(-22.0)

    def test_returns_none_tone_when_no_data_for_date(self):
        target = date(2021, 4, 1)
        # Series ends before target date
        series = make_tone_series(date(2021, 1, 1), 30)
        tone, mean, std, z = detect_tone_crash("eth_tigray", target, series)
        assert tone is None

    def test_baseline_mean_close_to_injected_value(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=60)
        series = make_tone_series(start, 61, baseline_tone=-4.0)
        _, mean, _, _ = detect_tone_crash("eth_tigray", target, series)
        assert abs(mean - (-4.0)) < 1.0  # within 1 unit

    def test_current_tone_returned_correctly(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=30)
        series = make_tone_series(start, 30)
        # Add a specific current day
        series.append({"event_date": target, "avg_tone": -7.5, "article_count": 100, "event_count": 40})
        tone, _, _, _ = detect_tone_crash("eth_tigray", target, series)
        assert tone == pytest.approx(-7.5)


# ─────────────────────────────────────────────
# BLACKOUT DETECTOR TESTS
# ─────────────────────────────────────────────

class TestDetectBlackout:

    def test_normal_volume_no_alarm(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        series = make_volume_series(start, 91, baseline_vol=200)
        vol, mean, std, z = detect_communication_blackout("eth_tigray", target, series)
        assert z > -2.0, f"Normal volume should not trigger blackout, got z={z:.2f}"

    def test_blackout_fires(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        series = make_volume_series(start, 91, baseline_vol=200,
                                    blackout_day=90, blackout_value=3)
        vol, mean, std, z = detect_communication_blackout("eth_tigray", target, series)
        assert z <= -2.0, f"Blackout should give z<=-2, got z={z:.2f}"
        assert vol == 3

    def test_missing_target_date_treated_as_zero(self):
        """A completely absent date (no articles at all) is treated as vol=0."""
        target = date(2021, 4, 1)
        start  = target - timedelta(days=30)
        series = make_volume_series(start, 30, baseline_vol=200)
        # Do NOT include target date in series
        vol, mean, std, z = detect_communication_blackout("eth_tigray", target, series)
        assert vol == 0
        assert z <= -2.0


# ─────────────────────────────────────────────
# VOLUME SPIKE DETECTOR TESTS
# ─────────────────────────────────────────────

class TestDetectVolumeSpike:

    def test_normal_volume_no_spike(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        series = make_volume_series(start, 91, baseline_vol=200)
        _, _, _, z = detect_volume_spike("eth_tigray", target, series)
        assert z < 2.0

    def test_spike_detected(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        series = make_volume_series(start, 91, baseline_vol=150,
                                    spike_day=90, spike_value=1500)
        _, _, _, z = detect_volume_spike("eth_tigray", target, series)
        assert z >= 2.0, f"Spike should give z>=2, got z={z:.2f}"


# ─────────────────────────────────────────────
# INTENSITY NORMALISATION TESTS
# ─────────────────────────────────────────────

class TestIntensityNormalisation:

    def test_at_threshold_gives_zero(self):
        assert _zscore_to_intensity(2.0) == pytest.approx(0.0)

    def test_below_threshold_gives_zero(self):
        assert _zscore_to_intensity(1.5) == 0.0

    def test_at_max_anchor_gives_one(self):
        # 2.5× threshold = 5.0 → should give 1.0
        assert _zscore_to_intensity(5.0) >= 1.0 or _zscore_to_intensity(5.0) == pytest.approx(1.0)

    def test_output_always_in_0_1(self):
        for z in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0]:
            result = _zscore_to_intensity(z)
            assert 0.0 <= result <= 1.0, f"z={z} gave intensity={result}"

    def test_negative_z_treated_same_as_positive(self):
        assert _zscore_to_intensity(-3.0) == pytest.approx(_zscore_to_intensity(3.0))


# ─────────────────────────────────────────────
# RUN_GDELT_DETECTION ORCHESTRATION TESTS
# ─────────────────────────────────────────────

class TestRunGdeltDetection:

    def _normal_series(self, target: date):
        start = target - timedelta(days=90)
        tone   = make_tone_series(start, 91, baseline_tone=-3.0)
        volume = make_volume_series(start, 91, baseline_vol=200)
        return tone, volume

    def _crash_series(self, target: date):
        start = target - timedelta(days=90)
        tone   = make_tone_series(start, 91, baseline_tone=-3.0,
                                  crash_day=90, crash_value=-22.0)
        volume = make_volume_series(start, 91, baseline_vol=200)
        return tone, volume

    def _blackout_series(self, target: date):
        start = target - timedelta(days=90)
        tone   = make_tone_series(start, 91)
        volume = make_volume_series(start, 91, baseline_vol=200,
                                    blackout_day=90, blackout_value=3)
        return tone, volume

    def test_no_anomaly_returns_none(self):
        target = date(2021, 4, 1)
        tone, volume = self._normal_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result is None

    def test_tone_crash_produces_event(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result is not None
        assert isinstance(result, AnomalyEvent)

    def test_tone_crash_signal_type(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result.signal_type == SignalType.TONE_CRASH

    def test_blackout_produces_event(self):
        target = date(2021, 4, 1)
        tone, volume = self._blackout_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result is not None
        assert result.signal_type == SignalType.COMMUNICATION_BLACKOUT

    def test_volume_spike_with_conflict_codes_gives_conflict_events(self):
        target = date(2021, 4, 1)
        start  = target - timedelta(days=90)
        tone   = make_tone_series(start, 91)
        volume = make_volume_series(start, 91, baseline_vol=150,
                                    spike_day=90, spike_value=2000)
        themes = make_themes(["193", "190"])  # conflict codes
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=themes)
        assert result is not None
        assert result.signal_type == SignalType.CONFLICT_EVENTS

    def test_event_source_is_gdelt(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result.source == AnomalySource.GDELT

    def test_event_region_matches(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result.region_id == "eth_tigray"

    def test_event_country_code_matches(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result.country_code == "ET"

    def test_event_timestamp_is_target_date(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert result.timestamp.date() == target

    def test_intensity_score_in_range(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert 0.0 <= result.intensity_score <= 1.0

    def test_raw_data_has_required_structure(self):
        target = date(2021, 4, 1)
        tone, volume = self._crash_series(target)
        result = run_gdelt_detection("eth_tigray", target,
                                     tone_series=tone, volume_series=volume, themes=[])
        assert "tone"   in result.raw_data
        assert "volume" in result.raw_data
        assert "all_fired_signals" in result.raw_data

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region_id"):
            run_gdelt_detection("not_real", date(2021, 4, 1),
                                tone_series=[], volume_series=[], themes=[])

    def test_all_monitored_regions_run_without_crash(self):
        from config import MONITORED_REGIONS
        target = date(2021, 4, 1)
        for region in MONITORED_REGIONS:
            start  = target - timedelta(days=90)
            tone   = make_tone_series(start, 91, baseline_tone=-3.0,
                                      crash_day=90, crash_value=-22.0)
            volume = make_volume_series(start, 91)
            result = run_gdelt_detection(region.region_id, target,
                                         tone_series=tone, volume_series=volume,
                                         themes=[])
            assert result is None or isinstance(result, AnomalyEvent)

    def test_fetch_failure_returns_none_gracefully(self):
        """If the BigQuery fetch fails, detection should return None, not raise."""
        with patch("detection.gdelt_anomaly.query_tone_timeseries",
                   side_effect=Exception("BigQuery connection refused")):
            result = run_gdelt_detection("eth_tigray", date(2021, 4, 1))
        assert result is None


# ─────────────────────────────────────────────
# CAMEO CODE SETS TESTS
# ─────────────────────────────────────────────

class TestCAMEOCodeSets:

    def test_concerning_codes_is_union(self):
        assert CONCERNING_CAMEO_CODES == CONFLICT_CAMEO_CODES | DISPLACEMENT_CAMEO_CODES

    def test_conflict_codes_not_empty(self):
        assert len(CONFLICT_CAMEO_CODES) > 5

    def test_displacement_codes_not_empty(self):
        assert len(DISPLACEMENT_CAMEO_CODES) > 2

    def test_known_conflict_code_present(self):
        assert "193" in CONFLICT_CAMEO_CODES   # airstrikes
        assert "204" in CONFLICT_CAMEO_CODES   # kill by physical assault

    def test_cost_estimate_scales_with_days(self):
        cost_10 = estimate_query_cost_mb(10, "timeseries")
        cost_20 = estimate_query_cost_mb(20, "timeseries")
        assert cost_20 == pytest.approx(cost_10 * 2)

    def test_cost_estimate_positive(self):
        assert estimate_query_cost_mb(30) > 0