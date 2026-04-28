"""
tests/test_day7_8.py — Satellite Pipeline Integration Tests

Tests the full path:
  fetch_tile_pair → change_detection → AnomalyEvent → pipeline stage → DB persist

All external I/O is mocked (Sentinel Hub API, PostgreSQL).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, call
import json

import numpy as np
import pytest

from scheduler.pipeline import run_satellite_stage, run_pipeline, run_backfill
from detection.change_detection import run_change_detection
from normalization.schema import AnomalyEvent, AnomalySource, SignalType
from config import MONITORED_REGIONS, REGIONS_BY_ID


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

def make_tile(red, nir, swir1, h=32, w=32):
    tile = np.zeros((h, w, 6), dtype=np.float32)
    tile[:, :, 2] = red
    tile[:, :, 3] = nir
    tile[:, :, 4] = swir1
    return tile


def forest_tile():  return make_tile(red=0.06, nir=0.45, swir1=0.08)
def cleared_tile(): return make_tile(red=0.18, nir=0.22, swir1=0.20)


def mock_tile_pair(before=None, after=None):
    b = forest_tile() if before is None else before
    a = cleared_tile() if after is None else after
    meta = {"target_date": "2021-03-01", "cache_hit": True, "resolution_m": 60}
    return (b, a, meta, meta)


# ─────────────────────────────────────────────
# SATELLITE STAGE TESTS
# ─────────────────────────────────────────────

class TestSatelliteStage:

    def test_returns_events_for_regions_with_change(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            result = run_satellite_stage(date(2021, 3, 1), resolution_m=60)
        assert result["events_created"] == len(MONITORED_REGIONS)
        assert all(isinstance(e, AnomalyEvent) for e in result["events"])

    def test_returns_zero_events_when_no_change(self):
        stable = forest_tile()
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair(before=stable, after=stable)):
            result = run_satellite_stage(date(2021, 3, 1))
        assert result["events_created"] == 0

    def test_returns_zero_events_when_no_imagery(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            result = run_satellite_stage(date(2021, 3, 1))
        assert result["events_created"] == 0

    def test_region_filter_respected(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            result = run_satellite_stage(
                date(2021, 3, 1),
                region_ids=["eth_tigray", "ukr_mariupol"],
            )
        assert result["events_created"] == 2

    def test_region_results_dict_covers_all_regions(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            result = run_satellite_stage(date(2021, 3, 1))
        for region in MONITORED_REGIONS:
            assert region.region_id in result["region_results"]

    def test_one_region_error_doesnt_stop_others(self):
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated API failure")
            return mock_tile_pair()

        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   side_effect=side_effect):
            result = run_satellite_stage(date(2021, 3, 1))

        assert result["errors"] == 1
        # Other regions still processed
        assert result["events_created"] == len(MONITORED_REGIONS) - 1

    def test_error_regions_recorded_in_results(self):
        def boom(*args, **kwargs):
            raise ConnectionError("timeout")

        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   side_effect=boom):
            result = run_satellite_stage(date(2021, 3, 1))

        for v in result["region_results"].values():
            assert v.startswith("error:")

    def test_events_have_correct_source(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            result = run_satellite_stage(date(2021, 3, 1))
        for event in result["events"]:
            assert event.source == AnomalySource.SATELLITE

    def test_events_intensity_scores_in_range(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            result = run_satellite_stage(date(2021, 3, 1))
        for event in result["events"]:
            assert 0.0 <= event.intensity_score <= 1.0

    def test_events_timestamps_match_target_date(self):
        target = date(2021, 3, 15)
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            result = run_satellite_stage(target)
        for event in result["events"]:
            assert event.timestamp.date() == target

    def test_dry_run_returns_events_without_persisting(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()), \
             patch("repository.save_anomaly_events_batch") as mock_save:
            result = run_satellite_stage(date(2021, 3, 1), dry_run=True)
        mock_save.assert_not_called()
        assert result["events_created"] == len(MONITORED_REGIONS)


# ─────────────────────────────────────────────
# FULL PIPELINE TESTS
# ─────────────────────────────────────────────

class TestRunPipeline:

    def test_returns_summary_dict(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        assert "status"        in summary
        assert "total_events"  in summary
        assert "target_date"   in summary
        assert "stage_results" in summary
        assert "elapsed_sec"   in summary

    def test_status_completed_on_success(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        assert summary["status"] == "COMPLETED"

    def test_defaults_to_yesterday(self):
        from datetime import date as _date
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(dry_run=True, use_db=False)
        yesterday = (_date.today() - timedelta(days=1)).isoformat()
        assert summary["target_date"] == yesterday

    def test_satellite_stage_in_results(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        assert "satellite" in summary["stage_results"]

    def test_pending_stages_present(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        assert summary["stage_results"]["gdelt"]["status"]       == "pending"
        assert summary["stage_results"]["procurement"]["status"] == "pending"
        assert summary["stage_results"]["agent"]["status"]       == "pending"

    def test_total_events_correct(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        assert summary["total_events"] == len(MONITORED_REGIONS)

    def test_region_filter_propagates(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            summary = run_pipeline(
                target_date=date(2021, 3, 1),
                region_ids=["eth_tigray"],
                dry_run=True, use_db=False,
            )
        assert summary["total_events"] == 1

    def test_db_persist_called_when_use_db_true(self):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)

        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()), \
             patch("db.get_db", return_value=mock_conn), \
             patch("repository.log_pipeline_run", return_value="test-run-id"), \
             patch("repository.save_anomaly_events_batch", return_value=5) as mock_save, \
             patch("repository.update_pipeline_run"):
            summary = run_pipeline(target_date=date(2021, 3, 1), use_db=True)

        mock_save.assert_called_once()

    def test_no_crash_when_db_unavailable(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None), \
             patch("db.get_db", side_effect=Exception("DB down")):
            # Should not raise — falls back to no-persist mode
            summary = run_pipeline(target_date=date(2021, 3, 1), use_db=True)
        assert summary["status"] in ("COMPLETED", "PARTIAL", "FAILED")

    def test_elapsed_sec_is_positive(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        assert summary["elapsed_sec"] >= 0


# ─────────────────────────────────────────────
# BACKFILL TESTS
# ─────────────────────────────────────────────

class TestRunBackfill:

    def test_runs_correct_number_of_days(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            results = run_backfill(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 1, 5),   # 5 days inclusive
                dry_run=True,
            )
        assert len(results) == 5

    def test_single_day_backfill(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            results = run_backfill(
                start_date=date(2021, 3, 15),
                end_date=date(2021, 3, 15),
                dry_run=True,
            )
        assert len(results) == 1

    def test_dates_are_sequential(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            results = run_backfill(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 1, 3),
                dry_run=True,
            )
        dates = [r["target_date"] for r in results]
        assert dates == ["2021-01-01", "2021-01-02", "2021-01-03"]

    def test_all_results_have_status(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            results = run_backfill(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 1, 3),
                dry_run=True,
            )
        for r in results:
            assert "status" in r

    def test_backfill_accumulates_events(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            results = run_backfill(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 1, 3),
                region_ids=["eth_tigray"],
                dry_run=True,
            )
        total = sum(r["total_events"] for r in results)
        assert total == 3   # 1 region × 3 days


# ─────────────────────────────────────────────
# REPOSITORY LAYER TESTS (no real DB needed)
# ─────────────────────────────────────────────

class TestRepository:

    def _make_event(self):
        return AnomalyEvent.make_satellite_event(
            region_id="eth_tigray", country_code="ET",
            lat=13.5, lng=39.0,
            timestamp=datetime(2021, 3, 15, tzinfo=timezone.utc),
            signal_type=SignalType.VEGETATION_LOSS,
            intensity_score=0.72,
            raw_data={"composite_score": 0.72},
        )

    def test_save_event_executes_insert(self):
        from repository import save_anomaly_event
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)

        event_id = save_anomaly_event(self._make_event(), mock_conn)
        assert event_id is not None
        mock_cursor.execute.assert_called_once()

    def test_save_batch_returns_count(self):
        from repository import save_anomaly_events_batch
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)

        events = [self._make_event() for _ in range(3)]
        count = save_anomaly_events_batch(events, mock_conn)
        assert count == 3

    def test_save_empty_batch_returns_zero(self):
        from repository import save_anomaly_events_batch
        mock_conn = MagicMock()
        count = save_anomaly_events_batch([], mock_conn)
        assert count == 0

    def test_log_pipeline_run_returns_uuid_string(self):
        from repository import log_pipeline_run
        import uuid
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)

        run_id = log_pipeline_run(mock_conn)
        assert run_id is not None
        uuid.UUID(run_id)   # raises if not a valid UUID


# ─────────────────────────────────────────────
# DIAGNOSTIC REPORT
# ─────────────────────────────────────────────

class TestDiagnosticReport:
    """Verifies the diagnostic summary output format expected in the spec."""

    def test_region_results_printable_summary(self):
        """
        Spec requires: diagnostic report prints region × source × anomaly_count.
        Verify the stage result dict supports building that table.
        """
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=mock_tile_pair()):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)

        sat = summary["stage_results"]["satellite"]
        assert "region_results" in sat
        # Every monitored region should appear
        for region in MONITORED_REGIONS:
            assert region.region_id in sat["region_results"]

    def test_summary_is_json_serialisable(self):
        with patch("detection.change_detection.fetch_tile_pair_for_region",
                   return_value=None):
            summary = run_pipeline(target_date=date(2021, 3, 1),
                                   dry_run=True, use_db=False)
        # Should not raise
        json.dumps(summary, default=str)