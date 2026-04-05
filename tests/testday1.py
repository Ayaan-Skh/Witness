"""
tests/test_day1.py — Day 1 Validation Tests

Run with: pytest tests/test_day1.py -v

These tests validate that:
1. All monitored regions have valid, non-empty bounding boxes
2. Bounding box coordinates are in the valid geographic range
3. Region centroids are computed correctly
4. All required config constants are present and within valid ranges
5. Database can be reached (if DATABASE_URL is valid)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import (
    MONITORED_REGIONS,
    REGIONS_BY_ID,
    SATELLITE_CHANGE_THRESHOLD,
    GDELT_ZSCORE_THRESHOLD,
    PROCUREMENT_ZSCORE_THRESHOLD,
    CONVERGENCE_SCORE_THRESHOLD,
    GEOGRAPHIC_CLUSTER_RADIUS_KM,
    TEMPORAL_CLUSTER_WINDOW_DAYS,
    LLM_MODEL,
    MonitoredRegion,
)


# ─────────────────────────────────────────────
# REGION VALIDATION TESTS
# ─────────────────────────────────────────────

class TestMonitoredRegions:

    def test_regions_list_is_not_empty(self):
        """There must be at least one region to monitor."""
        assert len(MONITORED_REGIONS) > 0, "MONITORED_REGIONS cannot be empty"

    def test_all_regions_have_valid_bbox(self):
        """
        A valid bounding box has:
        - Exactly 4 values: [min_lng, min_lat, max_lng, max_lat]
        - min_lng < max_lng (left edge is left of right edge)
        - min_lat < max_lat (bottom edge is below top edge)
        - Longitude in range [-180, 180]
        - Latitude in range [-90, 90]
        """
        for region in MONITORED_REGIONS:
            bbox = region.bbox
            assert len(bbox) == 4, f"{region.region_id}: bbox must have exactly 4 values"

            min_lng, min_lat, max_lng, max_lat = bbox
            assert min_lng < max_lng, f"{region.region_id}: min_lng must be less than max_lng"
            assert min_lat < max_lat, f"{region.region_id}: min_lat must be less than max_lat"
            assert -180 <= min_lng <= 180, f"{region.region_id}: min_lng out of range"
            assert -180 <= max_lng <= 180, f"{region.region_id}: max_lng out of range"
            assert -90 <= min_lat <= 90, f"{region.region_id}: min_lat out of range"
            assert -90 <= max_lat <= 90, f"{region.region_id}: max_lat out of range"

    def test_all_regions_have_valid_country_codes(self):
        """Country codes must be 2-character uppercase strings (ISO 3166-1 alpha-2)."""
        for region in MONITORED_REGIONS:
            code = region.country_code
            assert len(code) == 2, f"{region.region_id}: country_code must be 2 chars"
            assert code.isupper(), f"{region.region_id}: country_code must be uppercase"

    def test_all_region_ids_are_unique(self):
        """No two regions can have the same region_id."""
        ids = [r.region_id for r in MONITORED_REGIONS]
        assert len(ids) == len(set(ids)), "Duplicate region_ids found"

    def test_regions_by_id_lookup(self):
        """REGIONS_BY_ID dict must contain all regions."""
        for region in MONITORED_REGIONS:
            assert region.region_id in REGIONS_BY_ID
            assert REGIONS_BY_ID[region.region_id] is region

    def test_region_centroid_is_inside_bbox(self):
        """The computed centroid must fall within the bounding box."""
        for region in MONITORED_REGIONS:
            lat, lng = region.centroid()
            min_lng, min_lat, max_lng, max_lat = region.bbox
            assert min_lat <= lat <= max_lat, f"{region.region_id}: centroid lat outside bbox"
            assert min_lng <= lng <= max_lng, f"{region.region_id}: centroid lng outside bbox"

    def test_region_area_is_positive(self):
        """Every region must have a positive area (non-degenerate bounding box)."""
        for region in MONITORED_REGIONS:
            area = region.area_km2()
            assert area > 0, f"{region.region_id}: area must be positive"

    def test_specific_known_regions_exist(self):
        """Spot-check that our core calibration regions are present."""
        required_ids = {"eth_tigray", "chn_xinjiang", "ukr_mariupol"}
        actual_ids = {r.region_id for r in MONITORED_REGIONS}
        missing = required_ids - actual_ids
        assert not missing, f"Required regions missing from config: {missing}"


# ─────────────────────────────────────────────
# THRESHOLD VALIDATION TESTS
# ─────────────────────────────────────────────

class TestThresholds:

    def test_satellite_threshold_in_range(self):
        """Change scores are 0–1, so threshold must be in (0, 1)."""
        assert 0 < SATELLITE_CHANGE_THRESHOLD < 1

    def test_gdelt_zscore_threshold_is_positive(self):
        """Z-score thresholds should be positive (we compare abs value)."""
        assert GDELT_ZSCORE_THRESHOLD > 0

    def test_procurement_threshold_stricter_than_gdelt(self):
        """
        Procurement data is noisier, so we require a higher z-score.
        This documents an intentional design decision.
        """
        assert PROCUREMENT_ZSCORE_THRESHOLD >= GDELT_ZSCORE_THRESHOLD

    def test_convergence_threshold_between_zero_and_one(self):
        assert 0 < CONVERGENCE_SCORE_THRESHOLD < 1

    def test_cluster_radius_is_reasonable(self):
        """150km is reasonable for region-level clustering. Guard against typos."""
        assert 50 <= GEOGRAPHIC_CLUSTER_RADIUS_KM <= 500

    def test_cluster_time_window_is_reasonable(self):
        """21-day window. Guard against off-by-10x errors."""
        assert 7 <= TEMPORAL_CLUSTER_WINDOW_DAYS <= 90

    def test_llm_model_is_set(self):
        assert LLM_MODEL, "LLM_MODEL must not be empty"
        assert any(p in LLM_MODEL.lower() for p in ("gpt", "claude", "gemini"))


# ─────────────────────────────────────────────
# MONITORED REGION DATACLASS TESTS
# ─────────────────────────────────────────────

class TestMonitoredRegionDataclass:

    def test_centroid_simple(self):
        """Test centroid calculation on a simple box."""
        region = MonitoredRegion(
            region_id="test",
            name="Test",
            country_code="TS",
            admin1=None,
            bbox=[10.0, 20.0, 30.0, 40.0],
        )
        lat, lng = region.centroid()
        assert lat == pytest.approx(30.0)   # (20 + 40) / 2
        assert lng == pytest.approx(20.0)   # (10 + 30) / 2

    def test_buyer_ids_defaults_to_empty_list(self):
        """buyer_ids should default to empty list, not None."""
        region = MonitoredRegion(
            region_id="test2",
            name="Test",
            country_code="TS",
            admin1=None,
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        assert region.buyer_ids == []