"""
tests/test_day2.py — Schema Validation Tests

Run with: pytest tests/test_day2.py -v

These tests validate every constraint defined in normalization/schema.py.
Organized into three groups:
  1. AnomalyEvent construction and validation
  2. InvestigationBrief construction and validation
  3. Serialization round-trips (to_dict/from_dict, to_json/from_json)

Why test serialization round-trips?
  A round-trip test proves that serialize(deserialize(x)) == x.
  This matters because AnomalyEvents travel through the system as dicts
  (database ↔ Python) and JSON (API responses, logs). A silent deserialization
  bug — e.g., losing timezone info or truncating a float — would corrupt data
  without raising any errors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta

from normalization.schema import (
    AnomalyEvent,
    AnomalySource,
    InvestigationBrief,
    BriefStatus,
    ConfidenceTier,
    SignalType,
)


# ─────────────────────────────────────────────
# FIXTURES
# Reusable "canonical" objects used across many tests.
# ─────────────────────────────────────────────

@pytest.fixture
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def satellite_event(utc_now) -> AnomalyEvent:
    """A valid satellite AnomalyEvent — the 'gold standard' for tests."""
    return AnomalyEvent.make_satellite_event(
        region_id="eth_tigray",
        country_code="ET",
        lat=14.12,
        lng=38.72,
        timestamp=utc_now,
        signal_type=SignalType.LAND_COVER_CHANGE,
        intensity_score=0.71,
        raw_data={"ndvi_before": 0.62, "ndvi_after": 0.31, "change_score": 0.71},
        metadata={"cloud_cover_pct": 5, "resolution_m": 10},
    )


@pytest.fixture
def gdelt_event(utc_now) -> AnomalyEvent:
    """A valid GDELT AnomalyEvent."""
    return AnomalyEvent.make_gdelt_event(
        region_id="eth_tigray",
        country_code="ET",
        lat=14.12,
        lng=38.72,
        timestamp=utc_now,
        signal_type=SignalType.TONE_CRASH,
        intensity_score=0.85,
        raw_data={"tone_zscore": -3.4, "volume_zscore": 2.1},
        metadata={"dominant_themes": ["193", "MILITARY"]},
    )


@pytest.fixture
def procurement_event(utc_now) -> AnomalyEvent:
    """A valid procurement AnomalyEvent."""
    return AnomalyEvent.make_procurement_event(
        region_id="eth_tigray",
        country_code="ET",
        lat=9.03,
        lng=38.74,
        timestamp=utc_now,
        signal_type=SignalType.SPEND_SPIKE,
        intensity_score=0.60,
        raw_data={"category": "MILITARY", "spend_zscore": 3.1, "amount_usd": 4200000},
    )


@pytest.fixture
def valid_brief(satellite_event, gdelt_event, procurement_event, utc_now) -> InvestigationBrief:
    """A valid 3-source InvestigationBrief."""
    return InvestigationBrief(
        brief_id=str(uuid.uuid4()),
        region_id="eth_tigray",
        time_window_start=utc_now - timedelta(days=14),
        time_window_end=utc_now,
        confidence_score=0.82,
        confidence_tier=ConfidenceTier.HIGH,
        contributing_streams=["SATELLITE", "GDELT", "PROCUREMENT"],
        evidence={
            "SATELLITE": {"summary": "Significant land cover change."},
            "GDELT": {"summary": "Tone collapsed to extreme negative."},
            "PROCUREMENT": {"summary": "Military spend spike 3.1 std devs above baseline."},
        },
        agent_reasoning="All three independent data streams converged on Tigray...",
        historical_context="Similar signal detected in this region in November 2020...",
    )


# ─────────────────────────────────────────────
# ANOMALY EVENT CONSTRUCTION TESTS
# ─────────────────────────────────────────────

class TestAnomalyEventConstruction:

    def test_factory_methods_set_correct_source(self, satellite_event, gdelt_event, procurement_event):
        assert satellite_event.source == AnomalySource.SATELLITE
        assert gdelt_event.source == AnomalySource.GDELT
        assert procurement_event.source == AnomalySource.PROCUREMENT

    def test_factory_methods_generate_uuid(self, satellite_event):
        """event_id must be a valid UUID string."""
        parsed = uuid.UUID(satellite_event.event_id)  # raises ValueError if invalid
        assert str(parsed) == satellite_event.event_id

    def test_different_events_have_unique_ids(self, satellite_event, gdelt_event):
        assert satellite_event.event_id != gdelt_event.event_id

    def test_raw_data_defaults_to_empty_dict(self, utc_now):
        """Omitting raw_data should give an empty dict, not None."""
        event = AnomalyEvent.make_satellite_event(
            region_id="eth_tigray", country_code="ET",
            lat=14.0, lng=38.0, timestamp=utc_now,
            signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=0.5,
        )
        assert event.raw_data == {}
        assert event.metadata == {}

    def test_detected_at_is_auto_set(self, satellite_event):
        """detected_at should be automatically set to a UTC datetime."""
        assert satellite_event.detected_at is not None
        assert satellite_event.detected_at.tzinfo is not None

    def test_repr_is_informative(self, satellite_event):
        r = repr(satellite_event)
        assert "SATELLITE" in r
        assert "eth_tigray" in r
        assert "LAND_COVER_CHANGE" in r


# ─────────────────────────────────────────────
# ANOMALY EVENT VALIDATION TESTS
# ─────────────────────────────────────────────

class TestAnomalyEventValidation:

    def test_rejects_empty_event_id(self, utc_now):
        with pytest.raises(ValueError, match="event_id"):
            AnomalyEvent(
                event_id="", source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=14.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=0.5,
            )

    def test_rejects_latitude_above_90(self, utc_now):
        with pytest.raises(ValueError, match="lat"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.GDELT,
                region_id="eth_tigray", country_code="ET",
                lat=91.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.TONE_CRASH, intensity_score=0.5,
            )

    def test_rejects_latitude_below_minus_90(self, utc_now):
        with pytest.raises(ValueError, match="lat"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.GDELT,
                region_id="eth_tigray", country_code="ET",
                lat=-91.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.TONE_CRASH, intensity_score=0.5,
            )

    def test_rejects_longitude_out_of_range(self, utc_now):
        with pytest.raises(ValueError, match="lng"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=14.0, lng=181.0, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=0.5,
            )

    def test_rejects_intensity_above_1(self, utc_now):
        with pytest.raises(ValueError, match="intensity_score"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=14.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=1.01,
            )

    def test_rejects_negative_intensity(self, utc_now):
        with pytest.raises(ValueError, match="intensity_score"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=14.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=-0.1,
            )

    def test_rejects_invalid_country_code_length(self, utc_now):
        with pytest.raises(ValueError, match="country_code"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ETH",  # 3 chars, not 2
                lat=14.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=0.5,
            )

    def test_rejects_naive_datetime(self, utc_now):
        """Timestamps without timezone info must be rejected."""
        naive_dt = datetime(2024, 3, 15, 12, 0, 0)  # No timezone!
        with pytest.raises(ValueError, match="timezone-aware"):
            AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=14.0, lng=38.0, timestamp=naive_dt,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=0.5,
            )

    def test_accepts_boundary_intensity_values(self, utc_now):
        """0.0 and 1.0 are both valid intensity scores."""
        for score in [0.0, 1.0]:
            event = AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=14.0, lng=38.0, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=score,
            )
            assert event.intensity_score == score

    def test_accepts_boundary_coordinates(self, utc_now):
        """Poles and antimeridian are valid coordinates."""
        for lat, lng in [(90, 0), (-90, 0), (0, 180), (0, -180)]:
            event = AnomalyEvent(
                event_id=str(uuid.uuid4()), source=AnomalySource.SATELLITE,
                region_id="eth_tigray", country_code="ET",
                lat=lat, lng=lng, timestamp=utc_now,
                signal_type=SignalType.LAND_COVER_CHANGE, intensity_score=0.5,
            )
            assert event.lat == lat


# ─────────────────────────────────────────────
# ANOMALY EVENT SERIALIZATION TESTS
# ─────────────────────────────────────────────

class TestAnomalyEventSerialization:

    def test_to_dict_source_is_string(self, satellite_event):
        """Enum values should serialize to plain strings."""
        d = satellite_event.to_dict()
        assert d["source"] == "SATELLITE"
        assert isinstance(d["source"], str)

    def test_to_dict_signal_type_is_string(self, satellite_event):
        d = satellite_event.to_dict()
        assert d["signal_type"] == "LAND_COVER_CHANGE"
        assert isinstance(d["signal_type"], str)

    def test_to_dict_timestamp_is_iso_string(self, satellite_event):
        d = satellite_event.to_dict()
        assert isinstance(d["timestamp"], str)
        # Should be parseable as a datetime
        parsed = datetime.fromisoformat(d["timestamp"])
        assert parsed.tzinfo is not None

    def test_round_trip_dict(self, satellite_event):
        """to_dict → from_dict should produce an equivalent object."""
        d = satellite_event.to_dict()
        restored = AnomalyEvent.from_dict(d)
        assert restored.event_id       == satellite_event.event_id
        assert restored.source         == satellite_event.source
        assert restored.region_id      == satellite_event.region_id
        assert restored.country_code   == satellite_event.country_code
        assert restored.lat            == satellite_event.lat
        assert restored.lng            == satellite_event.lng
        assert restored.signal_type    == satellite_event.signal_type
        assert restored.intensity_score == satellite_event.intensity_score
        assert restored.raw_data       == satellite_event.raw_data
        assert restored.metadata       == satellite_event.metadata

    def test_round_trip_json(self, gdelt_event):
        """to_json → from_json should produce an equivalent object."""
        json_str = gdelt_event.to_json()
        assert isinstance(json_str, str)
        restored = AnomalyEvent.from_json(json_str)
        assert restored.event_id == gdelt_event.event_id
        assert restored.source   == AnomalySource.GDELT

    def test_json_is_valid_json(self, procurement_event):
        """The JSON output must be parseable by the standard library."""
        json_str = procurement_event.to_json()
        parsed = json.loads(json_str)
        assert parsed["source"] == "PROCUREMENT"

    def test_round_trip_preserves_raw_data(self, satellite_event):
        """Nested dicts in raw_data must survive a round-trip unchanged."""
        restored = AnomalyEvent.from_dict(satellite_event.to_dict())
        assert restored.raw_data == satellite_event.raw_data

    def test_from_dict_handles_z_suffix_timestamp(self, satellite_event):
        """Some APIs return 'Z' instead of '+00:00' — we must handle both."""
        d = satellite_event.to_dict()
        d["timestamp"] = d["timestamp"].replace("+00:00", "Z")
        restored = AnomalyEvent.from_dict(d)
        assert restored.timestamp.tzinfo is not None


# ─────────────────────────────────────────────
# INVESTIGATION BRIEF CONSTRUCTION TESTS
# ─────────────────────────────────────────────

class TestInvestigationBriefConstruction:

    def test_valid_brief_constructs_without_error(self, valid_brief):
        assert valid_brief.brief_id is not None
        assert valid_brief.confidence_score == 0.82
        assert valid_brief.confidence_tier == ConfidenceTier.HIGH

    def test_default_status_is_draft(self, valid_brief):
        assert valid_brief.status == BriefStatus.DRAFT

    def test_stream_count_computed_correctly(self, valid_brief):
        assert valid_brief.stream_count == 3

    def test_is_multi_source_for_3_streams(self, valid_brief):
        assert valid_brief.is_multi_source is True

    def test_is_multi_source_false_for_single_stream(self, utc_now):
        brief = InvestigationBrief(
            brief_id=str(uuid.uuid4()),
            region_id="eth_tigray",
            time_window_start=utc_now - timedelta(days=7),
            time_window_end=utc_now,
            confidence_score=0.30,
            confidence_tier=ConfidenceTier.LOW,
            contributing_streams=["SATELLITE"],
        )
        assert brief.is_multi_source is False
        assert brief.stream_count == 1

    def test_duration_days(self, valid_brief):
        assert valid_brief.duration_days == 14

    def test_summary_line_contains_key_info(self, valid_brief):
        line = valid_brief.summary_line()
        assert "HIGH" in line
        assert "eth_tigray" in line
        assert "0.82" in line
        assert "DRAFT" in line


# ─────────────────────────────────────────────
# INVESTIGATION BRIEF VALIDATION TESTS
# ─────────────────────────────────────────────

class TestInvestigationBriefValidation:

    def test_rejects_empty_contributing_streams(self, utc_now):
        with pytest.raises(ValueError, match="contributing_streams"):
            InvestigationBrief(
                brief_id=str(uuid.uuid4()),
                region_id="eth_tigray",
                time_window_start=utc_now - timedelta(days=7),
                time_window_end=utc_now,
                confidence_score=0.5,
                confidence_tier=ConfidenceTier.MEDIUM,
                contributing_streams=[],  # empty!
            )

    def test_rejects_unknown_stream_name(self, utc_now):
        with pytest.raises(ValueError, match="unknown source"):
            InvestigationBrief(
                brief_id=str(uuid.uuid4()),
                region_id="eth_tigray",
                time_window_start=utc_now - timedelta(days=7),
                time_window_end=utc_now,
                confidence_score=0.5,
                confidence_tier=ConfidenceTier.MEDIUM,
                contributing_streams=["SATELLITE", "TWITTER"],  # TWITTER not valid
            )

    def test_rejects_invalid_confidence_score(self, utc_now):
        with pytest.raises(ValueError, match="confidence_score"):
            InvestigationBrief(
                brief_id=str(uuid.uuid4()),
                region_id="eth_tigray",
                time_window_start=utc_now - timedelta(days=7),
                time_window_end=utc_now,
                confidence_score=1.5,   # above 1.0
                confidence_tier=ConfidenceTier.HIGH,
                contributing_streams=["SATELLITE"],
            )

    def test_rejects_inverted_time_window(self, utc_now):
        """start must be before end."""
        with pytest.raises(ValueError, match="time_window_start must be before"):
            InvestigationBrief(
                brief_id=str(uuid.uuid4()),
                region_id="eth_tigray",
                time_window_start=utc_now,                      # start = now
                time_window_end=utc_now - timedelta(days=7),    # end = 7 days ago!
                confidence_score=0.5,
                confidence_tier=ConfidenceTier.MEDIUM,
                contributing_streams=["SATELLITE"],
            )

    def test_rejects_naive_time_window(self, utc_now):
        naive_dt = datetime(2024, 1, 1)  # no timezone
        with pytest.raises(ValueError, match="timezone-aware"):
            InvestigationBrief(
                brief_id=str(uuid.uuid4()),
                region_id="eth_tigray",
                time_window_start=naive_dt,
                time_window_end=utc_now,
                confidence_score=0.5,
                confidence_tier=ConfidenceTier.MEDIUM,
                contributing_streams=["SATELLITE"],
            )


# ─────────────────────────────────────────────
# INVESTIGATION BRIEF SERIALIZATION TESTS
# ─────────────────────────────────────────────

class TestInvestigationBriefSerialization:

    def test_round_trip_dict(self, valid_brief):
        d = valid_brief.to_dict()
        restored = InvestigationBrief.from_dict(d)
        assert restored.brief_id          == valid_brief.brief_id
        assert restored.region_id         == valid_brief.region_id
        assert restored.confidence_score  == valid_brief.confidence_score
        assert restored.confidence_tier   == valid_brief.confidence_tier
        assert restored.status            == valid_brief.status
        assert restored.contributing_streams == valid_brief.contributing_streams

    def test_round_trip_json(self, valid_brief):
        json_str = valid_brief.to_json()
        restored = InvestigationBrief.from_json(json_str)
        assert restored.brief_id == valid_brief.brief_id

    def test_to_dict_enums_are_strings(self, valid_brief):
        d = valid_brief.to_dict()
        assert d["confidence_tier"] == "HIGH"
        assert d["status"] == "DRAFT"
        assert isinstance(d["confidence_tier"], str)

    def test_to_dict_datetimes_are_strings(self, valid_brief):
        d = valid_brief.to_dict()
        assert isinstance(d["time_window_start"], str)
        assert isinstance(d["time_window_end"], str)

    def test_round_trip_preserves_evidence(self, valid_brief):
        restored = InvestigationBrief.from_dict(valid_brief.to_dict())
        assert restored.evidence == valid_brief.evidence


# ─────────────────────────────────────────────
# ENUM TESTS
# ─────────────────────────────────────────────

class TestEnums:

    def test_anomaly_source_string_values(self):
        assert AnomalySource.SATELLITE   == "SATELLITE"
        assert AnomalySource.GDELT       == "GDELT"
        assert AnomalySource.PROCUREMENT == "PROCUREMENT"

    def test_brief_status_progression(self):
        """Verify all three lifecycle stages exist."""
        assert BriefStatus.DRAFT     == "DRAFT"
        assert BriefStatus.REVIEWED  == "REVIEWED"
        assert BriefStatus.PUBLISHED == "PUBLISHED"

    def test_confidence_tier_values(self):
        assert ConfidenceTier.LOW    == "LOW"
        assert ConfidenceTier.MEDIUM == "MEDIUM"
        assert ConfidenceTier.HIGH   == "HIGH"

    def test_all_signal_types_have_string_values(self):
        """Every SignalType must serialize cleanly to a non-empty string."""
        for signal_type in SignalType:
            assert isinstance(signal_type.value, str)
            assert len(signal_type.value) > 0