"""
tests/test_day11_12.py — Procurement Pipeline Tests

Covers:
  1. Contract categorization (keyword matching)
  2. Amount extraction and currency conversion
  3. Date extraction from varied OCDS formats
  4. Spend timeseries aggregation
  5. Rolling baseline computation
  6. Spend spike detection
  7. New vendor detection
  8. Emergency contract detection
  9. run_procurement_detection orchestration
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from ingestion.procurement import (
    categorize_contract,
    extract_amount_usd,
    extract_award_date,
    extract_vendor_name,
    get_spend_timeseries,
    get_new_vendors,
    CATEGORY_KEYWORDS,
    SENSITIVE_CATEGORIES,
)
from detection.procurement_anomaly import (
    compute_rolling_baseline,
    detect_spend_spike,
    detect_new_vendor_pattern,
    detect_emergency_contracts,
    run_procurement_detection,
    _zscore_to_intensity,
)

from normalization.schema import AnomalyEvent, AnomalySource, SignalType


# ─────────────────────────────────────────────
# CONTRACT BUILDERS
# ─────────────────────────────────────────────

def make_contract(
    title: str = "Supply of goods",
    amount: float = 100_000.0,
    currency: str = "USD",
    award_date: str = "2021-03-15",
    vendor: str = "ACME Corp",
    buyer: str = "Ministry of Finance",
    proc_method: str = "open",
    contract_id: str = "ocds-test-001",
) -> dict:
    return {
        "ocid":        contract_id,
        "title":       title,
        "description": title,
        "value":       {"amount": amount, "currency": currency},
        "date":        award_date,
        "awards": [{"suppliers": [{"name": vendor}]}],
        "buyer":  {"name": buyer},
        "tender": {
            "title":             title,
            "procurementMethod": proc_method,
            "value":             {"amount": amount, "currency": currency},
        },
        "parties": [],
    }


def make_record(
    category: str = "MILITARY",
    amount_usd: float = 500_000.0,
    award_date=None,
    vendor: str = "Defense Supplier A",
    contract_id: str = "ocds-001",
) -> dict:
    """Pre-normalized record (output of _normalize_ocds_record)."""
    if award_date is None:
        award_date = date(2021, 3, 1)
    if isinstance(award_date, str):
        award_date = date.fromisoformat(award_date)
    return {
        "contract_id": contract_id,
        "title":       f"{category} supplies",
        "description": "",
        "category":    category,
        "amount_usd":  amount_usd,
        "currency":    "USD",
        "award_date":  award_date,
        "vendor_name": vendor,
        "buyer_name":  "Ministry",
        "country_code": "ET",
        "raw":         {},
    }


# ─────────────────────────────────────────────
# CONTRACT CATEGORISATION TESTS
# ─────────────────────────────────────────────

class TestCategorizeContract:

    def test_ammunition_is_military(self):
        c = make_contract(title="Supply of 5.56mm ammunition for armed forces")
        assert categorize_contract(c) == "MILITARY"

    def test_body_bags_are_medical(self):
        c = make_contract(title="Procurement of body bags and mortuary supplies")
        assert categorize_contract(c) == "MEDICAL"

    def test_field_hospital_is_medical(self):
        c = make_contract(title="Establishment of field hospital units")
        assert categorize_contract(c) == "MEDICAL"

    def test_fuel_is_logistics(self):
        c = make_contract(title="Diesel fuel supply for government fleet")
        assert categorize_contract(c) == "LOGISTICS"

    def test_fencing_is_construction(self):
        c = make_contract(title="Installation of perimeter fencing and watchtower")
        assert categorize_contract(c) == "CONSTRUCTION"

    def test_radio_is_communications(self):
        c = make_contract(title="VHF radio communication equipment for civilian use")
        assert categorize_contract(c) == "COMMUNICATIONS"

    def test_unknown_is_other(self):
        c = make_contract(title="Office stationery and paper supplies")
        assert categorize_contract(c) == "OTHER"

    def test_case_insensitive(self):
        c = make_contract(title="AMMUNITION AND WEAPONS PROCUREMENT")
        assert categorize_contract(c) == "MILITARY"

    def test_military_takes_priority_over_logistics(self):
        # "military vehicle" should be MILITARY not LOGISTICS
        c = make_contract(title="Military vehicle fuel and maintenance")
        assert categorize_contract(c) == "MILITARY"

    def test_description_field_used(self):
        c = make_contract(title="Goods and services")
        c["description"] = "Supply of surgical instruments and blood products"
        assert categorize_contract(c) == "MEDICAL"

    def test_empty_contract_is_other(self):
        assert categorize_contract({}) == "OTHER"

    def test_all_sensitive_categories_have_keywords(self):
        for cat in SENSITIVE_CATEGORIES:
            assert cat in CATEGORY_KEYWORDS
            assert len(CATEGORY_KEYWORDS[cat]) > 0


# ─────────────────────────────────────────────
# AMOUNT EXTRACTION TESTS
# ─────────────────────────────────────────────

class TestExtractAmountUSD:

    def test_usd_unchanged(self):
        c = make_contract(amount=1_000_000.0, currency="USD")
        assert extract_amount_usd(c) == pytest.approx(1_000_000.0)

    def test_eur_converted(self):
        c = make_contract(amount=1_000_000.0, currency="EUR")
        result = extract_amount_usd(c)
        assert result > 1_000_000.0   # EUR > USD

    def test_uah_converted(self):
        c = make_contract(amount=1_000_000.0, currency="UAH")
        result = extract_amount_usd(c)
        assert result < 100_000.0     # UAH is much weaker

    def test_unknown_currency_treated_as_usd(self):
        c = make_contract(amount=500.0, currency="XYZ")
        assert extract_amount_usd(c) == pytest.approx(500.0)

    def test_none_amount_returns_none(self):
        c = make_contract(amount=None)
        c["value"]["amount"] = None
        assert extract_amount_usd(c) is None

    def test_zero_amount_returns_zero(self):
        c = make_contract(amount=0.0)
        assert extract_amount_usd(c) == 0.0

    def test_string_amount_converted(self):
        c = make_contract(amount="250000")
        c["value"]["amount"] = "250000"
        assert extract_amount_usd(c) == pytest.approx(250_000.0)


# ─────────────────────────────────────────────
# DATE EXTRACTION TESTS
# ─────────────────────────────────────────────

class TestExtractAwardDate:

    def test_iso_date_string(self):
        c = make_contract(award_date="2021-03-15")
        assert extract_award_date(c) == date(2021, 3, 15)

    def test_iso_datetime_string(self):
        c = {"date": "2021-03-15T14:22:00+00:00"}
        assert extract_award_date(c) == date(2021, 3, 15)

    def test_missing_date_returns_none(self):
        assert extract_award_date({}) is None

    def test_tender_period_fallback(self):
        c = {"tender": {"tenderPeriod": {"startDate": "2021-06-01T00:00:00Z"}}}
        result = extract_award_date(c)
        assert result == date(2021, 6, 1)


# ─────────────────────────────────────────────
# SPEND TIMESERIES TESTS
# ─────────────────────────────────────────────

class TestGetSpendTimeseries:

    def test_groups_by_month(self):
        records = [
            make_record("MILITARY", 100_000, date(2021, 1, 5)),
            make_record("MILITARY", 200_000, date(2021, 1, 20)),
            make_record("MILITARY", 300_000, date(2021, 2, 10)),
        ]
        series = get_spend_timeseries(records, "MILITARY")
        assert len(series) == 2
        jan = next(r for r in series if r["period_label"] == "2021-01")
        assert jan["total_usd"] == pytest.approx(300_000.0)
        assert jan["contract_count"] == 2

    def test_filters_by_category(self):
        records = [
            make_record("MILITARY", 500_000, date(2021, 3, 1)),
            make_record("MEDICAL",  200_000, date(2021, 3, 1)),
        ]
        series = get_spend_timeseries(records, "MILITARY")
        assert len(series) == 1
        assert series[0]["total_usd"] == pytest.approx(500_000.0)

    def test_empty_records_returns_empty(self):
        assert get_spend_timeseries([], "MILITARY") == []

    def test_no_matching_category_returns_empty(self):
        records = [make_record("MEDICAL", 100_000, date(2021, 3, 1))]
        assert get_spend_timeseries(records, "MILITARY") == []

    def test_sorted_by_period(self):
        records = [
            make_record("MILITARY", 100_000, date(2021, 3, 1)),
            make_record("MILITARY", 100_000, date(2021, 1, 1)),
            make_record("MILITARY", 100_000, date(2021, 2, 1)),
        ]
        series = get_spend_timeseries(records, "MILITARY")
        labels = [r["period_label"] for r in series]
        assert labels == sorted(labels)


# ─────────────────────────────────────────────
# ROLLING BASELINE TESTS
# ─────────────────────────────────────────────

class TestComputeRollingBaseline:

    def _make_series(self, amounts: list[float], current_label: str = "2021-04") -> list[dict]:
        series = []
        for i, amt in enumerate(amounts):
            month = i + 1
            label = f"2021-{month:02d}"
            series.append({"period_label": label, "total_usd": amt,
                            "period_start": date(2021, month, 1), "contract_count": 1})
        return series

    def test_excludes_current_period(self):
        series = self._make_series([100, 200, 300, 9999], "2021-04")
        mean, std, n = compute_rolling_baseline(series, "2021-04")
        assert mean == pytest.approx(200.0)  # mean of [100, 200, 300]
        assert n == 3

    def test_empty_baseline_safe(self):
        series = [{"period_label": "2021-01", "total_usd": 500.0,
                   "period_start": date(2021, 1, 1), "contract_count": 1}]
        mean, std, n = compute_rolling_baseline(series, "2021-01")
        assert mean == 0.0
        assert n == 0

    def test_single_baseline_point(self):
        series = [
            {"period_label": "2021-01", "total_usd": 500.0,
             "period_start": date(2021, 1, 1), "contract_count": 1},
            {"period_label": "2021-02", "total_usd": 9999.0,
             "period_start": date(2021, 2, 1), "contract_count": 1},
        ]
        mean, std, n = compute_rolling_baseline(series, "2021-02")
        assert mean == pytest.approx(500.0)
        assert n == 1

    def test_std_floor_prevents_zero(self):
        series = [
            {"period_label": "2021-01", "total_usd": 1000.0,
             "period_start": date(2021, 1, 1), "contract_count": 1},
            {"period_label": "2021-02", "total_usd": 1000.0,
             "period_start": date(2021, 2, 1), "contract_count": 1},
            {"period_label": "2021-03", "total_usd": 1000.0,
             "period_start": date(2021, 3, 1), "contract_count": 1},
            {"period_label": "2021-04", "total_usd": 999.0,
             "period_start": date(2021, 4, 1), "contract_count": 1},
        ]
        _, std, _ = compute_rolling_baseline(series, "2021-04")
        assert std > 0


# ─────────────────────────────────────────────
# SPEND SPIKE DETECTION TESTS
# ─────────────────────────────────────────────

class TestDetectSpendSpike:

    def _stable_series(self, spike_month: str = None, spike_amount: float = 0.0):
        series = []
        for m in range(1, 13):
            label  = f"2021-{m:02d}"
            amount = spike_amount if (spike_month and label == spike_month) else 500_000.0
            series.append({"period_label": label, "total_usd": amount,
                            "period_start": date(2021, m, 1), "contract_count": 1})
        return series

    def test_no_spike_gives_low_zscore(self):
        series = self._stable_series()
        _, _, _, zscore = detect_spend_spike(series, "2021-06")
        assert abs(zscore) < 1.0

    def test_spike_gives_high_zscore(self):
        # 10× normal spend in target month
        series = self._stable_series(spike_month="2021-11", spike_amount=5_000_000.0)
        _, _, _, zscore = detect_spend_spike(series, "2021-11")
        assert zscore > 2.5, f"Spike should give z>2.5, got {zscore:.2f}"

    def test_missing_current_period_gives_zero(self):
        series = self._stable_series()
        spend, mean, std, z = detect_spend_spike(series, "2021-13")
        assert spend == 0.0

    def test_returns_four_tuple(self):
        series = self._stable_series()
        result = detect_spend_spike(series, "2021-06")
        assert len(result) == 4


# ─────────────────────────────────────────────
# NEW VENDOR DETECTION TESTS
# ─────────────────────────────────────────────

class TestDetectNewVendors:

    def test_new_vendor_detected(self):
        baseline = [make_record("MILITARY", 100_000, vendor="Old Vendor")]
        current  = [make_record("MILITARY", 200_000, vendor="Brand New Supplier")]
        new = detect_new_vendor_pattern(current, baseline, "MILITARY")
        assert len(new) == 1
        assert new[0]["vendor_name"] == "Brand New Supplier"

    def test_existing_vendor_not_flagged(self):
        baseline = [make_record("MILITARY", 100_000, vendor="Trusted Vendor")]
        current  = [make_record("MILITARY", 500_000, vendor="Trusted Vendor")]
        new = detect_new_vendor_pattern(current, baseline, "MILITARY")
        assert len(new) == 0

    def test_non_sensitive_category_ignored(self):
        baseline = []
        current  = [make_record("OTHER", 500_000, vendor="New Vendor")]
        new = detect_new_vendor_pattern(current, baseline, "OTHER")
        assert len(new) == 0

    def test_empty_baseline_all_vendors_are_new(self):
        current = [
            make_record("MILITARY", 100_000, vendor="Vendor A"),
            make_record("MILITARY", 200_000, vendor="Vendor B"),
        ]
        new = detect_new_vendor_pattern(current, [], "MILITARY")
        assert len(new) == 2


# ─────────────────────────────────────────────
# EMERGENCY CONTRACT DETECTION TESTS
# ─────────────────────────────────────────────

class TestDetectEmergencyContracts:

    def _make_emergency_record(self, method: str, category: str = "MILITARY") -> dict:
        r = make_record(category, 300_000)
        r["raw"] = {"tender": {"procurementMethod": method}}
        return r

    def test_direct_procurement_flagged(self):
        records = [self._make_emergency_record("direct")]
        flagged = detect_emergency_contracts(records, "MILITARY")
        assert len(flagged) == 1

    def test_limited_procurement_flagged(self):
        records = [self._make_emergency_record("limited")]
        flagged = detect_emergency_contracts(records, "MILITARY")
        assert len(flagged) == 1

    def test_open_procurement_not_flagged(self):
        records = [self._make_emergency_record("open")]
        flagged = detect_emergency_contracts(records, "MILITARY")
        assert len(flagged) == 0

    def test_emergency_rationale_flagged(self):
        r = make_record("MILITARY", 300_000)
        r["raw"] = {"tender": {"procurementMethod": "open",
                                "procurementMethodRationale": "Emergency security situation"}}
        flagged = detect_emergency_contracts([r], "MILITARY")
        assert len(flagged) == 1

    def test_wrong_category_not_flagged(self):
        records = [self._make_emergency_record("direct", category="OTHER")]
        flagged = detect_emergency_contracts(records, "MILITARY")
        assert len(flagged) == 0


# ─────────────────────────────────────────────
# INTENSITY NORMALISATION TESTS
# ─────────────────────────────────────────────

class TestIntensity:

    def test_at_threshold_is_zero(self):
        assert _zscore_to_intensity(2.5, threshold=2.5) == pytest.approx(0.0)

    def test_high_z_gives_high_intensity(self):
        assert _zscore_to_intensity(6.0, threshold=2.5) >= 0.9  # approaches 1.0 at high z

    def test_output_in_range(self):
        for z in [0, 1, 2, 2.5, 3, 5, 10]:
            result = _zscore_to_intensity(float(z))
            assert 0.0 <= result <= 1.0


# ─────────────────────────────────────────────
# RUN_PROCUREMENT_DETECTION ORCHESTRATION TESTS
# ─────────────────────────────────────────────

class TestRunProcurementDetection:

    def _stable_records(self, months: int = 13) -> list[dict]:
        """Generate 13 months of stable MILITARY spend at $500K/month."""
        records = []
        base = date(2020, 3, 1)
        for i in range(months):
            d = date(base.year + (base.month + i - 1) // 12,
                     (base.month + i - 1) % 12 + 1, 1)
            records.append(make_record("MILITARY", 500_000, d,
                                       contract_id=f"c{i}"))
        return records

    def _spike_records(self, spike_month: date, spike_amount: float = 5_000_000.0) -> list[dict]:
        records = self._stable_records(12)
        records.append(make_record("MILITARY", spike_amount, spike_month,
                                   contract_id="spike"))
        return records

    def test_no_anomaly_returns_none(self):
        all_recs = self._stable_records(13)
        baseline = [r for r in all_recs if r["award_date"].month != 3
                    or r["award_date"].year != 2021]
        current  = [r for r in all_recs if r["award_date"] == date(2021, 3, 1)]
        result = run_procurement_detection(
            "eth_tigray", date(2021, 3, 1),
            current_records=current,
            baseline_records=baseline,
        )
        assert result is None

    def test_spend_spike_produces_event(self):
        target   = date(2021, 3, 1)
        all_recs = self._spike_records(target, spike_amount=5_000_000.0)
        baseline = [r for r in all_recs if not (
            r["award_date"].year == target.year and
            r["award_date"].month == target.month
        )]
        current = [r for r in all_recs if (
            r["award_date"].year == target.year and
            r["award_date"].month == target.month
        )]
        result = run_procurement_detection(
            "eth_tigray", target,
            current_records=current,
            baseline_records=baseline,
        )
        assert result is not None
        assert isinstance(result, AnomalyEvent)

    def test_event_signal_type_is_spend_spike(self):
        target   = date(2021, 3, 1)
        all_recs = self._spike_records(target, 5_000_000.0)
        baseline = [r for r in all_recs if not (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        current  = [r for r in all_recs if (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        result = run_procurement_detection("eth_tigray", target,
                                           current_records=current,
                                           baseline_records=baseline)
        assert result.signal_type == SignalType.SPEND_SPIKE

    def test_event_source_is_procurement(self):
        target   = date(2021, 3, 1)
        all_recs = self._spike_records(target, 5_000_000.0)
        baseline = [r for r in all_recs if not (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        current  = [r for r in all_recs if (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        result = run_procurement_detection("eth_tigray", target,
                                           current_records=current,
                                           baseline_records=baseline)
        assert result.source == AnomalySource.PROCUREMENT

    def test_intensity_in_range(self):
        target   = date(2021, 3, 1)
        all_recs = self._spike_records(target, 5_000_000.0)
        baseline = [r for r in all_recs if not (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        current  = [r for r in all_recs if (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        result = run_procurement_detection("eth_tigray", target,
                                           current_records=current,
                                           baseline_records=baseline)
        assert 0.0 <= result.intensity_score <= 1.0

    def test_raw_data_has_categories(self):
        target   = date(2021, 3, 1)
        all_recs = self._spike_records(target, 5_000_000.0)
        baseline = [r for r in all_recs if not (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        current  = [r for r in all_recs if (
            r["award_date"].year == target.year and r["award_date"].month == target.month)]
        result = run_procurement_detection("eth_tigray", target,
                                           current_records=current,
                                           baseline_records=baseline)
        assert "categories"     in result.raw_data
        assert "current_month"  in result.raw_data
        assert "dominant_zscore" in result.raw_data

    def test_new_vendor_triggers_event_when_no_spike(self):
        target   = date(2021, 3, 1)
        baseline = self._stable_records(12)
        current  = [make_record("MILITARY", 500_000, target,
                                vendor="Mysterious New Arms Dealer")]
        result = run_procurement_detection("eth_tigray", target,
                                           current_records=current,
                                           baseline_records=baseline)
        assert result is not None
        assert result.signal_type == SignalType.NEW_VENDOR_PATTERN

    def test_empty_records_returns_none(self):
        result = run_procurement_detection(
            "eth_tigray", date(2021, 3, 1),
            current_records=[], baseline_records=[],
        )
        assert result is None

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region_id"):
            run_procurement_detection("not_real", date(2021, 3, 1),
                                      current_records=[], baseline_records=[])

    def test_all_regions_run_without_crash(self):
        from config import MONITORED_REGIONS
        target = date(2021, 3, 1)
        for region in MONITORED_REGIONS:
            result = run_procurement_detection(
                region.region_id, target,
                current_records=[], baseline_records=[],
            )
            assert result is None   # empty records → no signal (no crash)