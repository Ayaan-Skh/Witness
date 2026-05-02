"""
Procurement Anomaly Detection

WHAT THIS DETECTS ?
Statistically anomalous government spending patterns in categories that correlate with conflict preparation, active operations, or displacement:
  MILITARY, MEDICAL, LOGISTICS, CONSTRUCTION, COMMUNICATIONS.

Three signal types:
  1. SPEND_SPIKE       — current month spend >> rolling 12-month baseline
  2. NEW_VENDOR_PATTERN — new suppliers appearing in sensitive categories
  3. EMERGENCY_CONTRACT — direct-award / single-source contracts (bypassing
                          normal competitive tender process)

THRESHOLD: 2.5 std devs (vs 2.0 for GDELT)
Procurement data is noisier than GDELT. Budgets are released in lump sums, fiscal years create seasonal spikes, and data publication lag means we sometimes see 3 months of contracts posted at once. The higher threshold (2.5σ vs 2.0σ) reduces false positives from these artifacts.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

from config import (
    PROCUREMENT_BASELINE_MONTHS,
    PROCUREMENT_NEW_VENDOR_LOOKBACK,
    PROCUREMENT_ZSCORE_THRESHOLD,
    REGIONS_BY_ID,
)
from ingestion.procurement import (
    SENSITIVE_CATEGORIES,
    categorize_contract,
    fetch_ocds_records,
    get_new_vendors,
    get_spend_timeseries,
)
from normalization.schema import AnomalyEvent, AnomalySource, SignalType

log = logging.getLogger("witness.procurement_anomaly")


# ─────────────────────────────────────────────
# BASELINE COMPUTATION
# ─────────────────────────────────────────────

def compute_rolling_baseline(
    spend_series: list[dict],
    current_period_label: str,
) -> tuple[float, float, int]:
    """
    Computes rolling mean and std dev from historical spend periods, excluding the current period.

    Args:
        spend_series:          Monthly spend timeseries from get_spend_timeseries().
        current_period_label:  The period label to exclude (e.g. "2021-03").

    Returns:
        (baseline_mean, baseline_std, sample_count)

    Why exclude the current period?
        We're asking "is this month anomalous vs history?" If we include the current month in the baseline, a truly anomalous month would pull the mean up and reduce its own z-score — self-concealment.
    """
    baseline_amounts = [
        row["total_usd"]
        for row in spend_series
        if row["period_label"] != current_period_label
        and row["total_usd"] > 0
    ]

    if len(baseline_amounts) == 0:
        return 0.0, 1.0, 0

    arr  = np.array(baseline_amounts, dtype=np.float64)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else float(arr[0] * 0.3)

    if std < 1.0:
        std = max(mean * 0.1, 1.0)  # floor at 10% of mean to handle near-zero baselines

    return mean, std, len(baseline_amounts)


def _zscore_to_intensity(zscore: float, threshold: float = 2.5) -> float:
    """Maps |z-score| to [0, 1] intensity. Same logic as GDELT module."""
    abs_z    = abs(zscore)
    anchored = abs_z - threshold
    scaled   = anchored / (threshold * 1.5)
    return float(np.clip(scaled, 0.0, 1.0))


# ─────────────────────────────────────────────
# SIGNAL DETECTORS
# ─────────────────────────────────────────────

def detect_spend_spike(
    spend_series: list[dict],
    current_period_label: str,
) -> tuple[float, float, float, float]:
    """
    Detects whether current-period spend is anomalously high vs baseline.

    Returns:
        (current_spend, baseline_mean, baseline_std, zscore)
    """
    current_rows = [r for r in spend_series if r["period_label"] == current_period_label]
    current_spend = current_rows[0]["total_usd"] if current_rows else 0.0

    mean, std, n = compute_rolling_baseline(spend_series, current_period_label)
    zscore = (current_spend - mean) / std if std > 0 else 0.0

    return current_spend, mean, std, zscore


def detect_new_vendor_pattern(
    current_records: list[dict],
    baseline_records: list[dict],
    category: str,
) -> list[dict]:
    """
    Returns contracts in current_records from vendors not present in
    baseline_records for the given category.
    Only checks SENSITIVE_CATEGORIES — new food vendors aren't a signal.
    """
    if category not in SENSITIVE_CATEGORIES:
        return []
    return get_new_vendors(current_records, category, baseline_records)


def detect_emergency_contracts(records: list[dict], category: str) -> list[dict]:
    """
    Identifies contracts awarded via direct/emergency procurement
    (bypassing competitive tender), filtered to a given category.

    Emergency procurement bypasses competition requirements, often justified
    by urgency — which is itself a signal of operational tempo.

    OCDS signals for emergency procurement:
      - procurementMethod = "direct" or "limited" or "selective"
      - procurementMethodRationale containing "emergency", "urgent", "security"
      - tender.status = "complete" with very short tender period (<3 days)
    """
    emergency_keywords = {"emergency", "urgent", "security", "direct", "limited", "single"}

    flagged = []
    for r in records:
        if r.get("category") != category:
            continue

        raw = r.get("raw", {})
        tender = raw.get("tender", {})

        # Check procurement method
        method     = (tender.get("procurementMethod") or "").lower()
        rationale  = (tender.get("procurementMethodRationale") or "").lower()
        proc_type  = (tender.get("procurementMethodType") or "").lower()

        is_emergency = (
            method in {"direct", "limited", "selective"}
            or any(kw in rationale for kw in emergency_keywords)
            or any(kw in proc_type for kw in emergency_keywords)
        )

        if is_emergency:
            flagged.append(r)

    return flagged


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

def run_procurement_detection(
    region_id: str,
    target_date: date,
    buyer_ids: Optional[list[str]] = None,
    current_records: Optional[list[dict]] = None,
    baseline_records: Optional[list[dict]] = None,
) -> Optional[AnomalyEvent]:
    """
    Full procurement anomaly detection for a region and month.

    Checks all SENSITIVE_CATEGORIES for:
      1. Spend spike (z-score > PROCUREMENT_ZSCORE_THRESHOLD)
      2. New vendor patterns
      3. Emergency contracts

    The highest-scoring signal across all categories becomes the event.
    All signals are stored in raw_data for agent context.

    Args:
        region_id:        Must match a MonitoredRegion in config.py.
        target_date:      The month to analyze (uses year + month only).
        buyer_ids:        OCDS entity IDs to filter on. None = all in region.
        current_records:  Pre-fetched records for current month (for testing).
        baseline_records: Pre-fetched records for baseline period (for testing).

    Returns:
        AnomalyEvent if any signal fires, else None.
    """
    region = REGIONS_BY_ID.get(region_id)
    if region is None:
        raise ValueError(f"Unknown region_id '{region_id}'")

    effective_buyer_ids = buyer_ids if buyer_ids is not None else region.buyer_ids
    country_code = region.country_code

    # Fetch data if not supplied 
    if current_records is None:
        current_start = date(target_date.year, target_date.month, 1)
        # Last day of target month
        if target_date.month == 12:
            current_end = date(target_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            current_end = date(target_date.year, target_date.month + 1, 1) - timedelta(days=1)

        all_records = []
        for buyer_id in (effective_buyer_ids or [None]):
            try:
                recs = fetch_ocds_records(country_code, buyer_id, current_start, current_end)
                all_records.extend(recs)
            except Exception as e:
                log.warning(f"{region_id}: fetch failed for buyer {buyer_id}: {e}")
        current_records = all_records

    if baseline_records is None:
        baseline_end   = date(target_date.year, target_date.month, 1) - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=PROCUREMENT_BASELINE_MONTHS * 31)

        all_baseline = []
        for buyer_id in (effective_buyer_ids or [None]):
            try:
                recs = fetch_ocds_records(country_code, buyer_id, baseline_start, baseline_end)
                all_baseline.extend(recs)
            except Exception as e:
                log.warning(f"{region_id}: baseline fetch failed: {e}")
        baseline_records = all_baseline

    if not current_records and not baseline_records:
        log.debug(f"{region_id}: no procurement records available")
        return None

    # Combine for timeseries computation 
    all_combined = baseline_records + current_records
    current_month_label = f"{target_date.year}-{target_date.month:02d}"

    best_signal:    Optional[SignalType] = None
    best_zscore:    float = 0.0
    best_intensity: float = 0.0
    all_signals:    dict  = {}

    # Check each sensitive category 
    for category in SENSITIVE_CATEGORIES:
        spend_series = get_spend_timeseries(all_combined, category, group_by="month")
        if not spend_series:
            continue

        current_spend, mean, std, zscore = detect_spend_spike(
            spend_series, current_month_label
        )

        all_signals[category] = {
            "current_spend_usd": current_spend,
            "baseline_mean_usd": mean,
            "baseline_std_usd":  std,
            "zscore":            zscore,
            "fired":             zscore >= PROCUREMENT_ZSCORE_THRESHOLD,
        }

        if zscore >= PROCUREMENT_ZSCORE_THRESHOLD:
            intensity = _zscore_to_intensity(zscore, PROCUREMENT_ZSCORE_THRESHOLD)
            if intensity > best_intensity:
                best_intensity = intensity
                best_zscore    = zscore
                best_signal    = SignalType.SPEND_SPIKE
                all_signals[category]["dominant"] = True

    # New vendor check
    new_vendors: list[dict] = []
    for category in SENSITIVE_CATEGORIES:
        nvs = detect_new_vendor_pattern(current_records, baseline_records, category)
        new_vendors.extend(nvs)

    if new_vendors and best_signal is None:
        # New vendors alone trigger at low intensity if no spend spike
        best_signal    = SignalType.NEW_VENDOR_PATTERN
        best_intensity = 0.35
        best_zscore    = PROCUREMENT_ZSCORE_THRESHOLD   # placeholder

    # Emergency contract check 
    emergency_contracts: list[dict] = []
    for category in SENSITIVE_CATEGORIES:
        ecs = detect_emergency_contracts(current_records, category)
        emergency_contracts.extend(ecs)

    if emergency_contracts and best_signal is None:
        best_signal    = SignalType.EMERGENCY_CONTRACT
        best_intensity = 0.40
        best_zscore    = PROCUREMENT_ZSCORE_THRESHOLD

    # ── No signal fired ───────────────────────────────────────────────
    if best_signal is None:
        return None

    # Build AnomalyEvent 
    lat, lng = region.centroid()
    timestamp = datetime(target_date.year, target_date.month, 1, tzinfo=timezone.utc)

    raw_data = {
        "categories":           all_signals,
        "new_vendors":          [
            {"vendor": v["vendor_name"], "category": v["category"],
             "amount_usd": v["amount_usd"]}
            for v in new_vendors[:10]
        ],
        "emergency_contracts":  [
            {"contract_id": e["contract_id"], "category": e["category"],
             "amount_usd": e["amount_usd"], "title": e["title"][:100]}
            for e in emergency_contracts[:5]
        ],
        "current_month":        current_month_label,
        "total_current_records": len(current_records),
        "dominant_zscore":      best_zscore,
    }
    metadata = {
        "country_code":    country_code,
        "buyer_ids":       effective_buyer_ids,
        "baseline_months": PROCUREMENT_BASELINE_MONTHS,
    }

    return AnomalyEvent.make_procurement_event(
        region_id=region_id,
        country_code=country_code,
        lat=lat,
        lng=lng,
        timestamp=timestamp,
        signal_type=best_signal,
        intensity_score=round(min(best_intensity, 1.0), 4),
        raw_data=raw_data,
        metadata=metadata,
    )