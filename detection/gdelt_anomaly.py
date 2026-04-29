"""
What this file does?
Takes the gdelt data (tone + volume) for a region and determines wheather the current peroid is statistically anomalous compared to the historical baseline. If it is produces a anomaly

What we detect
1. TONE_CRASH
    z-score of daily ang tone drops below -2.0(2 standarad deviation negative)
    A sudden shift to extremely negative coverage- the regions news goes from neutral/ mixed to wall to wall crisis coverage

2. COMMUNICATION_BLACKOUT
    z-score of daily mentions volume drops below -2.0
    Unsual scilence in coverage, fewer articals than normal
    Can indicate: Joirnalist access denial, communication infrastructure destruction or active supression of reporting.

3. VOLUME_SPIKE + CONFLICT_EVENTS
    z-score of daily volume rises above +2.0 AND concerning CAMEO codes are present. High coverage of conflict events.
    
Why 2 standard deviations?
->  2 std covers 95% of the normal distributions. Anything outside is in the top/bottom 2.3% of observations.
For a 90 day baseline, this means ~2 days will naturally fall outside the threshold by chance, an acceptable false positive rate for a system where alert is reviewed by humans before publications.    
"""


from __future__ import annotations

import logging 
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
from config import (
    GDELT_BASELINE_LOOKBACK_DAYS,
    GDELT_BLACKOUT_THRESHOLD,
    GDELT_TONE_CRASH_THRESHOLD,
    GDELT_ZSCORE_THRESHOLD,
    REGIONS_BY_ID,
)

from ingestion.gdelt import (
    fill_missing_dates,
    get_top_themes,
    query_tone_timeseries,
    query_volume_timeseries
)

from normalization.schema import AnomalyEvent, AnomalySource, SignalType

log=logging.info("witness.gdelt_anomaly")

#----------------------------
# Baseline computation
#----------------------------
def compute_baseline(
    values:list[float],
    excluded_zeros:bool=False
)->tuple[float,float]:
    """
    Computes the mean and standard deviation of a time series.
    Agrs:
        values:         List of daily metric values over the basleine period
        excluded_zeros: If true zero values are excluded from the baseline.
                        Used for tone: a day with zero events has no meaningful tone value, and including it would skew the baseline
    
    Returns:
    (mean, std_dev)- the 'normal' range for this metric in this range
    
    Edge cases:
        - If std_dev == 0 (all values identical), return std=1.0 to avoid
          division by zero. This happens for very quiet regions with flat
          timeseries - we treat any deviation as anomalous.
        - If fewer than 7 values, the baseline is unreliable. Callers should
          check sample count before trusting z-scores.
    """
    arr=np.array([v for v in values if v is not None], dtype=np.float64)
    if excluded_zeros:
        arr=arr[arr != 0]
    
    if len(arr) == 0:
        return 0.0,1.0
    
    mean=float(np.mean(arr))
    std=float(np.std(arr,ddof=1))
    
    if std < 1e-6:
        std=1.0   # Flat timeseries, any change is anomalous
    
    return mean, std    
        
def compute_zscore(
    value:float,
    mean:float,
    std:float
    )->float:
    """
    Standard z-score: how many standard deviations from the mean?
    Negative z = below average.  Positive z = above average.
    |z| > 2 = outside 95% of normal distribution.
    |z| > 3 = outside 99.7% — very unusual.
    """
    if std< 1e-6:
        return 0.0
    return (value-mean)/std

#---------------------------------
#  SIGNAL DETECTION
#---------------------------------
    
def detect_tone_crash(
    region_id:str,
    target_date:date,
    tone_series:list[dict]
)-> tuple[Optional[float],float,float,float]:
    
    """
    Detects a sudden shift to extreme negative sentiment.
 
    Strategy:
      - Split the tone timeseries into baseline (all days before target_date)
        and current (target_date only).
      - Compute baseline mean/std from the historical period.
      - Compute z-score for the current day's tone.
      - If z-score <= GDELT_TONE_CRASH_THRESHOLD (-2.0), it's anomalous.
 
    Returns:
        (current_tone, baseline_mean, baseline_std, zscore)
        Returns (None, 0, 1, 0) if current date has no data.
    """
    baseline_rows=[r for r in tone_series if r['event_date'] < target_date]
    current_rows=[r for r in tone_series if r['event_date'] == target_date]
    
    if not current_rows or current_rows[0]['avg_tone'] is None:
        log.debug(f"{region_id} has no tone data for {target_date}")
        return 0.0,1.0,0.0
    
    current_tone=current_rows[0]['avg_tone']
    
    baseline_tones=[r['avg_tone'] for r in baseline_rows if r['avg_tone'] is not None]
    if len(baseline_tones) < 7:
        log.warning(f'{region_id}: Only {len(baseline_tones)} baseline tone days data available')
    
    mean,std=compute_baseline(baseline_tones,excluded_zeros=True)
    zscore=compute_zscore(current_tone, mean, std)
    
    return current_tone, mean, std        
        

def detect_communication_blackout(
    region_id: str,
    target_date: date,
    volume_series: list[dict],
) -> tuple[Optional[int], float, float, float]:
    """
    Detects an anomalous drop in news coverage volume.
 
    Same structure as detect_tone_crash but for mention_count instead of tone.
    A strongly negative z-score (below GDELT_BLACKOUT_THRESHOLD = -2.0)
    means "much less coverage than normal."
 
    Returns:
        (current_volume, baseline_mean, baseline_std, zscore)
    """
    baseline_rows = [r for r in volume_series if r["event_date"] < target_date]
    current_rows  = [r for r in volume_series if r["event_date"] == target_date]
 
    if not current_rows:
        # No data for target_date at all — this IS a blackout signal.
        # Return 0 mentions with a very negative z-score.
        baseline_vols = [r["mention_count"] for r in baseline_rows]
        mean, std     = compute_baseline(baseline_vols)
        zscore        = compute_zscore(0, mean, std)
        return 0, mean, std, zscore
 
    current_vol = current_rows[0]["mention_count"]
    baseline_vols = [r["mention_count"] for r in baseline_rows]
 
    if len(baseline_vols) < 7:
        log.warning(f"{region_id}: only {len(baseline_vols)} baseline volume days")
 
    mean, std = compute_baseline(baseline_vols)
    zscore    = compute_zscore(current_vol, mean, std)
 
    return current_vol, mean, std, zscore
 
 
def detect_volume_spike(
    region_id: str,
    target_date: date,
    volume_series: list[dict],
) -> tuple[Optional[int], float, float, float]:
    """
    Detects an anomalous INCREASE in coverage volume.
 
    Positive z-score above GDELT_ZSCORE_THRESHOLD (+2.0) = more coverage
    than normal. Combined with concerning CAMEO codes, this is a strong
    conflict signal.
    """
    baseline_rows = [r for r in volume_series if r["event_date"] < target_date]
    current_rows  = [r for r in volume_series if r["event_date"] == target_date]
 
    if not current_rows:
        return None, 0.0, 1.0, 0.0
 
    current_vol   = current_rows[0]["mention_count"]
    baseline_vols = [r["mention_count"] for r in baseline_rows]
    mean, std     = compute_baseline(baseline_vols)
    zscore        = compute_zscore(current_vol, mean, std)
 
    return current_vol, mean, std, zscore


def _zscore_to_intensity(zscore: float, threshold: float = 2.0) -> float:
    """
    Maps an absolute z-score to a 0–1 intensity score.
 
    At z = threshold (2.0):  intensity = 0.0 (just barely anomalous).
    At z = threshold * 2.5 (5.0): intensity = 1.0 (extremely anomalous).
    Linear interpolation between.
 
    Why not just use z/max_z?
      Z-scores have no natural ceiling — a z of 10 is possible. We want a
      bounded, comparable score across all sources, so we anchor the
      "maximum" at 2.5× the threshold (5.0 std devs for the default threshold).
    """
    abs_z   = abs(zscore)
    anchored = abs_z - threshold              # 0 at threshold, positive above
    scaled   = anchored / (threshold * 1.5)  # 1.0 at 2.5× threshold
    return float(np.clip(scaled, 0.0, 1.0))
 
 
# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────
 
def run_gdelt_detection(
    region_id: str,
    target_date: date,
    tone_series: Optional[list[dict]]   = None,
    volume_series: Optional[list[dict]] = None,
    themes: Optional[list[dict]]        = None,
) -> Optional[AnomalyEvent]:
    """
    Full GDELT anomaly detection for a region and date.
 
    Accepts pre-fetched series (for testing / batch efficiency) or
    fetches them itself if not provided.
 
    Detection priority:
      1. COMMUNICATION_BLACKOUT (volume crash) — highest priority.
         Silence is often the strongest signal.
      2. TONE_CRASH — extreme negative sentiment shift.
      3. CONFLICT_EVENTS — volume spike with concerning CAMEO codes.
 
    If multiple signals fire, the one with the highest |z-score| wins
    as the primary signal_type; all signals are recorded in raw_data.
 
    Returns:
        AnomalyEvent if any signal exceeds threshold, else None.
    """
    region = REGIONS_BY_ID.get(region_id)
    if region is None:
        raise ValueError(f"Unknown region_id '{region_id}'")
 
    baseline_start = target_date - timedelta(days=GDELT_BASELINE_LOOKBACK_DAYS)
    baseline_end   = target_date   # inclusive — detector splits internally
 
    # ── Fetch if not supplied ─────────────────────────────────────────
    if tone_series is None:
        try:
            tone_series = query_tone_timeseries(
                region.country_code, region.admin1,
                baseline_start, baseline_end,
            )
        except Exception as e:
            log.error(f"{region_id}: tone fetch failed: {e}")
            return None
 
    if volume_series is None:
        try:
            volume_series = query_volume_timeseries(
                region.country_code, region.admin1,
                baseline_start, baseline_end,
            )
        except Exception as e:
            log.error(f"{region_id}: volume fetch failed: {e}")
            return None
 
    # Fill gaps with zeros so baseline stats are accurate
    tone_series   = fill_missing_dates(tone_series,   baseline_start, target_date)
    volume_series = fill_missing_dates(volume_series, baseline_start, target_date)
 
    # ── Run all detectors ─────────────────────────────────────────────
    current_tone,   tone_mean,   tone_std,   tone_z   = detect_tone_crash(
        region_id, target_date, tone_series)
    current_vol,    vol_mean,    vol_std,    vol_z    = detect_communication_blackout(
        region_id, target_date, volume_series)
    current_spike,  spike_mean,  spike_std,  spike_z  = detect_volume_spike(
        region_id, target_date, volume_series)
 
    # ── Determine which signals fired ─────────────────────────────────
    fired: dict[SignalType, float] = {}  # signal → abs(z-score)
 
    if tone_z   <= GDELT_TONE_CRASH_THRESHOLD:
        fired[SignalType.TONE_CRASH]             = abs(tone_z)
 
    if vol_z    <= GDELT_BLACKOUT_THRESHOLD:
        fired[SignalType.COMMUNICATION_BLACKOUT] = abs(vol_z)
 
    if spike_z  >= GDELT_ZSCORE_THRESHOLD:
        fired[SignalType.VOLUME_SPIKE]           = abs(spike_z)
 
    if not fired:
        return None   # Nothing anomalous
 
    # ── Dominant signal = highest |z| ─────────────────────────────────
    dominant_signal = max(fired, key=fired.__getitem__)
    dominant_z      = fired[dominant_signal]
    intensity       = _zscore_to_intensity(dominant_z)
 
    # ── Fetch themes if not supplied ──────────────────────────────────
    if themes is None:
        try:
            window_start = target_date - timedelta(days=7)
            themes = get_top_themes(
                region.country_code, region.admin1,
                window_start, target_date,
            )
        except Exception as e:
            log.warning(f"{region_id}: themes fetch failed (non-fatal): {e}")
            themes = []
 
    concerning_themes = [t for t in (themes or []) if t.get("is_concerning")]
    # If we have a volume spike AND concerning themes, upgrade to CONFLICT_EVENTS
    if spike_z >= GDELT_ZSCORE_THRESHOLD and concerning_themes:
        fired[SignalType.CONFLICT_EVENTS] = abs(spike_z)
        if dominant_signal == SignalType.VOLUME_SPIKE:
            dominant_signal = SignalType.CONFLICT_EVENTS
 
    # ── Build AnomalyEvent ────────────────────────────────────────────
    lat, lng = region.centroid()
    timestamp = datetime(target_date.year, target_date.month, target_date.day,
                         tzinfo=timezone.utc)
 
    raw_data = {
        "tone": {
            "current":       current_tone,
            "baseline_mean": tone_mean,
            "baseline_std":  tone_std,
            "zscore":        tone_z,
            "fired":         SignalType.TONE_CRASH in fired,
        },
        "volume": {
            "current":       current_vol,
            "baseline_mean": vol_mean,
            "baseline_std":  vol_std,
            "zscore_drop":   vol_z,
            "zscore_spike":  spike_z,
            "blackout_fired": SignalType.COMMUNICATION_BLACKOUT in fired,
            "spike_fired":   SignalType.VOLUME_SPIKE in fired,
        },
        "top_concerning_themes": concerning_themes[:5],
        "all_fired_signals":     {s.value: z for s, z in fired.items()},
        "baseline_days":         GDELT_BASELINE_LOOKBACK_DAYS,
    }
 
    metadata = {
        "country_code": region.country_code,
        "admin1":       region.admin1,
        "target_date":  target_date.isoformat(),
        "dominant_z":   dominant_z,
    }
 
    return AnomalyEvent.make_gdelt_event(
        region_id=region_id,
        country_code=region.country_code,
        lat=lat,
        lng=lng,
        timestamp=timestamp,
        signal_type=dominant_signal,
        intensity_score=round(intensity, 4),
        raw_data=raw_data,
        metadata=metadata,
    )




















