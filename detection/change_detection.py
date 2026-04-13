# Satellite change detection
from __future__ import annotations
from datetime import datetime,date,timezone
from operator import imod
import numpy as np
from scipy.ndimage import uniform_filter
from typing import Optional
from config import(
    REGIONS_BY_ID,
    SATELLITE_CHANGE_THRESHOLD,
    SATELLITE_NDVI_DROP_THRESHOLD,
)

from ingestion.satellite import(
    BAND_BLUE,
    BAND_GREEN,
    BAND_RED,
    BAND_NIR,
    BAND_SWIR1,
    BAND_SWIR2,
    fetch_tile_pair_for_region
)

from normalization.schema import AnomalyEvent, AnomalySource,SignalType

# -----------------
# SCORE WEIGHTS
# How much each sub original value to the final composite layer
# The sum must be 1.
# -----------------

WEIGHT_NDVI_DROP= 0.40  # Vegetation loss — strongest single signal
WEIGHT_SPECTRAL= 0.25  # General spectral change — broad catch-all
WEIGHT_THERMAL= 0.20  # SWIR heat / fire signal
WEIGHT_STRUCTURE= 0.15  # Built-up change — new construction


assert abs(WEIGHT_NDVI_DROP + WEIGHT_SPECTRAL + WEIGHT_THERMAL + WEIGHT_STRUCTURE - 1.0) < 1e-6


# ------------------------
# CORE SPECTRAL INDICES
# ------------------------

def compute_ndvi(tile:np.ndarray)->np.ndarray:
    """
    COmpute the Normalized Difference Vegetation Index for every pixel
    NDVI = (NIR-Red) / (NIR+Red)
    
    Output range: [-1.0, 1.0]
    
    > 0.6  : Dense healthy vegetation (forest, crops in growing season)
      0.2–0.6: Sparse vegetation, shrubland, grassland
      0.0–0.2: Bare soil, rock, dry vegetation
      < 0.0  : Water, snow, or clouds (NIR < Red)
 
    Args:
        tile: (H, W, 6) float32 array from the satellite ingestion module.
              Bands must be in the order defined by ingestion/satellite.py:
              [Blue, Green, Red, NIR, SWIR1, SWIR2]
 
    Returns:
        (H, W) float32 array of NDVI values in [-1, 1].
        No-data pixels (where NIR + Red == 0) are assigned NDVI = 0.
    """
    nir=tile[:,:,BAND_NIR].astype(np.float64)
    red=tile[:,:,BAND_RED].astype(np.float64)
    
    denom= red+nir
    with np.errstate(invalid="ignore",divide="ignore"):
        ndvi=np.where(denom > 1e-6, (nir-red)/denom, 0.0)
    return np.clip(ndvi,-1.0,1.0).astype(np.float32)

def compute_ndbi(tile:np.ndarray)->np.ndarray:
    """
    Computes the Normalized Difference Built-up Index.
 
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
 
    Built-up surfaces (concrete, asphalt, bare soil) reflect more SWIR
    and less NIR than vegetation. High NDBI → more built-up / impervious
    surface. An increase in NDBI between two dates indicates new construction
    or clearing.
 
    Output range: [-1.0, 1.0]
    """
    swir1=tile[:,:,BAND_SWIR1].astype(np.float64)
    nir   = tile[:, :, BAND_NIR].astype(np.float64)
    denom = swir1 + nir
    with np.errstate(invalid="ignore", divide="ignore"):
        ndbi = np.where(denom > 1e-6, (swir1 - nir) / denom, 0.0)
    return np.clip(ndbi, -1.0, 1.0).astype(np.float32)

# ----------------------------
# CHANGE SCORE COMPONENTS
# ----------------------------

def _ndvi_drop_score(tile_before:np.ndarray, tile_after:np.ndarray)->tuple[float,dict]:
    """
    Computes a 0-1 score based on NDVI loss between two tiles
    
    Logic:
        - Compute NDVI for each tile
        - COmpute pixel wise NDVI difference: before - after (positive=vegetation was lost)
        - Apply a spacial smoothing filter to reduce salt and pepper noise from indivisual pixels. We want      
            spatially coherent change, not single pixel artefacts.
        - The score is 95 percentile of the positive differences normalized to [0,1].
        
    Why the 95th percentile rather than the mean or maximum?
      - Mean: too diluted by unchanged pixels (most of a large tile is stable).
      - Maximum: too sensitive to single noisy pixels.
      - 95th percentile: captures the most severely changed area while
        ignoring the top 5% of potential noise/outliers.     
    
    """ 
    ndvi_before=compute_ndvi(tile_before)
    ndvi_after=compute_ndvi(tile_after)
    
    # Positive values = vegetation loss
    # negative = regrowth(we ignore)
    ndvi_loss=ndvi_before - ndvi_after
    
    
    # Spacial smoothening:  SUppresses single-pixel noise from cloud edges or calibration artifacts.
    # 3x3 window is a good balance samll enough to preserve spacial detail, large enough to suppress single pixel noise. 
    ndvi_loss_smooth = uniform_filter(ndvi_loss,size=3)

    # Focus on pixels where vegetation was actually lost
    loss_pixels = ndvi_loss_smooth[ndvi_loss_smooth > 0]
    if loss_pixels.size == 0:
        return 0.0, {"ndvi_before_mean": float(ndvi_before.mean()),
                     "ndvi_after_mean": float(ndvi_after.mean()),
                     "ndvi_drop_score": 0.0,
                     "loss_pixel_fraction": 0.0}
 
    # 95th percentile of the loss distribution, clipped to [0, 1]
    # NDVI is in [-1, 1] so the maximum possible drop is 2.0.
    # A drop of 0.4 is significant; 0.8 is severe. We scale accordingly.
    p95_loss = float(np.percentile(loss_pixels, 95))
    score = float(np.clip(p95_loss / 0.8, 0.0, 1.0))  # 0.8 = "severe" anchor
 
    loss_fraction = float((ndvi_loss_smooth > SATELLITE_NDVI_DROP_THRESHOLD).mean())
 
    return score, {
        "ndvi_before_mean":   float(ndvi_before.mean()),
        "ndvi_after_mean":    float(ndvi_after.mean()),
        "ndvi_drop_p95":      p95_loss,
        "ndvi_drop_score":    score,
        "loss_pixel_fraction": loss_fraction,
    }
 
 
def _spectral_change_score(tile_before: np.ndarray, tile_after: np.ndarray) -> tuple[float, dict]:
    """
    Computes a 0–1 score based on overall spectral change across all bands.
 
    This is the most general signal — it catches any change in surface
    reflectance properties, regardless of whether it's vegetation, water,
    soil, or built-up surface.
 
    Method: Mean Absolute Difference (MAD) across all 6 bands per pixel,
    then the 90th percentile of the per-pixel MAD across the scene.
    """
    diff = np.abs(tile_after.astype(np.float64) - tile_before.astype(np.float64))
 
    # Per-pixel mean across all 6 bands: (H, W)
    per_pixel_mad = diff.mean(axis=2)
 
    # Smooth to suppress noise
    per_pixel_mad_smooth = uniform_filter(per_pixel_mad, size=3)
 
    # 90th percentile as scene-level score
    # MAD in [0, 1] range; 0.15 is moderately significant, 0.35 is severe
    p90_mad = float(np.percentile(per_pixel_mad_smooth, 90))
    score = float(np.clip(p90_mad / 0.35, 0.0, 1.0))  # 0.35 = "severe" anchor
 
    return score, {
        "spectral_mad_mean": float(per_pixel_mad.mean()),
        "spectral_mad_p90":  p90_mad,
        "spectral_score":    score,
    }
 
 
def _thermal_score(tile_before: np.ndarray, tile_after: np.ndarray) -> tuple[float, dict]:
    """
    Detects unusual thermal / fire signals using the SWIR bands.
 
    Sentinel-2's SWIR1 (Band 11) responds to heat sources above ambient.
    Active fires produce extremely high SWIR values. Burn scars (areas
    recently burned) also show elevated SWIR long after the fire is out.
 
    Method:
      1. Compute absolute SWIR1 values in the "after" tile.
      2. Compute SWIR1 change (after - before).
      3. A high absolute SWIR1 in the after tile (not present in before) is
         a thermal anomaly — either an active fire or a fresh burn scar.
 
    Score: 90th percentile of positive SWIR1 changes, normalised.
    """
    swir_before = tile_before[:, :, BAND_SWIR1].astype(np.float64)
    swir_after  = tile_after[:, :, BAND_SWIR1].astype(np.float64)
 
    # Positive = SWIR increased (warmer / more emissive)
    swir_increase = swir_after - swir_before
    swir_increase_smooth = uniform_filter(swir_increase, size=3)
 
    hot_pixels = swir_increase_smooth[swir_increase_smooth > 0]
 
    if hot_pixels.size == 0:
        return 0.0, {"swir1_before_mean": float(swir_before.mean()),
                     "swir1_after_mean": float(swir_after.mean()),
                     "thermal_score": 0.0,
                     "hot_pixel_fraction": 0.0}
 
    p90_swir = float(np.percentile(hot_pixels, 90))
    # SWIR values are in [0, 1]; a jump of 0.3 is significant, 0.6 is severe
    score = float(np.clip(p90_swir / 0.6, 0.0, 1.0))
 
    hot_fraction = float((swir_increase_smooth > 0.1).mean())
 
    return score, {
        "swir1_before_mean": float(swir_before.mean()),
        "swir1_after_mean":  float(swir_after.mean()),
        "swir1_increase_p90": p90_swir,
        "thermal_score":     score,
        "hot_pixel_fraction": hot_fraction,
    }
 
 
def _structure_change_score(tile_before: np.ndarray, tile_after: np.ndarray) -> tuple[float, dict]:
    """
    Detects new built-up structures using NDBI change.
 
    An increase in NDBI (built-up index) between two dates indicates that
    previously vegetated or bare-soil land has been converted to impervious
    surface — construction, a new compound, or large-scale clearing and
    paving.
 
    Score: 90th percentile of positive NDBI changes (new construction),
    normalised to [0, 1].
    """
    ndbi_before = compute_ndbi(tile_before)
    ndbi_after  = compute_ndbi(tile_after)
 
    ndbi_increase = ndbi_after.astype(np.float64) - ndbi_before.astype(np.float64)
    ndbi_increase_smooth = uniform_filter(ndbi_increase, size=3)
 
    build_pixels = ndbi_increase_smooth[ndbi_increase_smooth > 0]
 
    if build_pixels.size == 0:
        return 0.0, {"ndbi_before_mean": float(ndbi_before.mean()),
                     "ndbi_after_mean": float(ndbi_after.mean()),
                     "structure_score": 0.0}
 
    p90_ndbi = float(np.percentile(build_pixels, 90))
    # NDBI increase of 0.25 is notable; 0.5 is severe
    score = float(np.clip(p90_ndbi / 0.5, 0.0, 1.0))
 
    return score, {
        "ndbi_before_mean":   float(ndbi_before.mean()),
        "ndbi_after_mean":    float(ndbi_after.mean()),
        "ndbi_increase_p90":  p90_ndbi,
        "structure_score":    score,
    }
 
 
# ---------------------------------
# COMPOSITE CHANGE SCORE
# ---------------------------------
 
def compute_change_score(
    tile_before: np.ndarray,
    tile_after:  np.ndarray,
) -> tuple[float, SignalType, dict]:
    """
    Computes a composite 0–1 change score from all four sub-signals.
 
    Returns:
        (composite_score, dominant_signal_type, detail_dict)
 
    composite_score:
        Weighted combination of the four sub-scores.
        0.0 = no detectable change.
        1.0 = maximum detectable change across all signals.
        The threshold for triggering an AnomalyEvent is SATELLITE_CHANGE_THRESHOLD
        (default 0.35), configurable in config.py.
 
    dominant_signal_type:
        The SignalType enum value of whichever sub-signal contributed most
        to the composite score. Used to populate the AnomalyEvent.signal_type
        field so the agent knows what kind of change was detected.
 
    detail_dict:
        Full breakdown of all sub-scores and their intermediate calculations.
        Stored in AnomalyEvent.raw_data for transparency and debugging.
 
    WHY WEIGHTED SUM RATHER THAN MAX?
    ─────────────────────────────────
    A max-based combination would give equal credit to a tile with one very
    strong signal and one with moderate convergence across three signals.
    The weighted sum rewards multi-signal convergence — a tile that shows
    moderate NDVI loss, spectral change, AND a thermal signal should score
    higher than a tile with only one extreme signal. This aligns with the
    overall system philosophy: convergence is the primary quality signal.
    """
    ndvi_s,      ndvi_detail   = _ndvi_drop_score(tile_before, tile_after)
    spectral_s,  spec_detail   = _spectral_change_score(tile_before, tile_after)
    thermal_s,   therm_detail  = _thermal_score(tile_before, tile_after)
    structure_s, struct_detail = _structure_change_score(tile_before, tile_after)
 
    composite = (
        WEIGHT_NDVI_DROP  * ndvi_s    +
        WEIGHT_SPECTRAL   * spectral_s +
        WEIGHT_THERMAL    * thermal_s  +
        WEIGHT_STRUCTURE  * structure_s
    )
    composite = float(np.clip(composite, 0.0, 1.0))
 
    # Map sub-scores to their signal types for dominant signal identification
    sub_scores = {
        SignalType.VEGETATION_LOSS:   ndvi_s    * WEIGHT_NDVI_DROP,
        SignalType.LAND_COVER_CHANGE: spectral_s * WEIGHT_SPECTRAL,
        SignalType.THERMAL_ANOMALY:   thermal_s  * WEIGHT_THERMAL,
        SignalType.STRUCTURE_CHANGE:  structure_s * WEIGHT_STRUCTURE,
    }
    dominant_signal = max(sub_scores, key=sub_scores.__getitem__)
 
    detail = {
        "composite_score":  composite,
        "sub_scores": {
            "ndvi_drop":   ndvi_s,
            "spectral":    spectral_s,
            "thermal":     thermal_s,
            "structure":   structure_s,
        },
        "weights": {
            "ndvi_drop":  WEIGHT_NDVI_DROP,
            "spectral":   WEIGHT_SPECTRAL,
            "thermal":    WEIGHT_THERMAL,
            "structure":  WEIGHT_STRUCTURE,
        },
        "dominant_signal":  dominant_signal.value,
        **ndvi_detail,
        **spec_detail,
        **therm_detail,
        **struct_detail,
    }
 
    return composite, dominant_signal, detail
 
 
def detect_thermal_anomaly(tile: np.ndarray) -> tuple[bool, float]:
    """
    Flags whether a SINGLE tile contains an active fire or extreme heat source.
 
    Unlike compute_change_score(), which requires a before/after pair,
    this function operates on a single tile. It's used for screening —
    if the "after" tile itself shows extreme SWIR values, we can flag it
    even without a clean "before" reference (e.g. when baseline imagery
    is unavailable due to cloud cover).
 
    Method: Checks whether the 99th percentile of SWIR1 values exceeds
    the "active fire" threshold (0.50 in normalised reflectance units).
    Active fires saturate the SWIR band; even burn scars are distinctly
    elevated.
 
    Returns:
        (is_anomaly, swir_p99_value)
    """
    swir1 = tile[:, :, BAND_SWIR1].astype(np.float64)
    p99 = float(np.percentile(swir1, 99))
    FIRE_THRESHOLD = 0.50  # Calibrated against known fire events
    return p99 >= FIRE_THRESHOLD, p99
 
 
# ------------------------------------
# ORCHESTRATION
# ------------------------------------
 
def run_change_detection(
    region_id:      str,
    date_before:    date,
    date_after:     date,
    resolution_m:   int = 60,  # Default 60m for pipeline runs (faster); 10m for deep dives
) -> Optional[AnomalyEvent]:
    """
    Full change detection pipeline for a single region and date pair.
 
    This is the function called by the scheduler. It:
      1. Fetches the before/after tile pair (cache-first).
      2. Runs compute_change_score().
      3. If the composite score exceeds SATELLITE_CHANGE_THRESHOLD,
         creates and returns an AnomalyEvent.
      4. Returns None if the score is below threshold or tiles are unavailable.
 
    The returned AnomalyEvent uses the region centroid as the event location.
    This is a simplification — for high-resolution analysis, the actual
    centroid of the changed pixels could be computed. For the pipeline,
    the region centroid is sufficient for geographic clustering.
 
    Args:
        region_id:    Must be a key in config.REGIONS_BY_ID.
        date_before:  Reference / baseline date.
        date_after:   Current / target date to check for change.
        resolution_m: Tile resolution in metres. Lower = more detail but
                      larger arrays, slower processing, more cache space.
 
    Returns:
        AnomalyEvent if change_score >= SATELLITE_CHANGE_THRESHOLD, else None.
    """
    region = REGIONS_BY_ID.get(region_id)
    if region is None:
        raise ValueError(f"Unknown region_id '{region_id}'")
 
    # Step 1: Fetch tile pair -----------------
    tile_pair = fetch_tile_pair_for_region(
        region_id=region_id,
        date_before=date_before,
        date_after=date_after,
        resolution_m=resolution_m,
    )
 
    if tile_pair is None:
        # No cloud-free imagery available for this date range
        return None
 
    tile_before, tile_after, meta_before, meta_after = tile_pair
 
    # Step 2: Single-tile thermal screening -----------------------
    # Check the "after" tile for active fires before running full change
    # detection. Active fires can saturate bands and corrupt other scores.
    is_fire, swir_p99 = detect_thermal_anomaly(tile_after)
 
    # Step 3: Full change detection -----------------------
    composite_score, dominant_signal, detail = compute_change_score(
        tile_before, tile_after
    )
 
    # If we detected an active fire in the standalone check but composite
    # score is low (fire is small), still report it — override the threshold.
    fire_override = is_fire and swir_p99 > 0.65
 
    if composite_score < SATELLITE_CHANGE_THRESHOLD and not fire_override:
        return None
 
    # Step 4: Build AnomalyEvent --------------
    lat, lng = region.centroid()
 
    # Use the "after" date as the event timestamp — that's when the change
    # is observed. The "before" date is context stored in raw_data.
    event_timestamp = datetime(
        date_after.year, date_after.month, date_after.day,
        tzinfo=timezone.utc,
    )
 
    signal_type = SignalType.THERMAL_ANOMALY if fire_override else dominant_signal
 
    raw_data = {
        **detail,
        "date_before":       date_before.isoformat(),
        "date_after":        date_after.isoformat(),
        "tile_shape":        list(tile_after.shape),
        "resolution_m":      resolution_m,
        "active_fire_flag":  is_fire,
        "swir_p99":          swir_p99,
        "tile_cache_before": meta_before.get("cache_hit", False),
        "tile_cache_after":  meta_after.get("cache_hit", False),
    }
 
    metadata = {
        "cloud_cover_max_pct":  30,  # pipeline default
        "satellite":            "Sentinel-2 L2A",
        "band_config":          "B02/B03/B04/B08/B11/B12",
    }
 
    return AnomalyEvent.make_satellite_event(
        region_id=region_id,
        country_code=region.country_code,
        lat=lat,
        lng=lng,
        timestamp=event_timestamp,
        signal_type=signal_type,
        intensity_score=round(composite_score, 4),
        raw_data=raw_data,
        metadata=metadata,
    )
