# Daily Pipeline Orchestrator
# Runs everyday at 02.00 UTC via APIScheduler
# Phase 1 :Satellite images only

# Each stage is completely isolated - one failing source never kills others
# All results are logged to pipeline_runs in PostresSQL

from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from numpy.lib.introspect import opt_func_info

from config import (
    MONITORED_REGIONS,
    PIPELINE_LOOKBACK_DAYS,
    PIPELINE_RUN_HOUR_UTC,
    PIPELINE_RUN_MINUTE_UTC
)

from detection.change_detection import run_change_detection
from normalization.schema import AnomalyEvent

logging.basicConfig(
    level=logging.info,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log=logging.getLogger("witness.pipeline")


# -------------------------
#   SATELLITE STAGE
# -------------------------

def run_satellite_stage(
    target_date:date,
    region_ids:Optional[list[str]]=None,
    resolution_m:int=60,
    dry_run:bool=False
):
    f"""
    Runs change for every monitored region for target_date.
    
    Uses (target_date - 30 days) as the refrence before the date.
    30 days is long enough to cross most seasonal artefacts within a month while being short enough rapid construction or clearance.
    
    Args:
        target_date:The "after" date to analyze
        region_ids:   Subset of regions to run. None = all configured regions.
        resolution_m: Tile resolution. 60m for daily runs (fast); 10m for deep dives.
        dry_run:  If True, fetch tiles and score but don't persist to DB.    
    
    Returns:
        {
          "events": [AnomalyEvent, ...],
          "region_results": {"eth_tigray": "ok" | "no_data" | "error: ...", ...},
          "events_created": int,
          "errors": int,
        }
    """
    regions=[r for r in MONITORED_REGIONS 
             if region_ids is None or r.region_id in region_ids]
    
    date_before=target_date-timedelta(days=30)
    logging.info(f"Satellite stage: {len(regions)} regions |" 
                 f"{date_before} ->{target_date} | res={resolution_m} m")
    
    events:list[AnomalyEvent]=[]
    region_results:dict[str,str] = {}
    error_count=0
    
    for region in regions:
        try:
            event = run_change_detection(
                region_id=region.region_id,
                date_before=date_before,
                date_after=target_date,
                resolution_m=resolution_m
            )
            if event is not None:
                events.append(event)
                region_results[region.region_id]=(
                    f"anomaly:{event.signal_type.value}: {event.intensity_score:.3f}"
                )
                log.info(
                    f"{region.region_id}:{event.signal_type.value}"
                    f" score={event.intensity_score:.3f}"
                )
            else:
                region_results[region.region_id] = "no_anomaly"
                log.info(f"{region.region_id}:below threshold / no imagery")
        except Exception as exe:
            error_count = 1
            msg=f"error:{type(exe).__name__}:{exe}"
            region_results[region.region_id]=msg
            log.error(f" {region.region_id}:{msg}") 
            log.debug(traceback.format_exc())
    
    log.info(f"Satellite stage complete: {len(events)} events, {error_count} errors")
    return {
        "events":         events,
        "region_results": region_results,
        "events_created": len(events),
        "errors":         error_count,
    }                   
    
    
def run_pipeline(
    target_date:Optional[date],
    region_ids:Optional[list[str]],
    resolution_m:int=60,
    dry_run:bool=False,
    use_db:bool=True
):
    """
    Runs complete ingestion -> detection -> agent pipeline for one date
    
    Args:
        target_date:  Date to analyze. Defaults to yesterday (last complete day).
        region_ids:   Restrict to specific regions. None = all.
        resolution_m: Tile resolution for satellite stage.
        dry_run:      Process data but skip all DB writes.
        use_db:       If False, skip DB entirely (for offline testing).
 
    Returns:
        Summary dict with per-stage results and totals.
    """
    if target_date is None:
        target_date=date.today()-timedelta(days=1)
    
    run_started=datetime.now(timezone.utc)
    logging.info(f"Pipeline running started | date={target_date} | dry_run={dry_run}")
    
    # Try to get DB connection
    conn=None
    run_id=None
    if use_db and not dry_run:
        try:
            from db import get_db
            from repository import log_pipeline_run, update_pipeline_run, save_anomaly_events_batch
            _db_ctx=get_db()
            conn=_db_ctx.__enter__
            run_id=log_pipeline_run(conn)
            conn.commit()
            logging.info(f"Pipeline run_id:{run_id}")
        except Exception as e:
            log.warning(f"DB unavailable — running in no-persist mode: {e}")
            conn = None
     