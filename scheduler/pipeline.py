# Daily Pipeline Orchestrator
# Runs everyday at 02.00 UTC via APIScheduler
# Phase 1 :Satellite images only

# Each stage is completely isolated - one failing source never kills others
# All results are logged to pipeline_runs in PostresSQL

from __future__ import annotations

from locale import currency
import logging
import traceback
from datetime import date, datetime, timedelta, timezone
from turtle import st
from typing import Optional

from dotenv import parser
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
    """
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
    stage_results:dict[str,dict]={}
    total_events=0
    total_errors=0
    final_status="COMPLETED"
    
    
    try:
        # --------- Stage 1: Satellite ----------------
        sat_result=run_satellite_stage(
            target_date=target_date,
            region_ids=region_ids,
            resolution_m=resolution_m,
            dry_run=dry_run
        )
        
        stage_results['satellite']={
            "status":"ok" if sat_result["errors"]== 0 else "partial",
            "events_created":sat_result["events_created"],
            "errors":sat_result["errors"],
            "regions_results":sat_result['region_results']
        }
        total_events+=sat_result['events_created']
        total_errors+=sat_result['region_results']
        
        # -------- Persistant events -----------------
        if conn and not dry_run and sat_result['events']:
            from repository import save_anomaly_events_batch
            saved=save_anomaly_events_batch(sat_result['events'], conn)
            conn.commit()
            log.info(f"Persisted {saved} satellite events to DB")
            
            
        #  ----------- Stage 2 and 3 (GDELT and Procurement) -------------- 
        stage_results['gdelt']={"status":"pending"}
        stage_results['procurement']={"status":"pending"}
        
        # --------------- Stage 4: Agent ----------------
        stage_results['agent'] = {"status":"pending"}
        
        if total_errors >0 and total_events == 0:
            final_status="FAILED"
        elif total_errors > 0:
            final_status="PARTIAL"     
    
    except Exception as exc:
        final_status="FAILED"
        log.error(f"Pipeline Run Failed:{exc}")
        log.debug(traceback.format_exc())
    
    finally:
        elapsed = (datetime.now(timezone.utc) - run_started).total_seconds()
        if conn and run_id:
            try:
                from repository import update_pipeline_run
                update_pipeline_run(
                    conn, run_id, final_status,
                    stage_results, total_events, 0
                )
                conn.commit()
            except Exception:
                pass
            try:
                from db import get_db
                _db_ctx.__exit__(None, None, None)
            except Exception:
                pass    
    
    summary = {
        "run_id":        run_id,
        "target_date":   target_date.isoformat(),
        "status":        final_status,
        "total_events":  total_events,
        "total_errors":  total_errors,
        "elapsed_sec":   round(elapsed, 2),
        "stage_results": stage_results,
    }
    log.info(f"Pipeline complete | status={final_status} | "
             f"events={total_events} | elapsed={elapsed:.1f}s")
    return summary                  
        
        
# ------------------------------------------------------------------------------- 
# BACKFILL RUNNER
# Runs the pipeline for N historical days — used for testing and calibration.
# -------------------------------------------------------------------------------

def run_backfill(
    start_date:date,
    end_date:date,
    region_ids:Optional[list[str]] = None,
    resolution_m:int=60,
    dry_run:bool=False
)->list[dict]:
    """
    Runs the pipeline for every date in [start_date, end_date]
    Returns a list of per-date summary dicts

    Use this to process historical data for calibration or demo prep.
    Tiles are cached after first prep so re-running is cheap.
    """ 
    results=[]
    current=start_date
    while current <= end_date:
        log.info(f"Backfill: processing {current}")
        result=run_pipeline(
            target_date=current,
            region_ids=region_ids,
            resolution_m=resolution_m,
            dry_run=dry_run,
            use_db=not dry_run
        )
        results.append(result)
        current+=timedelta(days=1)
    return results



# ---------------------------------------------
# APSCHEDULER SETUP
# ---------------------------------------------
 
def build_scheduler():
    """
    Creates and returns a configured APScheduler instance.
    Called by the main entry point (api/main.py or a standalone runner).
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
 
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        func=lambda: run_pipeline(use_db=True),
        trigger=CronTrigger(
            hour=PIPELINE_RUN_HOUR_UTC,
            minute=PIPELINE_RUN_MINUTE_UTC,
        ),
        id="daily_pipeline",
        name="Witness Daily Pipeline",
        misfire_grace_time=3600,   # If missed by up to 1 hour, still run
        coalesce=True,             # If multiple missed runs, run only once
    )
    return scheduler
 

# ------------------------
# CLI Entry Point
# ------------------------
if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Witness Pipeline Runner")
    
    parser.add_argument("--date",       type=str, help="YYYY-MM-DD (default: yesterday)")
    
    parser.add_argument("--regions",    type=str, help="Comma-separated region IDs")
    
    parser.add_argument("--resolution", type=int, default=60)
    
    parser.add_argument("--dry-run",    action="store_true", help="Skip DB writes")
    
    parser.add_argument("--no-db",      action="store_true", help="Skip DB entirely")
    
    parser.add_argument("--backfill",   type=str, help="start:end dates e.g. 2021-01-01:2021-03-01")
    args = parser.parse_args()
 
    region_ids = args.regions.split(",") if args.regions else None
 
    if args.backfill:
        start_str, end_str = args.backfill.split(":")
        summaries = run_backfill(
            start_date=date.fromisoformat(start_str),
            end_date=date.fromisoformat(end_str),
            region_ids=region_ids,
            resolution_m=args.resolution,
            dry_run=args.dry_run,
        )
        total_events = sum(s["total_events"] for s in summaries)
        print(f"\n✓ Backfill complete: {len(summaries)} days, {total_events} total events")
    else:
        target = date.fromisoformat(args.date) if args.date else None
        summary = run_pipeline(
            target_date=target,
            region_ids=region_ids,
            resolution_m=args.resolution,
            dry_run=args.dry_run,
            use_db=not args.no_db,
        )
        import json
        print(json.dumps(summary, indent=2, default=str))