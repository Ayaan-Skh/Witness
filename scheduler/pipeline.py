"""
scheduler/pipeline.py — Daily Pipeline Orchestrator

Runs every day at 02:00 UTC via APScheduler.
Phase 1 (Days 7-8): satellite pipeline only.
Later phases add GDELT and procurement stages.

Each stage is fully isolated — one failing source never kills the others.
All results are logged to pipeline_runs in PostgreSQL.
"""

from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from config import (
    MONITORED_REGIONS,
    PIPELINE_LOOKBACK_DAYS,
    PIPELINE_RUN_HOUR_UTC,
    PIPELINE_RUN_MINUTE_UTC,
    GDELT_BASELINE_LOOKBACK_DAYS,
)
from detection.change_detection import run_change_detection
from normalization.schema import AnomalyEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("witness.pipeline")


# ─────────────────────────────────────────────
# SATELLITE STAGE
# ─────────────────────────────────────────────

def run_satellite_stage(
    target_date: date,
    region_ids: Optional[list[str]] = None,
    resolution_m: int = 60,
    dry_run: bool = False,
) -> dict:
    """
    Runs change detection for every monitored region for target_date.

    Uses (target_date - 30 days) as the reference "before" date.
    30 days is long enough to cross most seasonal artefacts within a month
    while being short enough to catch rapid construction or clearance.

    Args:
        target_date:  The "after" date to analyze.
        region_ids:   Subset of regions to run. None = all configured regions.
        resolution_m: Tile resolution. 60m for daily runs (fast); 10m for deep dives.
        dry_run:      If True, fetch tiles and score but don't persist to DB.

    Returns:
        {
          "events": [AnomalyEvent, ...],
          "region_results": {"eth_tigray": "ok" | "no_data" | "error: ...", ...},
          "events_created": int,
          "errors": int,
        }
    """
    regions = [r for r in MONITORED_REGIONS
               if region_ids is None or r.region_id in region_ids]

    date_before = target_date - timedelta(days=30)
    log.info(f"Satellite stage: {len(regions)} regions | "
             f"{date_before} → {target_date} | res={resolution_m}m")

    events: list[AnomalyEvent] = []
    region_results: dict[str, str] = {}
    error_count = 0

    for region in regions:
        try:
            event = run_change_detection(
                region_id=region.region_id,
                date_before=date_before,
                date_after=target_date,
                resolution_m=resolution_m,
            )
            if event is not None:
                events.append(event)
                region_results[region.region_id] = (
                    f"anomaly:{event.signal_type.value}:{event.intensity_score:.3f}"
                )
                log.info(f"  ✓ {region.region_id}: {event.signal_type.value} "
                         f"score={event.intensity_score:.3f}")
            else:
                region_results[region.region_id] = "no_anomaly"
                log.info(f"  · {region.region_id}: below threshold / no imagery")

        except Exception as exc:
            error_count += 1
            msg = f"error: {type(exc).__name__}: {exc}"
            region_results[region.region_id] = msg
            log.error(f"  ✗ {region.region_id}: {msg}")
            log.debug(traceback.format_exc())

    log.info(f"Satellite stage complete: {len(events)} events, {error_count} errors")
    return {
        "events":         events,
        "region_results": region_results,
        "events_created": len(events),
        "errors":         error_count,
    }




# ─────────────────────────────────────────────
# GDELT STAGE
# ─────────────────────────────────────────────

def run_gdelt_stage(
    target_date,
    region_ids=None,
    dry_run: bool = False,
) -> dict:
    """
    Runs GDELT anomaly detection for every monitored region.
    Builds 90-day baseline + current-day timeseries per region.
    Returns same shape dict as run_satellite_stage.
    """
    from detection.gdelt_anomaly import run_gdelt_detection
    from ingestion.gdelt import query_tone_timeseries, query_volume_timeseries, fill_missing_dates

    regions = [r for r in MONITORED_REGIONS
               if region_ids is None or r.region_id in region_ids]

    baseline_start = target_date - timedelta(days=GDELT_BASELINE_LOOKBACK_DAYS)
    log.info(f"GDELT stage: {len(regions)} regions | {baseline_start} -> {target_date}")

    events = []
    region_results = {}
    error_count = 0

    for region in regions:
        try:
            tone_series   = query_tone_timeseries(
                region.country_code, region.admin1, baseline_start, target_date)
            volume_series = query_volume_timeseries(
                region.country_code, region.admin1, baseline_start, target_date)
            tone_series   = fill_missing_dates(tone_series,   baseline_start, target_date)
            volume_series = fill_missing_dates(volume_series, baseline_start, target_date)

            event = run_gdelt_detection(
                region_id=region.region_id,
                target_date=target_date,
                tone_series=tone_series,
                volume_series=volume_series,
            )
            if event:
                events.append(event)
                region_results[region.region_id] = (
                    f"anomaly:{event.signal_type.value}:{event.intensity_score:.3f}")
                log.info(f"  ✓ {region.region_id}: {event.signal_type.value}")
            else:
                region_results[region.region_id] = "no_anomaly"
        except Exception as exc:
            error_count += 1
            region_results[region.region_id] = f"error: {type(exc).__name__}: {exc}"
            log.error(f"  ✗ {region.region_id}: {exc}")

    return {
        "events": events, "region_results": region_results,
        "events_created": len(events), "errors": error_count,
    }


# ─────────────────────────────────────────────
# PROCUREMENT STAGE
# ─────────────────────────────────────────────

def run_procurement_stage(
    target_date,
    region_ids=None,
    dry_run: bool = False,
) -> dict:
    """
    Runs procurement anomaly detection for every monitored region.
    Fetches current month + 12-month baseline per region/buyer.
    """
    from detection.procurement_anomaly import run_procurement_detection

    regions = [r for r in MONITORED_REGIONS
               if region_ids is None or r.region_id in region_ids]

    log.info(f"Procurement stage: {len(regions)} regions | target={target_date.strftime('%Y-%m')}")

    events = []
    region_results = {}
    error_count = 0

    for region in regions:
        try:
            event = run_procurement_detection(
                region_id=region.region_id,
                target_date=target_date,
                buyer_ids=region.buyer_ids if region.buyer_ids else None,
            )
            if event:
                events.append(event)
                region_results[region.region_id] = (
                    f"anomaly:{event.signal_type.value}:{event.intensity_score:.3f}")
                log.info(f"  ✓ {region.region_id}: {event.signal_type.value}")
            else:
                region_results[region.region_id] = "no_anomaly"
        except Exception as exc:
            error_count += 1
            region_results[region.region_id] = f"error: {type(exc).__name__}: {exc}"
            log.error(f"  ✗ {region.region_id}: {exc}")

    return {
        "events": events, "region_results": region_results,
        "events_created": len(events), "errors": error_count,
    }

# ─────────────────────────────────────────────
# FULL PIPELINE RUN
# ─────────────────────────────────────────────

def run_pipeline(
    target_date: Optional[date] = None,
    region_ids: Optional[list[str]] = None,
    resolution_m: int = 60,
    dry_run: bool = False,
    use_db: bool = True,
) -> dict:
    """
    Runs the complete ingestion → detection → agent pipeline for one date.

    Currently runs satellite stage only (Days 7-8).
    GDELT and procurement stages are added in Days 9-16.
    Agent stage is added in Days 19-30.

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
        target_date = date.today() - timedelta(days=1)

    run_started = datetime.now(timezone.utc)
    log.info(f"Pipeline run starting | date={target_date} | dry_run={dry_run}")

    # ── Try to get a DB connection ────────────────────────────────────
    conn = None
    run_id = None
    if use_db and not dry_run:
        try:
            from db import get_db
            from repository import log_pipeline_run, update_pipeline_run, save_anomaly_events_batch
            _db_ctx = get_db()
            conn = _db_ctx.__enter__()
            run_id = log_pipeline_run(conn)
            conn.commit()
            log.info(f"Pipeline run_id: {run_id}")
        except Exception as e:
            log.warning(f"DB unavailable — running in no-persist mode: {e}")
            conn = None

    stage_results: dict[str, dict] = {}
    total_events = 0
    total_errors = 0
    final_status = "COMPLETED"

    try:
        # ── Stage 1: Satellite ────────────────────────────────────────
        sat_result = run_satellite_stage(
            target_date=target_date,
            region_ids=region_ids,
            resolution_m=resolution_m,
            dry_run=dry_run,
        )
        stage_results["satellite"] = {
            "status":         "ok" if sat_result["errors"] == 0 else "partial",
            "events_created": sat_result["events_created"],
            "errors":         sat_result["errors"],
            "region_results": sat_result["region_results"],
        }
        total_events += sat_result["events_created"]
        total_errors += sat_result["errors"]

        # ── Persist events ────────────────────────────────────────────
        if conn and not dry_run and sat_result["events"]:
            from repository import save_anomaly_events_batch
            saved = save_anomaly_events_batch(sat_result["events"], conn)
            conn.commit()
            log.info(f"Persisted {saved} satellite events to DB")

        # ── Stage 2: GDELT ───────────────────────────────────────────────
        gdelt_result = run_gdelt_stage(
            target_date=target_date, region_ids=region_ids, dry_run=dry_run)
        stage_results["gdelt"] = {
            "status":         "ok" if gdelt_result["errors"] == 0 else "partial",
            "events_created": gdelt_result["events_created"],
            "errors":         gdelt_result["errors"],
            "region_results": gdelt_result["region_results"],
        }
        total_events += gdelt_result["events_created"]
        total_errors += gdelt_result["errors"]

        if conn and not dry_run and gdelt_result["events"]:
            from repository import save_anomaly_events_batch
            save_anomaly_events_batch(gdelt_result["events"], conn)
            conn.commit()

        # ── Stage 3: Procurement ─────────────────────────────────────────
        proc_result = run_procurement_stage(
            target_date=target_date, region_ids=region_ids, dry_run=dry_run)
        stage_results["procurement"] = {
            "status":         "ok" if proc_result["errors"] == 0 else "partial",
            "events_created": proc_result["events_created"],
            "errors":         proc_result["errors"],
            "region_results": proc_result["region_results"],
        }
        total_events += proc_result["events_created"]
        total_errors += proc_result["errors"]

        if conn and not dry_run and proc_result["events"]:
            from repository import save_anomaly_events_batch
            save_anomaly_events_batch(proc_result["events"], conn)
            conn.commit()

        # ── Stage 4: Agent — added next session ──────────────────────────
        stage_results["agent"] = {"status": "pending"}

        if total_errors > 0 and total_events == 0:
            final_status = "FAILED"
        elif total_errors > 0:
            final_status = "PARTIAL"

    except Exception as exc:
        final_status = "FAILED"
        log.error(f"Pipeline run failed: {exc}")
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


# ─────────────────────────────────────────────
# BACKFILL RUNNER
# Runs the pipeline for N historical days — used for testing and calibration.
# ─────────────────────────────────────────────

def run_backfill(
    start_date: date,
    end_date: date,
    region_ids: Optional[list[str]] = None,
    resolution_m: int = 60,
    dry_run: bool = False,
) -> list[dict]:
    """
    Runs the pipeline for every date in [start_date, end_date].
    Returns a list of per-date summary dicts.

    Use this to process historical data for calibration or demo prep.
    Tiles are cached after the first fetch, so re-running is cheap.
    """
    results = []
    current = start_date
    while current <= end_date:
        log.info(f"Backfill: processing {current}")
        result = run_pipeline(
            target_date=current,
            region_ids=region_ids,
            resolution_m=resolution_m,
            dry_run=dry_run,
            use_db=not dry_run,
        )
        results.append(result)
        current += timedelta(days=1)
    return results


# ─────────────────────────────────────────────
# APSCHEDULER SETUP
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
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