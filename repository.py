# Database Read/Write layer
# All SQL lives here. No other module touches PostgreSQL
# This seperation means if we ever swap PostgreSQL for something else, only this file changes

from __future__ import annotations

from ast import List
import json 
import stat
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from normalization.schema import AnomalyEvent, AnomalySource, SignalType, InvestigationBrief, BriefStatus, ConfidenceTier

def save_anomaly_event(event:AnomalyEvent,conn)->str:
    """Insert an anolamy event to anomaly_events and return event_id"""
    with conn.cursor() as cur:
        cur.execute(
        """
        Insert into anomaly_events
            (event_id, source, region_id, country_code,
                 location, lat, lng, timestamp, signal_type,
                 intensity_score, raw_data, metadata)
            VALUES (
                %s, %s, %s, %s,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (event_id) DO NOTHING
        """,
        (
                event.event_id,
                event.source.value,
                event.region_id,
                event.country_code,
                event.lng, event.lat,   # ST_MakePoint takes (lng, lat)
                event.lat, event.lng,
                event.timestamp,
                event.signal_type.value,
                event.intensity_score,
                json.dumps(event.raw_data),
                json.dumps(event.metadata),
            )
        )
    return event.event_id


def save_anomaly_events_batch(events:list[AnomalyEvent],conn)->int:
    """ 
    Bulk insert of AnomalyEvents. Returns count inserted
    """
    if not events:
        return 0
    inserted = 0
    for event in events:
        save_anomaly_event(event=event,conn=conn)
        inserted += 1
    return inserted

def get_events_for_region(
        region_id:str,
        since:datetime,
        conn,
        source:Optional[str]=None
    )->list[AnomalyEvent]:
    """
    Fetch ANomalyEvents for a region since a given datetime.
    """
    query="""
        SELECTevent_id,source,region_id,country_code, lat,lng, timestamp, signal_type, intensity_score, raw_data, metadata, detected_at
        FROM anomaly_events
        WHERE region_id =%s AND timestamp>= %s
    """    
    params=[region_id,since]
    if source:
        query += "AND source = %s"
        params.append(source)
    query+="ORDER BY timestamp DESC"    
    
    with conn.cursor() as cur:
        cur.execute(query,params)
        rows=cur.featchall()
    events=[]
    for row in rows:
        events.append(AnomalyEvent(
            event_id=str(row[0]),
            source=AnomalySource(row[1]),
            region_id=row[2],
            country_code=row[3],
            lat=float(row[4]),
            lng=float(row[5]),
            timestamp=row[6] if row[6].tzinfo else row[6].replace(tzinfo=timezone.utc),
            signal_type=SignalType(row[7]),
            intensity_score=float(row[8]),
            raw_data=row[9] if isinstance(row[9], dict) else json.loads(row[9]),
            metadata=row[10] if isinstance(row[10], dict) else json.loads(row[10]),
            detected_at=row[11] if row[11].tzinfo else row[11].replace(tzinfo=timezone.utc),
        ))    
    return events


def log_pipeline_run(conn, status:str = "RUNNING")->str:
    """
    Create a pipeline runs record
        Returns run_id
    """
    run_id=str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_runs (run_id, status) VALUES (%s,%s)",
            (run_id,status)
        )
        
    return run_id    


def update_pipeline_run(conn, run_id: str, status: str,
                        stage_results: dict, events_created: int,
                        briefs_created: int, error_details: str = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs SET
                completed_at   = NOW(),
                status         = %s,
                stage_results  = %s,
                events_created = %s,
                briefs_created = %s,
                error_details  = %s
            WHERE run_id = %s
            """,
            (status, json.dumps(stage_results), events_created,
             briefs_created, error_details, run_id)
        )



def save_investigation_brief(brief, conn) -> str:
    """Insert or update an InvestigationBrief in investigation_briefs."""
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO investigation_briefs
                (brief_id, region_id, time_window_start, time_window_end,
                 confidence_score, contributing_streams, evidence,
                 agent_reasoning, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (brief_id) DO UPDATE SET
                confidence_score    = EXCLUDED.confidence_score,
                agent_reasoning     = EXCLUDED.agent_reasoning,
                status              = EXCLUDED.status,
                updated_at          = NOW()
            """,
            (
                brief.brief_id,
                brief.region_id,
                brief.time_window_start,
                brief.time_window_end,
                brief.confidence_score,
                brief.contributing_streams,
                _json.dumps(brief.evidence),
                brief.agent_reasoning,
                brief.status.value,
            )
        )
    return brief.brief_id