"""Read-only DB queries for the REST API."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from psycopg2.extras import RealDictCursor

from config import MONITORED_REGIONS, REGIONS_BY_ID


def score_to_tier(score: float, n_streams: int) -> str:
    if score >= 0.80 and n_streams >= 3:
        return "HIGH"
    if score < 0.40 or n_streams <= 1:
        return "LOW"
    return "MEDIUM"


def _tier_sql_expr() -> str:
    return """
        CASE
            WHEN b.confidence_score >= 0.80
                 AND cardinality(b.contributing_streams) >= 3 THEN 'HIGH'
            WHEN b.confidence_score < 0.40
                 OR cardinality(b.contributing_streams) <= 1 THEN 'LOW'
            ELSE 'MEDIUM'
        END
    """


def _row_to_brief_summary(row: dict) -> dict[str, Any]:
    streams = list(row["contributing_streams"] or [])
    tier = score_to_tier(float(row["confidence_score"]), len(streams))
    return {
        "brief_id": str(row["brief_id"]),
        "region_id": row["region_id"],
        "time_window_start": row["time_window_start"],
        "time_window_end": row["time_window_end"],
        "confidence_score": float(row["confidence_score"]),
        "confidence_tier": tier,
        "contributing_streams": streams,
        "status": row["status"],
        "created_at": row["created_at"],
    }


def list_briefs(
    conn,
    *,
    page: int = 1,
    page_size: int = 20,
    confidence_tier: Optional[str] = None,
    region_id: Optional[str] = None,
    status: Optional[str] = None,
) -> tuple[list[dict], int]:
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size

    where = ["1=1"]
    params: list[Any] = []

    if region_id:
        where.append("b.region_id = %s")
        params.append(region_id)
    if status:
        where.append("b.status = %s")
        params.append(status)
    if confidence_tier:
        where.append(f"({_tier_sql_expr()}) = %s")
        params.append(confidence_tier)

    where_sql = " AND ".join(where)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT COUNT(*) AS n FROM investigation_briefs b WHERE {where_sql}",
            params,
        )
        total = int(cur.fetchone()["n"])

        cur.execute(
            f"""
            SELECT b.brief_id, b.region_id, b.time_window_start, b.time_window_end,
                   b.confidence_score, b.contributing_streams, b.status, b.created_at
            FROM investigation_briefs b
            WHERE {where_sql}
            ORDER BY b.created_at DESC
            LIMIT %s OFFSET %s
            """,
            [*params, page_size, offset],
        )
        rows = cur.fetchall()

    items = [_row_to_brief_summary(r) for r in rows]
    return items, total


def get_brief(conn, brief_id: str) -> Optional[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT brief_id, region_id, time_window_start, time_window_end,
                   confidence_score, contributing_streams, evidence,
                   agent_reasoning, status, reviewer_notes, created_at
            FROM investigation_briefs
            WHERE brief_id = %s
            """,
            (brief_id,),
        )
        row = cur.fetchone()
    if not row:
        return None

    evidence = row["evidence"]
    if isinstance(evidence, str):
        evidence = json.loads(evidence)

    out = _row_to_brief_summary(row)
    out.update(
        {
            "evidence": evidence or {},
            "agent_reasoning": row["agent_reasoning"] or "",
            "historical_context": "",
            "reviewer_notes": row["reviewer_notes"] or "",
        }
    )
    return out


def list_regions(conn) -> list[dict]:
    counts: dict[str, dict[str, int]] = {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT region_id, source, COUNT(*)::int AS n
            FROM anomaly_events
            GROUP BY region_id, source
            """
        )
        for row in cur.fetchall():
            rid = row["region_id"]
            counts.setdefault(rid, {"SATELLITE": 0, "GDELT": 0, "PROCUREMENT": 0})
            src = row["source"]
            if src in counts[rid]:
                counts[rid][src] = row["n"]

    regions = []
    for cfg in MONITORED_REGIONS:
        lat, lng = cfg.centroid()
        bbox = cfg.bbox
        regions.append(
            {
                "region_id": cfg.region_id,
                "name": cfg.name,
                "country_code": cfg.country_code,
                "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                "centroid_lat": lat,
                "centroid_lng": lng,
                "anomaly_counts": counts.get(
                    cfg.region_id,
                    {"SATELLITE": 0, "GDELT": 0, "PROCUREMENT": 0},
                ),
            }
        )
    return regions


def list_anomalies(
    conn,
    *,
    page: int = 1,
    page_size: int = 20,
    source: Optional[str] = None,
    region_id: Optional[str] = None,
    min_intensity: Optional[float] = None,
) -> tuple[list[dict], int]:
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size

    where = ["1=1"]
    params: list[Any] = []

    if source:
        where.append("source = %s")
        params.append(source)
    if region_id:
        where.append("region_id = %s")
        params.append(region_id)
    if min_intensity is not None:
        where.append("intensity_score >= %s")
        params.append(min_intensity)

    where_sql = " AND ".join(where)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT COUNT(*) AS n FROM anomaly_events WHERE {where_sql}",
            params,
        )
        total = int(cur.fetchone()["n"])

        cur.execute(
            f"""
            SELECT event_id, source, region_id, country_code, lat, lng,
                   timestamp, signal_type, intensity_score, raw_data, metadata, detected_at
            FROM anomaly_events
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """,
            [*params, page_size, offset],
        )
        rows = cur.fetchall()

    items = []
    for row in rows:
        raw = row["raw_data"]
        meta = row["metadata"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(meta, str):
            meta = json.loads(meta)
        items.append(
            {
                "event_id": str(row["event_id"]),
                "source": row["source"],
                "region_id": row["region_id"],
                "country_code": row["country_code"].strip()
                if row["country_code"]
                else REGIONS_BY_ID.get(row["region_id"], MONITORED_REGIONS[0]).country_code,
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "timestamp": row["timestamp"],
                "signal_type": row["signal_type"],
                "intensity_score": float(row["intensity_score"]),
                "raw_data": raw or {},
                "metadata": meta or {},
                "detected_at": row["detected_at"],
            }
        )
    return items, total
