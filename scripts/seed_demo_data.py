"""
Insert sample regions (if needed), anomaly events, and investigation briefs
so the dashboard has data to display.

Run from repo root (Postgres must be up):
  python scripts/seed_demo_data.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from db import get_db, init_pool, seed_regions
from normalization.schema import BriefStatus

NOW = datetime.now(timezone.utc)


def seed():
    init_pool()
    seed_regions()

    briefs = [
        {
            "brief_id": str(uuid.uuid4()),
            "region_id": "eth_tigray",
            "time_window_start": NOW - timedelta(days=14),
            "time_window_end": NOW - timedelta(days=2),
            "confidence_score": 0.87,
            "contributing_streams": ["SATELLITE", "GDELT", "PROCUREMENT"],
            "status": "DRAFT",
            "evidence": {
                "SATELLITE": {
                    "summary": "NDVI drop and land-cover change across multiple tiles near Mekelle.",
                    "change_score_max": 0.72,
                },
                "GDELT": {
                    "summary": "Tone crash and conflict CAMEO spike in regional media.",
                    "tone_zscore_min": -3.4,
                    "top_concerning_themes": [
                        {"cameo_code": "193", "mention_count": 420},
                        {"cameo_code": "194", "mention_count": 180},
                    ],
                },
                "PROCUREMENT": {
                    "summary": "Military-category spend 3.2σ above 12-month baseline.",
                    "spend_zscore_max": 3.2,
                },
            },
            "agent_reasoning": (
                "### WHAT EACH STREAM FOUND\n"
                "Satellite imagery shows sustained vegetation loss. GDELT reports elevated conflict framing. "
                "Procurement data shows anomalous military spend.\n\n"
                "### CONFIDENCE ASSESSMENT\nHIGH\n\n"
                "### RECOMMENDED NEXT STEPS\n"
                "Cross-check satellite tiles with independent reporting; verify procurement contract metadata."
            ),
        },
        {
            "brief_id": str(uuid.uuid4()),
            "region_id": "ukr_mariupol",
            "time_window_start": NOW - timedelta(days=21),
            "time_window_end": NOW - timedelta(days=5),
            "confidence_score": 0.71,
            "contributing_streams": ["SATELLITE", "GDELT"],
            "status": "REVIEWED",
            "evidence": {
                "SATELLITE": {
                    "summary": "Structure-change signal cluster near industrial zone.",
                    "change_score_max": 0.65,
                },
                "GDELT": {
                    "summary": "Elevated coverage volume with negative tone shift.",
                    "volume_zscore_max": 2.9,
                },
            },
            "agent_reasoning": (
                "Two-source convergence (satellite + news). Medium-high confidence pending procurement corroboration.\n"
                "### CONFIDENCE ASSESSMENT\nMEDIUM"
            ),
            "reviewer_notes": "Satellite tiles reviewed — change pattern consistent with prior siege documentation.",
        },
        {
            "brief_id": str(uuid.uuid4()),
            "region_id": "mmr_rakhine",
            "time_window_start": NOW - timedelta(days=30),
            "time_window_end": NOW - timedelta(days=10),
            "confidence_score": 0.38,
            "contributing_streams": ["GDELT"],
            "status": "DRAFT",
            "evidence": {
                "GDELT": {
                    "summary": "Single-source tone anomaly; no satellite or procurement corroboration yet.",
                    "tone_zscore_min": -2.1,
                },
            },
            "agent_reasoning": "Single-stream signal only. Treat as low confidence until additional sources converge.",
        },
        {
            "brief_id": str(uuid.uuid4()),
            "region_id": "ind_delhi",
            "time_window_start": NOW - timedelta(days=7),
            "time_window_end": NOW - timedelta(days=1),
            "confidence_score": 0.62,
            "contributing_streams": ["SATELLITE", "PROCUREMENT"],
            "status": "PUBLISHED",
            "evidence": {
                "SATELLITE": {"summary": "Localized thermal anomaly detected.", "change_score_max": 0.41},
                "PROCUREMENT": {"summary": "Emergency contract pattern in medical supplies.", "spend_zscore_max": 2.6},
            },
            "agent_reasoning": "Published after human review. Two-source convergence without GDELT corroboration.",
        },
    ]

    events_sql = """
        INSERT INTO anomaly_events
            (event_id, source, region_id, country_code, location, lat, lng,
             timestamp, signal_type, intensity_score, raw_data, metadata)
        VALUES (
            %s, %s, %s, %s,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
            %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (event_id) DO NOTHING
    """

    with get_db() as conn:
        with conn.cursor() as cur:
            for b in briefs:
                cur.execute(
                    """
                    INSERT INTO investigation_briefs
                        (brief_id, region_id, time_window_start, time_window_end,
                         confidence_score, contributing_streams, evidence,
                         agent_reasoning, status, reviewer_notes)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (brief_id) DO NOTHING
                    """,
                    (
                        b["brief_id"],
                        b["region_id"],
                        b["time_window_start"],
                        b["time_window_end"],
                        b["confidence_score"],
                        b["contributing_streams"],
                        json.dumps(b["evidence"]),
                        b["agent_reasoning"],
                        b["status"],
                        b.get("reviewer_notes", ""),
                    ),
                )

                region = b["region_id"]
                country = {"eth_tigray": "ET", "ukr_mariupol": "UA", "mmr_rakhine": "MM", "ind_delhi": "IN"}.get(
                    region, "XX"
                )
                lat, lng = {
                    "eth_tigray": (14.0, 38.5),
                    "ukr_mariupol": (47.1, 37.5),
                    "mmr_rakhine": (19.8, 93.9),
                    "ind_delhi": (28.6, 77.2),
                }[region]

                for src in b["contributing_streams"]:
                    cur.execute(
                        events_sql,
                        (
                            str(uuid.uuid4()),
                            src,
                            region,
                            country,
                            lng,
                            lat,
                            lat,
                            lng,
                            b["time_window_end"],
                            "LAND_COVER_CHANGE" if src == "SATELLITE" else "TONE_CRASH",
                            0.7,
                            json.dumps({}),
                            json.dumps({}),
                        ),
                    )

    print(f"Seeded {len(briefs)} investigation briefs (+ anomaly events).")


if __name__ == "__main__":
    seed()
