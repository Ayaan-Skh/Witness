"""In-memory demo payloads when PostgreSQL is unavailable."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import MONITORED_REGIONS

NOW = datetime.now(timezone.utc)


def _regions() -> list[dict]:
    out = []
    for r in MONITORED_REGIONS:
        lat, lng = r.centroid()
        bbox = r.bbox
        out.append(
            {
                "region_id": r.region_id,
                "name": r.name,
                "country_code": r.country_code,
                "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                "centroid_lat": lat,
                "centroid_lng": lng,
                "anomaly_counts": {"SATELLITE": 2, "GDELT": 3, "PROCUREMENT": 1},
            }
        )
    return out


def _briefs() -> list[dict]:
    return [
        {
            "brief_id": "demo-eth-tigray-001",
            "region_id": "eth_tigray",
            "time_window_start": NOW - timedelta(days=14),
            "time_window_end": NOW - timedelta(days=2),
            "confidence_score": 0.87,
            "confidence_tier": "HIGH",
            "contributing_streams": ["SATELLITE", "GDELT", "PROCUREMENT"],
            "status": "DRAFT",
            "created_at": NOW - timedelta(days=1),
            "evidence": {
                "SATELLITE": {"summary": "NDVI drop and land-cover change near Mekelle.", "change_score_max": 0.72},
                "GDELT": {
                    "summary": "Tone crash and conflict CAMEO spike.",
                    "top_concerning_themes": [{"cameo_code": "193", "mention_count": 420}],
                },
                "PROCUREMENT": {"summary": "Military spend 3.2σ above baseline.", "spend_zscore_max": 3.2},
            },
            "agent_reasoning": "Three-source convergence. HIGH confidence pending human review.",
            "historical_context": "",
            "reviewer_notes": "",
        },
        {
            "brief_id": "demo-ukr-mariupol-002",
            "region_id": "ukr_mariupol",
            "time_window_start": NOW - timedelta(days=21),
            "time_window_end": NOW - timedelta(days=5),
            "confidence_score": 0.71,
            "confidence_tier": "MEDIUM",
            "contributing_streams": ["SATELLITE", "GDELT"],
            "status": "REVIEWED",
            "created_at": NOW - timedelta(days=3),
            "evidence": {
                "SATELLITE": {"summary": "Structure-change cluster near industrial zone."},
                "GDELT": {"summary": "Elevated coverage with negative tone."},
            },
            "agent_reasoning": "Two-source convergence. MEDIUM confidence.",
            "historical_context": "",
            "reviewer_notes": "Satellite tiles reviewed.",
        },
        {
            "brief_id": "demo-mmr-rakhine-003",
            "region_id": "mmr_rakhine",
            "time_window_start": NOW - timedelta(days=30),
            "time_window_end": NOW - timedelta(days=10),
            "confidence_score": 0.38,
            "confidence_tier": "LOW",
            "contributing_streams": ["GDELT"],
            "status": "DRAFT",
            "created_at": NOW - timedelta(days=7),
            "evidence": {"GDELT": {"summary": "Single-source tone anomaly only."}},
            "agent_reasoning": "Single-stream signal. LOW confidence.",
            "historical_context": "",
            "reviewer_notes": "",
        },
        {
            "brief_id": "demo-ind-delhi-004",
            "region_id": "ind_delhi",
            "time_window_start": NOW - timedelta(days=7),
            "time_window_end": NOW - timedelta(days=1),
            "confidence_score": 0.62,
            "confidence_tier": "MEDIUM",
            "contributing_streams": ["SATELLITE", "PROCUREMENT"],
            "status": "PUBLISHED",
            "created_at": NOW - timedelta(hours=12),
            "evidence": {
                "SATELLITE": {"summary": "Localized thermal anomaly."},
                "PROCUREMENT": {"summary": "Emergency medical contract pattern."},
            },
            "agent_reasoning": "Published after review. Two-source convergence.",
            "historical_context": "",
            "reviewer_notes": "",
        },
    ]


BRIEFS = _briefs()
REGIONS = _regions()
BRIEF_BY_ID = {b["brief_id"]: b for b in BRIEFS}
