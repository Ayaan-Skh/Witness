# GDELT news intelligent Ingestion
# What is GDELT?
# The global databse of events, language and tone is a machine-reading system that has processed virtually every news article published online since 1979 in 65+ languages, from 150+ countries
# Every article is parsed into structured events using CAMEO(Conflict and Mediation Event Observation)
# taxonomy: 300+ event codes that describe real-world actions like
# "Express intent to use military force" (code 175) or "Displace persons" (code 203).
 
# THE REAL-LIFE ANALOGY
# ─────────────────────
# GDELT is like having a researcher who reads every newspaper in every language
# every day and fills out the same standardized form for each story:
#   "On DATE, ACTOR did ACTION (code 193 = 'Use conventional military force')
#    in LOCATION. The article tone was -8.3 (very negative). Source: Reuters."
 
# You then query those forms. "Show me all forms from Tigray, Ethiopia, in the
# last 90 days where the action was conflict-related and tone crashed."
 
# BIGQUERY COST RULES — READ BEFORE MODIFYING
# ────────────────────────────────────────────
# GDELT tables are 500GB+. A naive full-table scan burns your entire 1TB/month
# free tier in a single query. Every function here follows three rules:
#   1. ALWAYS filter SQLDATE first — it's the partition column (free to filter).
#   2. ALWAYS filter ActionGeo_CountryCode before any other column.
#   3. NEVER SELECT * — name only the columns you need.
 
#  Each query here scans ~100–500MB. A 5-region daily pipeline ≈ 25MB/day.

from __future__ import annotations
import code
import logging 
import os 
from datetime import datetime,date,timedelta
from typing import Optional

log=logging.getLogger("witness.gdelt")


try:
    from google.cloud import bigquery
    from google.oauth2 import service_connect
    BIGQUERY_AVAILABLE=True
except ImportError:
    bigquery=None
    service_account=None
    BIGQUERY_AVAILABLE=False
        
from config import GOOGLE_APPLICATION_CREDENTIALS,GOOGLE_CLOUD_PROJECT    

# CAMEO code groups
# Code directly relevent to to human rights monitoring
# Refrence: https://parusanalytics.com/eventdata/cameo.dir/CAMEO.Manual.1.1b3.pdf

CONFLICT_CAMEO_CODES={
      "18", "180", "181", "182", "183", "185", "186",
    "19", "190", "193", "195", "196",
    "20", "203", "204",
}
DISPLACEMENT_CAMEO_CODES={
    "137", "138", "1381", "139", "223", "2231"
}
CONCERNING_CAMEO_CODES=CONFLICT_CAMEO_CODES | DISPLACEMENT_CAMEO_CODES


# --------------
# CLient
# --------------

def _get_bq_client()->"bigquery.Client":
    if not BIGQUERY_AVAILABLE:
        raise RuntimeError("google-cloud-bigquery not installed. Run pip install google-cloud-bigquery")
    if not GOOGLE_CLOUD_PROJECT:
        raise ValueError('GOOGLE_CLOUD_PROJECT not set in .env')
    
    if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        creds=service_account.Credentials.from_service_account_file(
             GOOGLE_APPLICATION_CREDENTIALS,
             scopes=["https://www.googleapis.com/auth/bigquery.readonly"]
        )
        return bigquery.Client(project=GOOGLE_CLOUD_PROJECT, credentials=creds)
    return bigquery.Client(project=GOOGLE_CLOUD_PROJECT)


def _run_query(sql: str) -> list[dict]:
    """Execute a BigQuery query, log bytes scanned, return rows as dicts."""
    client = _get_bq_client()
    job    = client.query(sql)
    result = job.result()
    mb     = (job.total_bytes_processed or 0) / 1e6
    log.info(f"BigQuery: {mb:.1f} MB scanned")
    return [dict(row) for row in result]                


def _gdelt_date(d: date) -> int:
    """Convert Python date → GDELT integer format (20210315)."""
    return int(d.strftime("%Y%m%d"))
 
 
def _parse_gdelt_date(val) -> date:
    """Convert GDELT integer date (20210315) → Python date."""
    s = str(int(val))
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
 
 
def _location_filter(country_code: str, admin1_code: Optional[str]) -> str:
    if admin1_code:
        return f"ActionGeo_ADM1Code = '{admin1_code}'"
    return f"ActionGeo_CountryCode = '{country_code}'"
 
#------------------------------------- 
# Query Functions
#------------------------------------- 
 
def query_events_by_region(
    country_code:str,
    admin1_code:Optional[str],
    start_date:date,
    end_date:date,
    cameo_root_filter:Optional[list[str]]=None
)-> list[dict]:
    """
    Raw GDELT events for a region + time window.
    Each row: SQLDate, EventCode, EventRootCode, AvgTone, NumMentions, GoldstienScale, Actor1Name, Actor2Name, ActionGeo_FullName, SOURCEURL

    GoldsteinScale: +10 (most cooperative) to -10 (most conflictual).    
    Filtering by cameo_root_filter restricts to specific event families.
    """
    loc=_location_filter(country_code,admin1_code)
    d_from=_gdelt_date(start_date)
    d_to=_gdelt_date(end_date)
    
    cameo_clause=""
    if cameo_root_filter:
        codes=", ".join(f"'{c}'" for c in cameo_root_filter)
        cameo_clause=f"AND EventRootCode IN ({codes})"
    
    sql = f"""
        SELECT
            SQLDATE, Actor1Name, Actor2Name,
            EventCode, EventRootCode, ActionGeo_FullName,
            AvgTone, NumMentions, GoldsteinScale, SOURCEURL
        FROM `gdelt-bq.full.events`
        WHERE SQLDATE BETWEEN {d_from} AND {d_to}
          AND {loc}
          {cameo_clause}
        ORDER BY SQLDATE DESC
        LIMIT 10000
    """
    return _run_query(sql)   

def query_tone_timeseries(
    country_code:str,
    admin1_code:Optional[str],
    start_date:date,
    end_date:date
)->list[dict]:
    """
    Daily mention weighted average tone for a region
    Each row:{event_date,avg_tone,article_count,event_count}
    
    Avg tone range: ~ -100 (maximum negative) to +100
    Typical normal range: -5 to +3
    Crash to -15 or below = String negative anomaly signal
    
    Weighted by numMentions so widely covered events count more than obsecure ones
    """
    
    loc= _location_filter(country_code,admin1_code)
    d_from=_gdelt_date(start_date)
    d_to=_gdelt_date(end_date)
    
    sql=f"""
        SELECT
            SQLDATE AS event_date,
            SUM(AvgTone * NumMentions) / NULLIF(SUM(NUMMentions),0) AS avg_tone,
            SUM(NumMentions) AS article_count,
            Count(*) AS events_count
        FROM `gdelt-bq.full.events`
        WHERE SQLDATE BETWEEN {d_from} AND {d_to} AND {loc}
        GROUP BY SQLDATE
        OORDER BY SQLDATE ASC     
    """
    rows=_run_query(sql=sql)
    return [
        {
            "event_date":_parse_gdelt_date(row['event_date']),
            "avg_tone":float(row["avg_tone"]) if row['avg_tone'] is not None else None,
            "article_count": int(row['article_count']),
            "event_count":int(row['event_count']),
            
        }
        for row in rows
    ]
    

def query_volume_timeseries(
    country_code: str,
    admin1_code: Optional[str],
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Daily article mention volume for a region.
    Each row: { event_date, mention_count, event_count }
 
    mention_count = sum of NumMentions (total article references).
    A sudden DROP to near-zero is the "communication blackout" signal —
    as significant as a spike because silence can mean access denial
    or infrastructure destruction.
    """
    loc    = _location_filter(country_code, admin1_code)
    d_from = _gdelt_date(start_date)
    d_to   = _gdelt_date(end_date)
 
    sql = f"""
        SELECT
            SQLDATE          AS event_date,
            SUM(NumMentions) AS mention_count,
            COUNT(*)         AS event_count
        FROM `gdelt-bq.full.events`
        WHERE SQLDATE BETWEEN {d_from} AND {d_to}
          AND {loc}
        GROUP BY SQLDATE
        ORDER BY SQLDATE ASC
    """
    rows = _run_query(sql)
    return [
        {
            "event_date":    _parse_gdelt_date(row["event_date"]),
            "mention_count": int(row["mention_count"]),
            "event_count":   int(row["event_count"]),
        }
        for row in rows
    ]
    
def get_top_themes(
    country_code: str,
    admin1_code: Optional[str],
    start_date: date,
    end_date: date,
    top_n: int = 20,
) -> list[dict]:
    """
    Most frequent CAMEO event codes in a region + window.
    Each row: { cameo_code, event_count, mention_count, avg_tone,
                avg_goldstein, is_concerning }
 
    is_concerning = True if the code is in CONCERNING_CAMEO_CODES.
    Used by the agent to explain WHAT drove a tone crash, not just that
    a statistical anomaly occurred.
    """
    loc    = _location_filter(country_code, admin1_code)
    d_from = _gdelt_date(start_date)
    d_to   = _gdelt_date(end_date)
 
    sql = f"""
        SELECT
            EventCode           AS cameo_code,
            COUNT(*)            AS event_count,
            SUM(NumMentions)    AS mention_count,
            AVG(AvgTone)        AS avg_tone,
            AVG(GoldsteinScale) AS avg_goldstein
        FROM `gdelt-bq.full.events`
        WHERE SQLDATE BETWEEN {d_from} AND {d_to}
          AND {loc}
        GROUP BY EventCode
        ORDER BY mention_count DESC
        LIMIT {top_n}
    """
    rows = _run_query(sql)
    return [
        {
            "cameo_code":    row["cameo_code"],
            "event_count":   int(row["event_count"]),
            "mention_count": int(row["mention_count"]),
            "avg_tone":      float(row["avg_tone"]) if row["avg_tone"] is not None else None,
            "avg_goldstein": float(row["avg_goldstein"]) if row["avg_goldstein"] is not None else None,
            "is_concerning": row["cameo_code"] in CONCERNING_CAMEO_CODES,
        }
        for row in rows
    ]    
    
def query_source_diversity(
    country_code: str,
    admin1_code: Optional[str],
    start_date: date,
    end_date: date,
) -> dict:
    """
    Count of distinct news source domains covering a region in a window.
 
    Low diversity (3 sources vs normal 50) combined with low volume
    is a stronger blackout signal than either metric alone.
    Uses APPROX_COUNT_DISTINCT — BigQuery's cheap probabilistic distinct count.
    """
    loc    = _location_filter(country_code, admin1_code)
    d_from = _gdelt_date(start_date)
    d_to   = _gdelt_date(end_date)
 
    sql = f"""
        SELECT
            APPROX_COUNT_DISTINCT(
                REGEXP_EXTRACT(SOURCEURL, r'https?://([^/]+)')
            ) AS distinct_sources,
            COUNT(*) AS total_events
        FROM `gdelt-bq.full.events`
        WHERE SQLDATE BETWEEN {d_from} AND {d_to}
          AND {loc}
    """
    rows = _run_query(sql)
    if not rows:
        return {"distinct_sources": 0, "total_events": 0}
    return {
        "distinct_sources": int(rows[0]["distinct_sources"] or 0),
        "total_events":     int(rows[0]["total_events"] or 0),
    }    
    
def fill_missing_dates(
    timeseries: list[dict],
    start_date: date,
    end_date: date,
    date_key: str = "event_date",
) -> list[dict]:
    """
    Fills gaps in a timeseries with zero-value rows.
 
    GDELT only returns rows for dates that have data. Gaps must be explicitly
    filled with zeros — NOT forward-filled — because a gap IS the signal.
    Forward-filling would mask the communication blackout we're detecting.
    """
    existing = {row[date_key]: row for row in timeseries}
    fill_template = {k: 0 for k, v in (timeseries[0].items() if timeseries else {}.items())
                     if isinstance(v, (int, float)) and k != date_key}
 
    result = []
    current = start_date
    while current <= end_date:
        if current in existing:
            result.append(existing[current])
        else:
            result.append({date_key: current, **fill_template})
        current += timedelta(days=1)
    return result
 
 
def estimate_query_cost_mb(days: int, query_type: str = "timeseries") -> float:
    """Conservative estimate of MB scanned per query. Used for pre-flight checks."""
    MB_PER_DAY = {"timeseries": 2.5, "events": 8.0, "themes": 3.0}
    return MB_PER_DAY.get(query_type, 5.0) * days
     