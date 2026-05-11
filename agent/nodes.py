"""
LangGraph agent node functions

ARCHITECTURE
Every function here is a node in LangGraph graph.
    - Every node recives full WitnessState
    - They perform exactly 1 thing
    - Return a dict of state fields to update  
    
Only one node (generate_brief) calls the LLM. Everything else is deterministic Python - geometry, statistics, weighted math. This is intentional: LLMs are slow, expensive, and non-deterministic. We use them exactly where language reasoning is genuinely required (synthesizing evidence into a natural language brief) and nowhere else.    
    
Nodes in this file:
1. cluster_anomalies     — Pure geometry. Groups events by location + time
2. score_convergence     — Pure math, Scores clusters by multi-source strength    
"""

from __future__ import annotations
from datetime import date,datetime,timezone
import math
import hashlib    
from config import (
    CONVERGENCE_SCORE_THRESHOLD,
    GEOGRAPHIC_CLUSTER_RADIUS_KM,
    TEMPORAL_CLUSTER_WINDOW_DAYS
)
from normalization.schema import AnomalyEvent,AnomalySource
from agent.state import WitnessState
import logging

log=logging.getLogger("witness.agent")

# -------------------------
# HAVERSINE DISTANCE
# -------------------------

def haversine_km(lat1:float,lng1:float,lat2:float,lng2:float)->float:
    """
    Great circle distance between two points on earths surface. 
    The haversine formula gives the shortest distance over the earths surface between two lat/lng points.
    """
    
    R = 6371.0  # Earth's mean radius in km
 
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
 
    # angular separation between points
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    # Converts angular separation -> arc length.
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c



# ---------------------------------
# Node 1: CLustering Anomalies
# ---------------------------------

def cluster_anomalies(state:WitnessState)->dict:
    """
    Groups anomaly events into geographic + temporal clusters
    
    Clustering algorithm:
    Simple greedy single linkage clustering:
        1. Process events in chronological order
        2. for every event chech all the existing cluster
        3. If the event lies within the cluster centroid of GEOGRAPHIC_CLUSTER_RADIUS_KM and within the TEMPORAL_CLUSTER_WINDOW_DAYS of the clusters time window, add it to that cluster.
        4. Otherwise start new window with this cluster as seed
        
        Returns:
            {"clusters": [...], "reasoning_trace": [...]}    
    """
    events=state["anomaly_events"]
    
    if not events:
        return{
            "clusters":[],
            "reasoning_trace":["cluster_anomalies:no anomalies to cluster"],
        }
    # Sort chronologically — ensures earlier events seed clusters first,
    # giving clusters a stable identity based on the first event observed.
    sorted_events = sorted(events, key=lambda e: e.timestamp)
 
    clusters: list[dict] = []
    
    for event in sorted_events:
        placed=False
        
        for cluster in clusters:
                        # ── Geographic check ──────────────────────────────────────
            dist_km = haversine_km(
                event.lat, event.lng,
                cluster["centroid_lat"], cluster["centroid_lng"],
            )
            if dist_km > GEOGRAPHIC_CLUSTER_RADIUS_KM:
                continue
 
            # ── Temporal check ────────────────────────────────────────
            days_from_start = (event.timestamp - cluster["time_start"]).days
            days_to_end     = (cluster["time_end"] - event.timestamp).days
 
            # Event must fall within [time_start - window, time_end + window]
            if days_from_start < -TEMPORAL_CLUSTER_WINDOW_DAYS:
                continue
            if days_to_end < -TEMPORAL_CLUSTER_WINDOW_DAYS:
                continue
 
            # ── Add to cluster ────────────────────────────────────────
            cluster["events"].append(event)
            cluster["sources"] = list({e.source.value for e in cluster["events"]})
 
            # Update time window to span all events
            cluster["time_start"] = min(cluster["time_start"], event.timestamp)
            cluster["time_end"]   = max(cluster["time_end"],   event.timestamp)
 
            # Update centroid as running mean of all event locations
            n = len(cluster["events"])
            cluster["centroid_lat"] = (
                (cluster["centroid_lat"] * (n - 1) + event.lat) / n
            )
            cluster["centroid_lng"] = (
                (cluster["centroid_lng"] * (n - 1) + event.lng) / n
            )
 
            placed = True
            break   # Each event goes into at most one cluster
 
        if not placed:
            # ----------------------- Seed a new cluster -----------------------
            cluster_id = _make_cluster_id(event)
            clusters.append({
                "cluster_id":   cluster_id,
                "region_id":    event.region_id,
                "centroid_lat": event.lat,
                "centroid_lng": event.lng,
                "time_start":   event.timestamp,
                "time_end":     event.timestamp,
                "events":       [event],
                "sources":      [event.source.value],
            })
 
    trace = (
        f"cluster_anomalies: {len(events)} events → {len(clusters)} clusters "
        f"(radius={GEOGRAPHIC_CLUSTER_RADIUS_KM}km, window={TEMPORAL_CLUSTER_WINDOW_DAYS}d)"
    )
    log.info(trace)
 
    return {"clusters": clusters, "reasoning_trace": [trace]}
    
def _make_cluster_id(seed_event: AnomalyEvent) -> str:
    """
    Generates a stable, human-readable cluster ID from the seed event.
    Format: {region_id}_{YYYYMMDD}_{short_hash}
    Example: eth_tigray_20211115_a3f7c2b1
    """
    date_str  = seed_event.timestamp.strftime("%Y%m%d")
    hash_input = f"{seed_event.region_id}_{date_str}_{seed_event.event_id}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return f"{seed_event.region_id}_{date_str}_{short_hash}"    



# -------------------------------------
# Node 2: SOURCE CONVERGENCE NODE
# -------------------------------------

def score_convergence(state:WitnessState)->dict:
    """
    Scores each anomaly cluster for cross-source convergence and decides
    which clusters are strong enough to trigger brief generation.

    For every cluster in state["clusters"] it:
      - Computes a stream diversity score based on how many distinct sources
        are present in the cluster.
      - Computes the mean intensity score across all events in the cluster.
      - Applies a time-decay factor so that older clusters contribute less,
        scaled by TEMPORAL_CLUSTER_WINDOW_DAYS.
    These three factors are combined into a single composite convergence
    score in [0, 1]. Any cluster whose score exceeds
    CONVERGENCE_SCORE_THRESHOLD is added to "briefs_to_generate".
    
    -----------------------------------
    COMPOSITE SCORE COMPONENTS
    -----------------------------------
    1. stream_diversity  (weight 0.50) — how many distinct sources?
       The most important component. Convergence is the core thesis.
 
    2. intensity         (weight 0.30) — how strong are the individual signals?
       Average intensity_score across all events in the cluster.
 
    3. recency           (weight 0.20) — how recent is the most recent event?
       Decays from 1.0 (today) to 0.0 (21+ days ago).
       Recent events warrant more urgent attention.
 
    Clusters scoring above CONVERGENCE_SCORE_THRESHOLD (default 0.55)
    are added to briefs_to_generate.

    Returns:
        {
            "convergence_scores": {cluster_id: score},
            "briefs_to_generate": [cluster_ids ready for brief generation],
            "reasoning_trace":    [human-readable scoring summary],
        }
    """
    clusters = state["clusters"]
 
    if not clusters:
        return {
            "convergence_scores": {},
            "briefs_to_generate": [],
            "reasoning_trace":    ["score_convergence: no clusters to score"],
        }
 
    now = datetime.now(timezone.utc)
    scores: dict[str, float] = {}
    to_generate: list[str]   = []
    score_details: list[str] = []
 
    for cluster in clusters:
        events = cluster["events"]
 
        # ---------------------- 1. Stream diversity score (non-linear) ----------------------
        n_sources = len(set(e.source.value for e in events))
        stream_diversity = {1: 0.20, 2: 0.60, 3: 1.00}.get(n_sources, 1.00)
 
        # ---------------------- 2. Mean intensity score ----------------------
        intensity = sum(e.intensity_score for e in events) / len(events)
 
        # ---------------------- 3. Recency decay ----------------------
        # Most recent event in cluster
        latest_event_time = max(e.timestamp for e in events)
        # Make 'now' timezone-aware if needed
        if latest_event_time.tzinfo is None:
            latest_event_time = latest_event_time.replace(tzinfo=timezone.utc)
 
        days_ago = max(0.0, (now - latest_event_time).total_seconds() / 86400)
        recency  = max(0.0, 1.0 - (days_ago / TEMPORAL_CLUSTER_WINDOW_DAYS))
 
        # ---------------------- Composite ---------------------- 
        composite = (
            0.50 * stream_diversity +
            0.30 * intensity        +
            0.20 * recency
        )
        composite = round(min(composite, 1.0), 4)
 
        scores[cluster["cluster_id"]] = composite
 
        fired = composite >= CONVERGENCE_SCORE_THRESHOLD
        if fired:
            to_generate.append(cluster["cluster_id"])
 
        detail = (
            f"  {cluster['cluster_id']}: score={composite:.3f} "
            f"[diversity={stream_diversity:.2f}({n_sources}src) "
            f"intensity={intensity:.2f} recency={recency:.2f}] "
            f"{'→ BRIEF' if fired else '→ below threshold'}"
        )
        score_details.append(detail)
        log.info(detail)
 
    trace = (
        f"score_convergence: {len(clusters)} clusters scored, "
        f"{len(to_generate)} above threshold ({CONVERGENCE_SCORE_THRESHOLD})\n"
        + "\n".join(score_details)
    )
 
    return {
        "convergence_scores": scores,
        "briefs_to_generate": to_generate,
        "reasoning_trace":    [trace],
    }     





