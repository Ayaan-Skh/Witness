"""
LangGraph Agent State

The langgraph runtime passes this TypeDict to every node in the garph. Each node recives full state, does exactly one thing and returns dict of fields to update. LangGraph merges the update back into the state before passing it to the next node
"""

from __future__ import annotations
from ast import List
from typing import TypedDict,Annotated
import operator

from normalization.schema import AnomalyEvent,InvestigationBrief

class WitnessState(TypedDict):
    """
    Shared state flowing through entire langGraph pipeline.
    
    field lifecycle:
        anomaly_events      - Set by caller before graph invocation
        clusters            - written by cluster anomalies node
        convergence_score   - written by score_convergence node
        briefs_to_generate  - written by score_convergence node 
        historical_context  - written by retrieve_historical_context node
        generated_briefs    - written by generate_brief node
        reasoning_trace     - append by every node
        error_log           - append by non fatal error
    """
    # ------------------ INPUT ------------------
    # The full list of AnomalyEvents from all three pipelines for this run window. Populated by the caller before graph.invoke().
    anomaly_events:List[AnomalyEvent]
    
    # ------------------ CLuster Node output ------------------
    # Events grouped by geographic proximity +temporal overlap
    # Each cluster dict has:
    # {
        #   "cluster_id":   str (e.g. "eth_tigray_20211101"),
        #   "region_id":    str,
        #   "centroid_lat": float,
        #   "centroid_lng": float,
        #   "time_start":   datetime,
        #   "time_end":     datetime,
        #   "events":       list[AnomalyEvent],
        #   "sources":      list[str]  (e.g. ["SATELLITE", "GDELT"])
    # }
    clusters:list[dict]
    
    # ------------------ Convergence scoring node output ------------------
    # Maps cluster_id → composite convergence score (0.0–1.0).
    # Scores above CONVERGENCE_SCORE_THRESHOLD go into briefs_to_generate.
    convergence_score:dict[str,float]
    
    # ------------------ Threshold filter output ------------------
    # cluster Ids that have cleared convergence threshold and should have a investigation brief generated
    briefs_to_generate:list[str]
    
    # ------------------ Memory retrival node output ------------------
    # maps cluster_id -> natural language summary of similar historical cases retrived from chromaDB vector memory
    historical_context:dict[str,str]
    
    # ------------------ brief generation node output ------------------
    # Fully generated InvestigationBrief objects ready to persist
    generated_briefs:list[InvestigationBrief]
    
    # ------------------ Complete flow log from node to node operations ------------------  
    # Only upload logs. Every node shoudl have one line entry about what it did and what it decided. This is printed all for a full run and stored in pipeline_run table for review later.
    # Using Annoted[list[str],operator.add] tells langGraph to merge these by concatenation rather than replacement when nodes return updates 
    reasoning_trace:Annotated[list[str],operator.add]
    
    # ------------------ Error Logs ------------------
    # Non fatal error(eg: memory retrival failed for one cluster)
    # Pipeline continues and error are viewed in final summary
    error_log:Annotated[list[str],operator.add]
    
def make_initial_state(anomaly_events:list[AnomalyEvent])->WitnessState:
    """
    Creates a fresh state dict for a new graph invocation.
 
    All list/dict fields initialized to empty — nodes are responsible
    for populating them. The only pre-populated field is anomaly_events.
    """    
    return WitnessState(
        anomaly_events=anomaly_events,
        clusters=[],
        convergence_scores={},
        briefs_to_generate=[],
        historical_context={},
        generated_briefs=[],
        reasoning_trace=[],
        error_log=[],
    )