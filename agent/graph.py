"""
LangGraph Agent Graph



-------------------
GRAPH STRUCTURE
-------------------
START
  → cluster_anomalies          (pure geometry — groups events)
  → score_convergence          (pure math — scores multi-source strength)
  → [conditional routing]
      if briefs_to_generate is empty → END
      else → retrieve_historical_context
  → retrieve_historical_context (ChromaDB vector memory lookup)
  → generate_brief              (Gemini LLM — only node that calls LLM)
  → save_briefs_to_db           (persist InvestigationBrief objects)
  → END

"""
from __future__ import annotations
import logging
from datetime import date, datetime, timezone
from math import e
from re import L
import trace
from typing import Literal

from langchain_core.messages import SystemMessage

log=logging.getLogger("witness.agent.graph")

try:
    from langgraph.graph import StateGraph, END, START
    LANGGRAPH_AVAILABLE=True
except:
    LANGGRAPH_AVAILABLE=False
    StateGraph=False
    END="END"
    START="START"
    
from agent.state import WitnessState,make_initial_state
from agent.nodes import cluster_anomalies, score_convergence
from normalization.schema import AnomalyEvent    
        
# --------------------------------    
# STUB Nodes
# --------------------------------    
def retrieve_historical_context(state:WitnessState)->dict:
    """
    Queries chromaDB for top k chunks for similar related cases for each cluster in brief_to_generate
    
    For each cluster, embeds a structured text representation that retrives semantcally similar past InvestigationBriefs. The formatted context string is passed to the Gemini brief generation node.
    """ 
    from memory.store import retrieve_similar_cases, format_historical_context

    cluster_map={c["cluster_id"]: c for c in state["clusters"]}
    context:dict[str,str]={}
    errors:list[str]=[]
    
    for cluster_id in state["briefs_to_generate"]:
        cluster=cluster_map.get(cluster_id)
        if not cluster:
            context[cluster_id]="CLuster not found in state"
            continue
        try:
            events=cluster.get("events",[])
            cases=retrieve_similar_cases(
                cluster=cluster,
                events=events,
                region_id=cluster.get("region_id")
            )
            context[cluster_id]=format_historical_context(cases)
            log.info(
                f"Memory: {len(cases)} historical cases retrieved "
                f"for cluster {cluster_id}"
            )
        except Exception as e:
            msg = f"Memory retrieval failed for {cluster_id}: {e}"
            log.warning(msg)
            errors.append(msg)
            context[cluster_id] = "Historical context unavailable (retrieval error)."
    trace = (
        f"retrieve_historical_context: retrieved context for "
        f"{len(context)} clusters"
    )
    return {
        "historical_context":context,
        "reasoning_trace":trace,
        "error_log":errors
    }   

def generate_brief(state:WitnessState)->dict:
    """
    Calls Gemini to generate InvestigationBrief objects for each cluster in brief_to_generate list.
    
    This is the only node that makes an LLm call, everything else is deterministic math and geometry. This is intentional: LLMs are slow, expensive, and non-deterministic. We use them exactly where language reasoning is genuinely required (synthesizing evidence into a natural language brief) and nowhere else.    
    """        
    import re
    import uuid
    from normalization.schema import InvestigationBrief,BriefStatus,ConfidenceTier
    from datetime import timezone as tz
    from agent.prompts import build_brief_prompt, BRIEF_GENERATION_SYSTEM_PROMPT
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SyatemMessage
    except ImportError:    
        log.error("langchain-google-genai not installed")
        return {
            "generated_briefs": [],
            "reasoning_trace":  ["generate_brief: langchain-google-genai not installed"],
            "error_log":        ["langchain-google-genai not installed"],
        }
    
    from config import GEMINI_API_KEY,LLM_MODEL,LLM_TEMPERATURE,LLM_MAX_TOKENS
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set")
        return {
            "generated_briefs":[],
            "reasoning_trace":["generated_briefs: GEMINI_API_KEY not set"],
            "error_log":["GEMINI_API_KEY not configured"]
        }
    
    llm=ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS
    )    
    
    cluster_map={c['cluster_id']:c for c in state['clusters']}
    scores=state.get("convergence_score")
    hist_ctx=state.get("historical_context")
    
    briefs:list[InvestigationBrief]=[]
    errors:list[str]=[]
    
    for cluster_id in state['briefs_to_generate']:
        cluster=cluster_map.get({cluster_id})
        if not cluster:
            continue
        
        events=cluster.get("events",[])
        convergence_score=scores.get(cluster_id,0.0)
        historical_ctx=hist_ctx.get(cluster_id,"No historical context available")
        
        try:
            prompt=build_brief_prompt(
                cluster=cluster,
                events=events,
                convergence_score=convergence_score,
                historical_context=historical_ctx
            )
            
            messages=[
                SystemMessage(contemt=BRIEF_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response=llm.invoke(messages)
            text=response.content
            
            # Parse confidence tier ----------
            tier=ConfidenceTier.MEDIUM  #Safe default
            for t in ["HIGH","MEDIUM","LOW"]:
                if f"### CONFIDENCE ASSESSMENT" in text and t in text.split("### CONFIDENCE ASSESMENT")[-1][:200]:
                    tier=ConfidenceTier(t)
                    break
            
            # MAP CONVERGENCE score to confidence if LLM parse fails  
            if tier == ConfidenceTier.MEDIUM:
                n_sources = len(set(cluster.get("sources", [])))
                if convergence_score >= 0.80 and n_sources == 3:
                    tier = ConfidenceTier.HIGH
                elif convergence_score < 0.40 or n_sources == 1:
                    tier = ConfidenceTier.LOW
             
            t_start = cluster["time_start"]
            t_end   = cluster["time_end"]
            if not hasattr(t_start, "tzinfo") or t_start.tzinfo is None:
                from datetime import datetime
                t_start = datetime.fromisoformat(str(t_start)).replace(tzinfo=tz.utc)
            if not hasattr(t_end, "tzinfo") or t_end.tzinfo is None:
                from datetime import datetime
                t_end = datetime.fromisoformat(str(t_end)).replace(tzinfo=tz.utc)
  
            brief=InvestigationBrief(
                brief_id=str(uuid.uuid4()),
                region_id=cluster["region_id"],
                time_window_start=t_start,
                time_window_end=t_end,
                confidence_score=round(convergence_score,4),
                confidence_tier=tier,
                contributing_streams=list(set(cluster.get("sources",[]))),
                evidence={
                    src:{"events":[e.to_dict() for e in events if e.source.value == src]} for src in set(cluster.get("sources",[]))
                },
                agent_reasoning=text,
                historical_context=historical_ctx,
                status=BriefStatus.DRAFT
            )   
            briefs.append(brief)
            log.info(
                f"Brief generated:{cluster_id} |"
                f"tier={tier.value} | score={convergence_score:.3f}"
            )
            
            # Store in chroma db
            try:
                from memory.store import store_brief
                store_brief(brief)
            except Exception as memory_error:
                log.warning(f"Failed to store in memory:{memory_error}")
        
        except Exception as e:
            msg=f"Brief Generation failed for {cluster_id}:{type(e).__name__}: {e}"
            log.error(msg)
            errors.append(msg)
    
    trace=(
        f"generate_brief: {len(briefs)} briefs generated via Gemini"
        f"({len(errors)} errors)"
    )                
    
    return {
        "generated_briefs":briefs,
        "reasoning_trace":trace,
        "error_log":errors
    }
                    
    
def save_briefs_to_db(state:WitnessState)->dict:
    """
    Persists generated InvestigationBrief objects to PostgreSQL.
    Gracefully skips if DB is unavailable.
    """
    briefs = state.get("generated_briefs", [])
    if not briefs:
        return {"reasoning_trace": ["save_briefs_to_db: no briefs to persist"]}
 
    saved  = 0
    errors: list[str] = []
 
    try:
        from db import get_db
        from repository import save_investigation_brief
 
        with get_db() as conn:
            for brief in briefs:
                try:
                    save_investigation_brief(brief, conn)
                    saved += 1
                except Exception as e:
                    errors.append(f"Failed to save brief {brief.brief_id}: {e}")
            conn.commit()
    except Exception as e:
        errors.append(f"DB unavailable — briefs not persisted: {e}")
        log.warning(f"save_briefs_to_db: DB unavailable: {e}")
 
    trace = f"save_briefs_to_db: {saved}/{len(briefs)} briefs persisted"
    return {
        "reasoning_trace": [trace],
        "error_log":       errors,
    }
    
    
# -----------------------------------
# CONDITIONAL ROUTING 
# ----------------------------------- 
def should_generate_briefs(state:WitnessState)->Literal["retrieve_historical_context","__end__"]:
    """
    ROuting function called afetr score retieval
    
    if the event surpasses the convergence score then call the memory retrival node then the briefs generation
    else, if it dosen't pass the threshold then just direct the flow towards END  
    """
    if state.get("briefs_to_generate"):
        log.info(
            f"Routing: {len(state["briefs_to_generate"])} clusters above the threshold"
            f"-> Proceding to brief generation"
        )
        return "retrieve_historical_context"
    else:
        log.info("Routing: No clusters above the threshold -> END")
        return "__end__"
    
    
# -----------------------------------------
# BUILD GRAPH ASSEMBLY 
# -----------------------------------------    
def build_graph():
    """
    Assemble and compile full langGraph agent Graph
    
    Returns compiled ready gtaph for invocation with:
        result=graph.invoke(make_initial_state(event))
        
    The complied graph is a StateGraph object that LangGraph uses to orchestrate node execution, Statepassing and conditional routing
    """    
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            f"LangGraph not installed. Run pip install langgraph"
        )
    builder = StateGraph(WitnessState)
    
    # Register Nodes
    builder.add_node("cluster_anomalies", )
    builder.add_node("score_convergence", score_convergence)
    builder.add_node("retrieve_historical_context", retrieve_historical_context)
    builder.add_node("generate_brief", generate_brief)
    builder.add_node("save_briefs_to_db", save_briefs_to_db)
    
    # Unconditional Edges ------------------
    builder.add_edge(START,"cluster_anomalies")
    builder.add_edge("cluster_anomalies","score_convergence")
    builder.add_edge("retrieve_historical_context","generate_brief")
    builder.add_edge("generate_brief","save_briefs_to_db")
    builder.add_edge("save_briefs_to_db",END)
     
    # Conditional edges ---------------------
    builder.add_conditional_edges(
        "score_convergence",
        should_generate_briefs,
        {
            "retrieve_historical_context":"retrieve_historical_context",
            "__end__":END
        }
    ) 
    
    return builder.compile()
    
    
# PURE-PYTHON FALLBACK RUNNER
# Runs the graph sequentially without LangGraph dependency.
# Used for testing and environments where langgraph isn't installed.

def run_agent_pipeline(events:list[AnomalyEvent])->WitnessState:
    """ 
    Runs full agent pipeline
    if langGraph is installed, uses the compiled graph
    otherwise runs nodes sequentially as plain python functions
    
    Return the final state dict after all nodes are executed
    """
    state=make_initial_state(events)
    if LANGGRAPH_AVAILABLE:
        graph=build_graph()
        result=graph.invoke(state)
        return result
    else:
        # Sequential fallback merge lists manually
        def apply(update:dict):
            for k, v in update.items():
                if isinstance(v,list) and isinstance(k,list):
                    state[k]=state[k]+v
                else:
                    state[k]=v
        
        apply(cluster_anomalies(state))
        apply(score_convergence(state))
        if state.get("briefs_to_generate"):
            apply(retrieve_historical_context(state))                
            apply(generate_brief(state))                
            apply(save_briefs_to_db(state))
        else:
            state["reasoning_trace"].append (
                "run_agent_pipeline: no briefs to generate, skip LLM nodes"
            )                  
        return state    
            
      
    
    
    
    
    
