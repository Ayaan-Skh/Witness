"""
Gemini Prompt Templates

All LLM prompt text lives here. No prompts are scattered inside node
functions. This makes iteration fast — tuning a prompt is editing one
file, not hunting through the codebase.

PROMPT DESIGN PRINCIPLES
------------------------
1. Structure over prose. We ask Gemini to respond in clearly delimited
   sections so we can parse and validate the output programmatically.

2. Epistemic honesty is mandatory. The prompt explicitly requires a
   "What this system cannot conclude" section. An AI that presents
   evidence without uncertainty is a dangerous tool. This section is
   enforced, not optional.

3. Evidence first, inference second. The prompt supplies all evidence
   in structured form. The LLM's job is synthesis and language, not
   data retrieval. It should never invent numbers.

4. Conservative confidence. We instruct the model to assign confidence
   tiers conservatively — HIGH requires all three sources, not just
   strong signals from one.

5. Human-first framing. The output is addressed to a human researcher,
   not to an automated system. The language should be appropriate for
   a professional who will make real decisions based on this brief.
"""

from __future__ import annotations

BRIEF_GENERATION_SYSTEM_PROMPT = """You are an analytical assistant for Witness, an OSINT platform that detects potential human rights violations by cross-correlating satellite imagery, news intelligence, and government procurement data.

Your role is to synthesize evidence collected by automated detection systems into structured investigation briefs for human journalists and researchers. You do not make final determinations. You present evidence, explain convergence, and flag what requires human follow-up.

CRITICAL RULES:
- Never state that a human rights violation has occurred. Use language like "the data is consistent with", "may indicate", "warrants investigation".
- Never invent statistics. Every number you cite must come from the evidence provided to you.
- The "What this system cannot conclude" section is mandatory and must be substantive, not a formality.
- Assign confidence tier conservatively. HIGH requires convergent evidence from all three independent sources."""


BRIEF_GENERATION_PROMPT = """You are analyzing a convergent anomaly cluster detected by the Witness OSINT system.

## Cluster Overview
Region: {region_id}
Time Window: {time_start} to {time_end}
Sources contributing: {sources}
Convergence score: {convergence_score:.2f} / 1.00
Number of anomaly events: {event_count}

## Evidence by Source

{evidence_sections}

## Historical Context from Memory
{historical_context}

---

Generate an investigation brief with EXACTLY these sections, using the exact headers shown:

### SUMMARY
2-3 sentences. What the data shows across all sources. Use cautious language ("consistent with", "may indicate").

### EVIDENCE BY STREAM
For each contributing source, one paragraph describing what was detected, the statistical strength of the signal, and what it means in isolation.

### CONVERGENCE ANALYSIS
Why the multi-source pattern is significant. Explain that these three data streams are independent — satellite imagery is unaffected by media suppression, procurement records are unaffected by satellite coverage, etc. Explain what the overlap means probabilistically.

### HISTORICAL CONTEXT
Summarize the relevant historical cases from memory. If none exist, state that explicitly. Note whether past alerts from this region were confirmed or were false positives.

### CONFIDENCE ASSESSMENT
Assign exactly one tier: LOW, MEDIUM, or HIGH.
- LOW: single source, or weak signals, or high baseline noise in region
- MEDIUM: two sources, or one strong source with supporting context
- HIGH: all three sources with statistically significant signals, low historical false positive rate

State the tier clearly and give explicit reasoning for why it was assigned.

### RECOMMENDED INVESTIGATIVE STEPS
Bullet list of 3-5 concrete actions a human researcher should take. Be specific (e.g., "Cross-reference with UNHCR displacement reports for the Tigray region dated November-December 2020" not "check external sources").

### WHAT THIS SYSTEM CANNOT CONCLUDE
Mandatory section. List at least 3 specific limitations:
- What alternative explanations exist for the observed signals
- What data the system does not have access to
- What the system's detection thresholds may have missed
- Why the signals are necessary but not sufficient evidence of a human rights violation"""


def format_evidence_sections(cluster: dict, events: list) -> str:
    """
    Formats evidence from the cluster's events into structured text
    for insertion into the prompt.

    Groups events by source and formats each group's key statistics.
    """
    from normalization.schema import AnomalySource

    by_source: dict[str, list] = {}
    for event in events:
        src = event.source.value
        by_source.setdefault(src, []).append(event)

    sections = []

    if "SATELLITE" in by_source:
        sat_events = by_source["SATELLITE"]
        scores     = [e.intensity_score for e in sat_events]
        signals    = sorted(set(e.signal_type.value for e in sat_events))
        max_score  = max(scores)

        # Pull detail from raw_data of the highest-scoring event
        best     = max(sat_events, key=lambda e: e.intensity_score)
        raw      = best.raw_data
        sub      = raw.get("sub_scores", {})

        section = f"""**SATELLITE (Sentinel-2)**
Signal types detected: {', '.join(signals)}
Peak intensity score: {max_score:.3f} / 1.00
Events in cluster: {len(sat_events)}"""

        if sub:
            section += f"""
Sub-scores: NDVI drop={sub.get('ndvi_drop', 0):.2f}, spectral change={sub.get('spectral', 0):.2f}, thermal={sub.get('thermal', 0):.2f}, structure={sub.get('structure', 0):.2f}"""
        if raw.get("ndvi_before_mean") is not None:
            section += f"""
NDVI before: {raw.get('ndvi_before_mean', 0):.3f} → after: {raw.get('ndvi_after_mean', 0):.3f}"""
        if raw.get("active_fire_flag"):
            section += f"\n⚠ Active fire / thermal anomaly flagged (SWIR p99={raw.get('swir_p99', 0):.3f})"

        sections.append(section)

    if "GDELT" in by_source:
        gdelt_events = by_source["GDELT"]
        best         = max(gdelt_events, key=lambda e: e.intensity_score)
        raw          = best.raw_data
        tone_data    = raw.get("tone", {})
        vol_data     = raw.get("volume", {})
        themes       = raw.get("top_concerning_themes", [])

        section = f"""**GDELT NEWS INTELLIGENCE**
Signal types detected: {', '.join(set(e.signal_type.value for e in gdelt_events))}
Peak intensity score: {best.intensity_score:.3f} / 1.00"""

        if tone_data.get("current") is not None:
            section += f"""
Tone: current={tone_data.get('current', 0):.2f}, baseline mean={tone_data.get('baseline_mean', 0):.2f} (z-score={tone_data.get('zscore', 0):.2f})"""
        if vol_data.get("current") is not None:
            section += f"""
Volume: current={vol_data.get('current', 0)} mentions, baseline mean={vol_data.get('baseline_mean', 0):.0f} (z-score={vol_data.get('zscore_drop', 0):.2f})"""
        if themes:
            theme_codes = [t.get("cameo_code") for t in themes[:3]]
            section += f"\nTop concerning CAMEO codes: {', '.join(str(c) for c in theme_codes)}"

        sections.append(section)

    if "PROCUREMENT" in by_source:
        proc_events = by_source["PROCUREMENT"]
        best        = max(proc_events, key=lambda e: e.intensity_score)
        raw         = best.raw_data
        categories  = raw.get("categories", {})

        section = f"""**PROCUREMENT (OCDS)**
Signal types detected: {', '.join(set(e.signal_type.value for e in proc_events))}
Peak intensity score: {best.intensity_score:.3f} / 1.00
Analysis month: {raw.get('current_month', 'unknown')}"""

        for cat, stats in categories.items():
            if stats.get("fired"):
                section += (
                    f"\n{cat}: current spend=${stats.get('current_spend_usd', 0):,.0f} "
                    f"vs baseline=${stats.get('baseline_mean_usd', 0):,.0f} "
                    f"(z-score={stats.get('zscore', 0):.2f})"
                )
        if raw.get("new_vendors"):
            section += f"\nNew vendors in sensitive categories: {len(raw['new_vendors'])}"

        sections.append(section)

    return "\n\n".join(sections) if sections else "No evidence detail available."


def build_brief_prompt(
    cluster: dict,
    events: list,
    convergence_score: float,
    historical_context: str,
) -> str:
    """
    Builds the full brief generation prompt by filling the template.
    """
    t_start = cluster.get("time_start")
    t_end   = cluster.get("time_end")

    def fmt_dt(dt):
        if dt is None:
            return "unknown"
        if isinstance(dt, str):
            from datetime import datetime
            dt = datetime.fromisoformat(dt)
        return dt.strftime("%Y-%m-%d")

    evidence_text = format_evidence_sections(cluster, events)

    return BRIEF_GENERATION_PROMPT.format(
        region_id=cluster.get("region_id", "unknown"),
        time_start=fmt_dt(t_start),
        time_end=fmt_dt(t_end),
        sources=", ".join(sorted(set(cluster.get("sources", [])))),
        convergence_score=convergence_score,
        event_count=len(events),
        evidence_sections=evidence_text,
        historical_context=historical_context,
    )