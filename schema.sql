-- schema.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Witness Database Schema
--
-- Run this file once against your PostgreSQL instance to initialize the schema:
--   psql -U witness_user -d witness_db -f schema.sql
--
-- Prerequisites:
--   CREATE DATABASE witness_db;
--   CREATE USER witness_user WITH PASSWORD 'witness_pass';
--   GRANT ALL PRIVILEGES ON DATABASE witness_db TO witness_user;
--   \c witness_db
--   CREATE EXTENSION IF NOT EXISTS postgis;
--   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- ─────────────────────────────────────────────────────────────────────────────


-- ─────────────────────────────────────────────
-- WHY PostGIS?
--
-- Plain PostgreSQL stores coordinates as two plain numbers (lat, lng).
-- That works until you need spatial queries like:
--   "Find all anomaly events within 150km of this point"
--   "Which events fall inside this bounding box?"
--   "What's the distance between these two events?"
--
-- PostGIS adds a native geometry type (GEOGRAPHY/GEOMETRY) and spatial
-- indexes (GIST), enabling these queries in pure SQL with sub-millisecond
-- performance. Without PostGIS, you'd compute distances in Python in a loop —
-- which is ~1000x slower once you have tens of thousands of events.
--
-- Think of PostGIS as adding a GPS navigation system to PostgreSQL.
-- Without it, your database knows cities exist. With it, it knows how
-- to calculate the shortest route between them.
-- ─────────────────────────────────────────────


-- ─────────────────────────────────────────────
-- TABLE: monitored_regions
-- Mirrors the MonitoredRegion config objects, persisted to DB for joins.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS monitored_regions (
    region_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    country_code    CHAR(2) NOT NULL,
    admin1          TEXT,
    -- PostGIS polygon representing the bounding box
    -- ST_MakeEnvelope(min_lng, min_lat, max_lng, max_lat, 4326)
    -- SRID 4326 = standard GPS coordinate system (WGS 84)
    boundary        GEOGRAPHY(POLYGON, 4326),
    buyer_ids       TEXT[],   -- array of OCDS entity IDs
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regions_country ON monitored_regions(country_code);


-- ─────────────────────────────────────────────
-- TABLE: anomaly_events
--
-- The central facts table. Every anomaly detected by every ingestion
-- pipeline ends up here as a row. The source column (SATELLITE/GDELT/
-- PROCUREMENT) tells you which pipeline produced it.
--
-- This is the "incident log" — like a police department's CAD system
-- where every call, regardless of type, gets a timestamped record.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_events (
    event_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source          TEXT NOT NULL CHECK (source IN ('SATELLITE', 'GDELT', 'PROCUREMENT')),
    region_id       TEXT NOT NULL REFERENCES monitored_regions(region_id),
    country_code    CHAR(2) NOT NULL,

    -- PostGIS point — the geographic center of this anomaly
    -- Enables spatial queries: ST_DWithin, ST_Distance, etc.
    location        GEOGRAPHY(POINT, 4326) NOT NULL,

    -- Stored also as plain columns for quick non-spatial filtering
    lat             DOUBLE PRECISION NOT NULL,
    lng             DOUBLE PRECISION NOT NULL,

    -- When this anomaly was observed (not when we detected it)
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Human-readable description of the anomaly type
    -- Examples: 'land_cover_change', 'tone_crash', 'procurement_spike'
    signal_type     TEXT NOT NULL,

    -- 0.0 = barely anomalous, 1.0 = maximally anomalous
    -- Each source's detection module normalizes its raw score to this range
    intensity_score DOUBLE PRECISION NOT NULL CHECK (intensity_score BETWEEN 0.0 AND 1.0),

    -- The full raw data from the source as JSON
    -- For satellite: tile metadata, NDVI values, change score breakdown
    -- For GDELT: event codes, tone values, article counts
    -- For procurement: contract IDs, amounts, categories
    raw_data        JSONB NOT NULL DEFAULT '{}',

    -- Additional context that doesn't fit the core schema
    -- For satellite: cloud cover %, resolution
    -- For GDELT: dominant CAMEO codes
    -- For procurement: vendor names, commodity categories
    metadata        JSONB NOT NULL DEFAULT '{}',

    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    pipeline_run_id UUID  -- links to pipeline_runs table
);

-- Spatial index — powers fast radius queries in the clustering node
CREATE INDEX IF NOT EXISTS idx_events_location ON anomaly_events USING GIST(location);

-- Covering index for the most common query pattern: "events in region X after date Y"
CREATE INDEX IF NOT EXISTS idx_events_region_time ON anomaly_events(region_id, timestamp DESC);

-- Index for source filtering: "show me all GDELT events this week"
CREATE INDEX IF NOT EXISTS idx_events_source ON anomaly_events(source, timestamp DESC);


-- ─────────────────────────────────────────────
-- TABLE: investigation_briefs
--
-- The output of the LangGraph agent. Each brief is the result of the agent
-- deciding that a cluster of anomaly events represents something a human
-- investigator should look at.
--
-- These are the "detective memos" — the synthesized output after the agent
-- has correlated the raw incident log into a coherent picture.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS investigation_briefs (
    brief_id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id               TEXT NOT NULL REFERENCES monitored_regions(region_id),

    -- The time window this brief covers (not necessarily the same as detection date)
    time_window_start       TIMESTAMPTZ NOT NULL,
    time_window_end         TIMESTAMPTZ NOT NULL,

    -- 0.0 to 1.0 — the agent's computed confidence this is a real signal
    confidence_score        DOUBLE PRECISION NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 1.0),

    -- Which sources contributed: ['SATELLITE', 'GDELT', 'PROCUREMENT']
    contributing_streams    TEXT[] NOT NULL,

    -- Evidence JSON, keyed by source:
    -- { "SATELLITE": { "change_score": 0.7, "tile_paths": [...] },
    --   "GDELT": { "tone_zscore": -3.1, "top_themes": [...] } }
    evidence                JSONB NOT NULL DEFAULT '{}',

    -- The LLM's full reasoning text
    agent_reasoning         TEXT,

    -- DRAFT = generated, not yet human-reviewed
    -- REVIEWED = a human researcher has examined it
    -- PUBLISHED = verified and approved for public release
    status                  TEXT NOT NULL DEFAULT 'DRAFT'
                            CHECK (status IN ('DRAFT', 'REVIEWED', 'PUBLISHED')),

    -- Human analyst notes added during review
    reviewer_notes          TEXT,

    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_briefs_region ON investigation_briefs(region_id, time_window_start DESC);
CREATE INDEX IF NOT EXISTS idx_briefs_confidence ON investigation_briefs(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_briefs_status ON investigation_briefs(status);


-- ─────────────────────────────────────────────
-- TABLE: region_baselines
--
-- Stores the computed baseline statistics for each region+source+metric
-- combination. The anomaly detectors compare current readings against
-- these baselines to compute z-scores.
--
-- Refreshed daily by the pipeline. Think of this as the "what is normal
-- here" reference card that the anomaly detectors consult.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS region_baselines (
    id              SERIAL PRIMARY KEY,
    region_id       TEXT NOT NULL REFERENCES monitored_regions(region_id),
    source          TEXT NOT NULL,  -- SATELLITE / GDELT / PROCUREMENT
    metric          TEXT NOT NULL,  -- 'ndvi', 'news_volume', 'tone', 'military_spend', etc.
    baseline_mean   DOUBLE PRECISION NOT NULL,
    baseline_std    DOUBLE PRECISION NOT NULL,
    sample_count    INTEGER NOT NULL,  -- how many data points went into this baseline
    computed_from   DATE NOT NULL,     -- start of baseline window
    computed_to     DATE NOT NULL,     -- end of baseline window (usually yesterday)
    computed_at     TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (region_id, source, metric, computed_to)
);

CREATE INDEX IF NOT EXISTS idx_baselines_lookup
    ON region_baselines(region_id, source, metric, computed_to DESC);


-- ─────────────────────────────────────────────
-- TABLE: pipeline_runs
--
-- Audit log for every pipeline execution. If something goes wrong,
-- this table tells you exactly what ran, when, and whether it succeeded.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'RUNNING'
                    CHECK (status IN ('RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED')),
    -- Per-source outcome: { "satellite": "ok", "gdelt": "ok", "procurement": "failed" }
    stage_results   JSONB NOT NULL DEFAULT '{}',
    events_created  INTEGER DEFAULT 0,
    briefs_created  INTEGER DEFAULT 0,
    error_details   TEXT
);


-- ─────────────────────────────────────────────
-- TRIGGER: auto-update investigation_briefs.updated_at
-- ─────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER briefs_updated_at
    BEFORE UPDATE ON investigation_briefs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();