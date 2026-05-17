# Witness

**Open-source intelligence system for detecting potential human rights crises through convergent analysis of satellite imagery, global news, and government procurement records.**

![Witness — three data streams converging](WITNESS.png)

---

## What it does

Witness watches a set of monitored regions continuously and asks one question: are three completely independent data sources — each with different failure modes, controlled by different actors, with no reason to agree — all flagging the same place at the same time?

- A **satellite** showing land cover change could be farming
- A **news sentiment crash** could be an election
- A **procurement spike** in medical supplies could be a budget cycle

But when all three converge on the same region in the same two-week window, that pattern is worth a human investigator looking at.

The system never draws conclusions. It surfaces evidence, scores convergence, and explicitly states what it cannot determine. A human researcher reviews every output.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Daily Pipeline (02:00 UTC)              │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │  Satellite  │  │    GDELT    │  │   Procurement    │   │
│  │  Sentinel-2 │  │  BigQuery   │  │   OCDS / STAC   │   │
│  │  via MS PC  │  │  100M news  │  │  Gov contracts  │   │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘   │
│         │                │                   │             │
│         └────────────────┼───────────────────┘             │
│                          ▼                                  │
│              ┌─────────────────────┐                       │
│              │   AnomalyEvent[]    │  ← unified schema      │
│              └──────────┬──────────┘                       │
│                         ▼                                   │
│         ┌───────────────────────────────┐                  │
│         │      LangGraph Agent          │                  │
│         │                               │                  │
│         │  cluster_anomalies            │  ← pure geometry  │
│         │       ↓                       │                  │
│         │  score_convergence            │  ← pure math      │
│         │       ↓                       │                  │
│         │  [score ≥ 0.55?] ────── END   │  ← short-circuit  │
│         │       ↓                       │                  │
│         │  retrieve_historical_context  │  ← ChromaDB       │
│         │       ↓                       │                  │
│         │  generate_brief               │  ← Gemini 2.0     │
│         └───────────────┬───────────────┘                  │
│                         ▼                                   │
│              ┌─────────────────────┐                       │
│              │  InvestigationBrief │  → PostgreSQL          │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Satellite imagery | Microsoft Planetary Computer (free) · Sentinel-2 L2A · stackstac |
| News intelligence | GDELT via Google BigQuery · 100M+ events · 65 languages |
| Procurement data | OCDS open standard · Prozorro (Ukraine) · World Bank dataset |
| Change detection | NDVI · SWIR thermal · NDBI structure change · scipy |
| Anomaly detection | Z-score baseline deviation · rolling 90-day windows |
| Agent framework | LangGraph 0.2 · multi-node directed graph with conditional routing |
| LLM | Gemini 2.0 Flash (free tier) via LangChain |
| Vector memory | ChromaDB · all-MiniLM-L6-v2 embeddings |
| Database | PostgreSQL 16 + PostGIS · geospatial clustering |
| API | FastAPI · Pydantic · OpenAPI docs at `/docs` |
| Dashboard | Next.js 14 · Mapbox GL JS · Tailwind CSS |
| Scheduling | APScheduler · daily 02:00 UTC |

---

## Monitored Regions

| Region | Country | Purpose |
|---|---|---|
| Tigray | Ethiopia | Primary calibration region — 2020-2022 conflict |
| Xinjiang | China | Satellite-primary (media access limited) |
| Mariupol | Ukraine | Primary demo — strong all-3-source signals |
| Amazon Arc | Brazil | Satellite calibration (deforestation baseline) |
| Rakhine State | Myanmar | Secondary validation |

---

## Detection Signals

**Satellite (Sentinel-2 L2A)**
- `VEGETATION_LOSS` — NDVI drop between two dates (deforestation, burning, clearing)
- `LAND_COVER_CHANGE` — spectral change across all 6 bands
- `THERMAL_ANOMALY` — SWIR1 spike indicating fire or heat source
- `STRUCTURE_CHANGE` — NDBI increase indicating new construction

**GDELT News Intelligence**
- `TONE_CRASH` — daily sentiment z-score drops below −2σ from 90-day baseline
- `COMMUNICATION_BLACKOUT` — mention volume drops below −2σ (silence as signal)
- `VOLUME_SPIKE` — coverage surge above +2σ
- `CONFLICT_EVENTS` — volume spike with CAMEO conflict codes (airstrikes, displacement)

**Procurement (OCDS)**
- `SPEND_SPIKE` — category spend above +2.5σ from 12-month rolling baseline
- `NEW_VENDOR_PATTERN` — new suppliers appearing in military/medical categories
- `EMERGENCY_CONTRACT` — direct-award procurement bypassing competitive tender

---

## Convergence Scoring

Each cluster of co-located, co-timed events is scored 0–1:

```
score = 0.50 × stream_diversity + 0.30 × intensity + 0.20 × recency

stream_diversity:
  1 source  → 0.20   (could be noise)
  2 sources → 0.60   (statistically notable)
  3 sources → 1.00   (independent convergence — very strong signal)
```

Clusters scoring ≥ 0.55 proceed to brief generation. Everything below exits without an LLM call.

---

## Project Structure

```
witness/
├── ingestion/
│   ├── satellite.py        # Planetary Computer — Sentinel-2 tile fetching
│   ├── gdelt.py            # BigQuery — GDELT tone/volume timeseries
│   └── procurement.py      # OCDS — contract fetching and categorization
├── normalization/
│   └── schema.py           # AnomalyEvent + InvestigationBrief dataclasses
├── detection/
│   ├── change_detection.py # NDVI/SWIR/NDBI change scoring
│   ├── gdelt_anomaly.py    # Z-score anomaly detection on GDELT timeseries
│   └── procurement_anomaly.py # Rolling baseline deviation detection
├── agent/
│   ├── graph.py            # LangGraph graph — nodes + conditional routing
│   ├── nodes.py            # cluster_anomalies + score_convergence nodes
│   ├── state.py            # WitnessState TypedDict
│   └── prompts.py          # Gemini prompt templates
├── memory/
│   └── store.py            # ChromaDB vector store for historical context
├── api/
│   └── main.py             # FastAPI — /briefs /anomalies /regions endpoints
├── dashboard/              # Next.js 14 + Mapbox GL JS
├── scheduler/
│   └── pipeline.py         # Daily pipeline — all 3 stages + agent
├── tests/                  # 400+ tests, zero external dependencies
├── config.py               # All constants and monitored region definitions
├── schema.sql              # PostgreSQL + PostGIS schema
└── docker-compose.yml      # PostgreSQL + PostGIS local setup
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- Node.js 18+ (for dashboard)
- [Microsoft Planetary Computer account](https://planetarycomputer.microsoft.com) — free, no credit card
- [Google Cloud project](https://console.cloud.google.com) — free tier, BigQuery enabled
- [Gemini API key](https://aistudio.google.com/app/apikey) — free tier
- [Mapbox token](https://account.mapbox.com) — free tier

### Backend

```bash
# 1. Clone and enter
git clone https://github.com/yourusername/witness.git
cd witness

# 2. Virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment
cp .env.example .env
# Fill in: GEMINI_API_KEY, GOOGLE_CLOUD_PROJECT,
#          GOOGLE_APPLICATION_CREDENTIALS, NEXT_PUBLIC_MAPBOX_TOKEN

# 5. Database
docker compose up -d
python db.py   # initialises schema + seeds regions

# 6. Run tests
pytest tests/ -v

# 7. Start API
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### Dashboard

```bash
cd dashboard
cp .env.example .env.local
# Fill in NEXT_PUBLIC_API_URL and NEXT_PUBLIC_MAPBOX_TOKEN

npm install
npm run dev
# → http://localhost:3000
```

### Run the pipeline manually

```bash
# Single date (yesterday by default)
python scheduler/pipeline.py --dry-run

# Specific date
python scheduler/pipeline.py --date 2022-03-01 --dry-run

# Historical backfill (Mariupol siege window)
python scheduler/pipeline.py --backfill 2022-02-01:2022-05-01 \
  --regions ukr_mariupol --dry-run
```

---

## API

FastAPI auto-generates interactive docs at `http://localhost:8000/docs`

| Endpoint | Description |
|---|---|
| `GET /briefs` | Paginated briefs — filter by tier, region, status, date |
| `GET /briefs/{id}` | Full brief with evidence and agent reasoning |
| `GET /anomalies` | Raw anomaly events — filter by source, region, intensity |
| `GET /regions` | Monitored regions with 30-day anomaly counts |
| `POST /pipeline/run` | Manually trigger pipeline (requires `X-API-Key` header) |

---

## Ethical Design

This system is built with explicit constraints:

- **No automatic publishing.** Every brief is `DRAFT` until a human marks it `REVIEWED` then `PUBLISHED`. The pipeline cannot publish anything itself.
- **Epistemic honesty enforced in prompts.** The Gemini prompt requires a `### WHAT THIS SYSTEM CANNOT CONCLUDE` section with at least three specific limitations. The LLM is instructed: *"Never state that a human rights violation has occurred."*
- **Evidence only, not verdicts.** Witness surfaces statistical anomalies for human investigation. It is not a detection system, not a verification system, and not a legal instrument.
- **Open source.** The detection methodology, thresholds, and calibration data are fully public so they can be scrutinized and challenged.

See `METHODOLOGY.md` for full documentation of detection thresholds, calibration cases, and known limitations.

---

## Tests

```bash
pytest tests/ -v          # all tests
pytest tests/ -q          # summary only
pytest tests/test_day2.py # specific day
```

400+ tests covering all detection modules, the agent pipeline, and the API layer. Zero tests require external API credentials — all network calls are mocked.

---

## Built by

**Ayaan** · Pre-final year B.Tech Computer Science · Walchand Institute of Technology, Solapur

Open to internship opportunities in ML Engineering and Backend Engineering.

[LinkedIn](https://linkedin.com/in/yourprofile) · [Email](mailto:your@email.com)

---

*Witness is an open-source research tool. It is not affiliated with any government, NGO, or intelligence agency.*