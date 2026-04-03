import os
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass, field

load_dotenv()

SENTINEL_HUB_CLIENT_ID= os.getenv("SENTINEL_HUB_CLIENT_ID")
SENTINEL_HUB_CLIENT_SECRET= os.getenv("SENTINEL_HUB_CLIENT_SECRET")

GOOGLE_CLOUD_PROJECT =os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

GEMINI_KEY =os.getenv("GEMINI_KEY")

DATABASE_URL=os.getenv(
    "DATABASE_URL",
    "postgresql://witness_user:witness_pass@localhost:5432/witness_db"
)

CHROMA_PERSIST_DIR=os.getenv("CHROMA_PERSIST_DIR",' ./data/chroma_db')
TITLE_CACHE_DIR = os.getenv("TITLE_CACHE_DIR","./data/title_cache")

# LLM Configurations

LLM_MODEL="gemini-2.0-flash"
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048


# Detection Thresholds
SATELLITE_CHANGE_THRESHOLD = 0.35    # Change score above this triggers an AnomalyEvent
SATELLITE_NDVI_THRESHOLD = 0.15      # NDVI drop (vegetation change indicator) above this flags it
SATELLITE_RESOLUTION_M = 10 # 10 meters per pixel 

# GDELT anamoly detection thresholds
GDELT_ZSCORE_THRESHOLD = 2.0        # 2 standard deviation= statistically unsual (covers 95% of normal)
GDELT_BLACKOUT_THRESHOLD = -2.0     # VOlume z-score below this = "communication blackout" signal
GDELT_TONE_CRASH_THRESHOLD = -2.0   # Tone z-score below this= extreme negative tone shift
GDELT_BASELINE_LOOKBACK_DAYS = 90       # How many days of history to use to get a baseline data for z-score calculations


#Procurment anamoly detection analysis
PROCUREMENT_ZSCORE_THRESHOLD=2.5 # A LITTLE GREATER THAN gdelt AS IT IS VERY NOISY
PROCUREMENT_BASELINE_MONTHS=12 # 12 months rolling window for seasonal handling
PROCUREMENT_NEW_VENDOR_LOOKBACK=6 #Months of history to check before flagging new vendor

# Langgraph agent thresholds
CONVERGENCE_SCORE_THRESHOLD = 0.55 # Minimum score for a cluster to become a brief candidate
GEOGRAPHIC_CLUSTER_RADIUS_KM = 150 #  Events occuring within this radius are related geographically
TEMPORAL_CLUSTER_WINDOW_DAYS = 21 #Events within 21 days are considered to be temporarily related
MEMORY_RETRIEVAL_TOP_K = 3 # How many historical cases to retrieve per cluster    







