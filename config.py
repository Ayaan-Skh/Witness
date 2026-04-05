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


@dataclass
class MonitoredRegion:
    region_id:str
    name:str
    country_code:str
    admin1:str|None
    bbox:List[float] #[min_lon, min_lat, max_lon, max_lat]
    buyer_ids:List[str]=field(default_factory=list)
    
    def centroid(self)->tuple[float,float]:
        """Returns the geographic center of the bounding box as (lat, lng) """
        lat=(self.bbox[1]+self.bbox[3])/2
        lng=(self.bbox[0]+self.bbox[2])/2
        return (lat,lng)
    
    def area_km2(self)->float:
        """
            Rough area estimate to get flat earth approximations.
            GOod for regions not for continents.
        """
        lat_km=(self.bbox[3]-self.bbox[1])*111.0
        lng_km=(self.bbox[2]-self.bbox[0]) *111.0 * abs(self.bbox[1]+self.bbox[3])/2/90.0
        
        return lat_km*lng_km
        
MONITORED_REGIONS:List[MonitoredRegion]=[
        # ── Tigray Region, Ethiopia ──────────────────────────────────────────────
    # Scene of a documented conflict 2020–2022 with satellite, news, and
    # procurement signals all present in the historical record.
    # Used as the primary calibration/validation region.
    MonitoredRegion(
        region_id="eth_tigray",
        name="Tigray, Ethiopia",
        country_code="ET",
        admin1="ET.TI",
        bbox=[36.45, 12.30, 40.00, 15.00],
        buyer_ids=["ET-MOD-001"],  # Ethiopian Ministry of Defence (placeholder)
    ),
 
    # ── Xinjiang Uyghur Autonomous Region, China ────────────────────────────
    # Extensive satellite documentation of facility construction 2017–2020.
    # GDELT signals limited due to media restrictions — a good test of
    # multi-source correlation where one stream is intentionally suppressed.
    MonitoredRegion(
        region_id="chn_xinjiang",
        name="Xinjiang, China",
        country_code="CN",
        admin1="CN.XJ",
        bbox=[73.40, 34.20, 96.40, 49.10],
        buyer_ids=[],  # Chinese OCDS data not publicly available
    ),
 
    # ── Mariupol Area, Ukraine ───────────────────────────────────────────────
    # February–May 2022 siege. Strong signals across all three streams.
    # Used as the primary demo case because satellite imagery is widely
    # available and ground truth is extensively documented.
    MonitoredRegion(
        region_id="ukr_mariupol",
        name="Mariupol, Ukraine",
        country_code="UA",
        admin1="UA.DK",  # Donetsk Oblast
        bbox=[37.00, 46.85, 37.80, 47.30],
        buyer_ids=["UA-MOD-001"],  # Ukrainian MoD (placeholder)
    ),
 
    # ── Amazon Deforestation Arc, Brazil ────────────────────────────────────
    # Used specifically for satellite pipeline calibration — deforestation
    # produces clear, measurable NDVI drops with no ambiguity.
    # Not a human rights case; used for technical baseline testing only.
    MonitoredRegion(
        region_id="bra_amazon_arc",
        name="Amazon Deforestation Arc, Brazil",
        country_code="BR",
        admin1="BR.PA",  # Pará state
        bbox=[-58.00, -10.00, -46.00, -3.00],
        buyer_ids=[],
    ),
 
    # ── Rakhine State, Myanmar ───────────────────────────────────────────────
    # 2017 crisis generated signals in all three streams.
    # Used as a secondary validation region.
    MonitoredRegion(
        region_id="mmr_rakhine",
        name="Rakhine State, Myanmar",
        country_code="MM",
        admin1="MM.RA",
        bbox=[92.10, 17.50, 95.70, 22.00],
        buyer_ids=[],
    ),
    
        MonitoredRegion(
        region_id="ind_delhi",
        name="Delhi, India",
        country_code="IN",
        admin1="IN.DL",
        bbox=[76.8, 28.4, 77.5, 28.9],
        buyer_ids=["IN-MOD-001"],
    ),
        MonitoredRegion(
        region_id="ind_maharashtra",
        name="Maharashtra, India",
        country_code="IN",
        admin1="IN.MH",
        bbox=[72.6, 15.6, 80.9, 22.2],  # Approx: [Min Long, Min Lat, Max Long, Max Lat]
        buyer_ids=["IN-MOD-001"],       # Placeholder for Indian Ministry of Defence
    ), 
        
        MonitoredRegion(
        region_id="ind_national",
        name="India",
        country_code="IN",
        admin1=None,  # Not applicable for country-wide
        bbox=[68.1, 6.7, 97.4, 35.5],
        buyer_ids=["IN-MOD-001"],
    )
]
    
REGIONS_BY_ID:dict[str,MonitoredRegion]={r.region_id: r for r in MONITORED_REGIONS}    
    
# SCHEDULE CONFIGURATIONS    
PIPELINE_RUN_HOUR_UTC = 2 #Mostly after data is updated everywhere.
PIPELINE_RUNMINUTE_UTC = 1 # Avoid running exactly on the hour to prevent conflicts with data updates that often happen at the tip of hour.
PIPELINE_LOOKBACK_DAYS = 3

# API CONFIGURATIONS
API_HOST='0.0.0.0'
API_PORT=8000
API_RELOAD = os.getenv("ENV", "development") == "development" # AUTO RELODE IN DEVELOPMENT
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "insecure-dev-key-replace-in-production") # In production, this should be a secure, randomly generated string. Used for signing tokens and securing endpoints.


# Pagination
DEFAULT_PAGE_SIZE = 20 # Default number of items per page for API responses
MAX_PAGE_SIZE = 100 # Maximum number of items per page for API responses

