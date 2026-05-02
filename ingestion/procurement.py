"""
GOVERNMENT PROCUREMENT INGESTION

The Open Contracting Data Standard (OCDS) is an international transparency framework that governments use to publish procurement data — every contract awarded, to whom, for how much, and for what. 60+ countries publish in this format, including Ukraine, Philippines, Nigeria, and Moldova.


DATA SOURCES
────────────
Primary:   World Bank OCDS dataset (global coverage, REST API)
Secondary: Country-specific portals
  - Ukraine: prozorro.gov.ua (excellent coverage, real-time)
  - Philippines: philgeps.gov.ph
  - Nigeria: nocopo.gov.ng
  - Moldova: mtender.gov.md
 
Fallback:  Static OCDS JSON files downloaded and cached locally
           (used when portals are unavailable or for historical analysis)
"""


from __future__ import annotations

import hashlib
import json 
import logging
from datetime import date, datetime,timezone, timedelta,UTC

from pathlib import Path
from typing import Optional

from numpy import full
import requests
from config import TILE_CACHE_DIR
log=logging.getLogger("witness.procurement")

# Cache procurement responses locally — same philosophy as satellite tiles.
# OCDS APIs can be slow or rate-limited; caching keeps the pipeline fast.
PROCUREMENT_CACHE_DIR=Path(TILE_CACHE_DIR).parent/"procurement.cache"

# -------------------------------------------------------------------------------
#  CONTRACT CATEGORIES
# We classify contracts into categories using keyword matching on title and description fields.The categories are chosen specifically because they correlate with the pre-/during/post-conflict operational cycle.
# -------------------------------------------------------------------------------

CATEGORY_KEYWORDS:dict[str,list[str]]={
    "MILITARY": [
        "ammunition", "ammo", "weapon", "rifle", "artillery", "missile",
        "armored", "armoured", "military vehicle", "combat", "tactical",
        "explosive", "grenade", "mortar", "tank", "helicopter gunship",
        "drone", "uav", "body armor", "ballistic", "night vision",
        "military uniform", "camouflage", "military boot",
    ],
    "MEDICAL": [
        "medical", "medicine", "pharmaceutical", "hospital", "surgical",
        "ambulance", "blood", "bandage", "tourniquet", "morphine",
        "field hospital", "triage", "trauma", "oxygen", "ventilator",
        "body bag", "coffin", "mortuary", "autopsy", "forensic",
        "stretcher", "wheelchair", "prosthetic",
    ],
    "LOGISTICS": [
        "fuel", "diesel", "petrol", "gasoline", "truck", "lorry",
        "transport vehicle", "logistics", "convoy", "warehouse",
        "storage facility", "supply chain", "ration", "food supply",
        "water purification", "generator", "tent", "shelter",
    ],
    "CONSTRUCTION": [
        "construction", "building", "facility", "fencing", "barbed wire",
        "watchtower", "perimeter", "prefabricated", "concrete barrier",
        "detention", "compound", "enclosure", "checkpoint",
    ],
    "COMMUNICATIONS": [
        "communication", "radio", "satellite phone", "signal", "frequency",
        "encryption", "surveillance", "cctv", "monitoring", "interception",
        "telecommunications", "network infrastructure",
    ],
    "OTHER": [],   # catch-all — matched last
}

# Ordered list — earlier categories take priority on keyword overlap
CATEGORY_ORDER = ["MILITARY", "MEDICAL", "LOGISTICS", "CONSTRUCTION", "COMMUNICATIONS", "OTHER"]
 
# Categories flagged as "sensitive" — these drive anomaly detection
SENSITIVE_CATEGORIES = {"MILITARY", "MEDICAL", "LOGISTICS", "CONSTRUCTION"}

OCDS_ENDPOINTS = {
    # World Bank OCDS bulk dataset — global, updated monthly
    "world_bank": "https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/ocds/records",
    # Ukraine Prozorro — best real-time coverage
    "ukraine":    "https://public.api.openprocurement.org/api/2.5/tenders",
    # Philippines GEPS
    "philippines": "https://api.philgeps.gov.ph/gepsnonpilot/api/search/bids",
}

#-----------------------------------
# CACHING
#-----------------------------------

def _procurement_cache_key(
    country_code:str,
    buyer_id:Optional[str],
    start_date:date,
    end_date:date
    )->str:
    raw=f"{country_code}_{buyer_id}_{start_date.isoformat()}_{end_date.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _read_procurement_cache(key:str)->Optional[list[dict]]:
    PROCUREMENT_CACHE_DIR.mkdir(parents=True,exist_ok=True)
    path=PROCUREMENT_CACHE_DIR/f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data=json.load(f)
        # Cache expire after 24 hrs
        cached_at =datetime.fromisoformat(data.get("cached_at","2000-01-01T00:00:00"))
        if (datetime.now(UTC) - cached_at).total_seconds() > 86400:
            return None
        
        return data.get("records",[])
         
    except Exception:
        return None
    

def _write_procurement_cache(key:str, records:list[dict])->None:
    PROCUREMENT_CACHE_DIR.mkdir(parents=True,exist_ok=True)
    path=PROCUREMENT_CACHE_DIR/f"{key}.json" 
    with open(path,"w") as f:
        json.dump({
            "cached_at":datetime.now(UTC).isoformat(),
            "records":records,
        },f,indent=2,default=str)
        

#----------------------------------------
# CONTRACT CATEGORIZATION 
#---------------------------------------- 

def categorize_contract(contract:dict)->str:
    """
    Maps our raw OCDS to one of our predefined categories using keyword matching on title and description fields
    
    Strategy: Case insensitive substring matching. Simple but effective. Procurement titles like "Supply of 5.56mm ammunition" don't need semantic understanding, just pattern recognition.
 
    Returns the category string ("MILITARY", "MEDICAL", etc.).
    The "OTHER" catch-all is returned if no keywords match.
    """
    
    text_fields=[
        contract.get("title",""),
        contract.get("description",""),
        contract.get("tender", {}).get("title", ""),
        contract.get("tender", {}).get("description", ""),
        contract.get("items", [{}])[0].get("description", "") if contract.get("items") else "",    
    ]
    full_text=" ".join(str(f) for f in text_fields if f).lower()
    
    for category in CATEGORY_ORDER[:-1]:
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword.lower() in full_text:
                return category

    return "OTHER"

def extract_amount_usd(contract: dict) -> Optional[float]:
    """
    Extracts and normalizes contract value to USD.
 
    OCDS stores amounts in the contract's native currency. We do a best-effort
    conversion using rough exchange rates for common currencies. For precise
    analysis, a live FX API would be used — for anomaly detection, order-of-
    magnitude accuracy is sufficient.
    """
    # OCDS schema: contract.value.amount and contract.value.currency
    value = contract.get("value") or contract.get("tender", {}).get("value", {})
    if not value:
        return None
 
    amount   = value.get("amount")
    currency = (value.get("currency") or "USD").upper()
 
    if amount is None:
        return None
 
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return None
 
    # Rough USD conversion rates (sufficient for spike detection)
    FX_TO_USD = {
        "USD": 1.0, "EUR": 1.08, "GBP": 1.26, "UAH": 0.027,
        "ETB": 0.018, "MMK": 0.00048, "CNY": 0.14,
        "PHP": 0.018, "NGN": 0.00065, "MDL": 0.056,
        "BRL": 0.20,
    }
    rate = FX_TO_USD.get(currency, 1.0)  # unknown currency → assume USD
    return round(amount * rate, 2)

def extract_award_date(contract: dict) -> Optional[date]:
    """Extracts the award/tender date from an OCDS contract."""
    date_fields = [
        contract.get("date"),
        contract.get("awardDate"),
        contract.get("tender", {}).get("tenderPeriod", {}).get("startDate"),
        contract.get("tender", {}).get("datePublished"),
    ]
    for field in date_fields:
        if field:
            try:
                return datetime.fromisoformat(
                    str(field)[:10]  # take first 10 chars: YYYY-MM-DD
                ).date()
            except ValueError:
                continue
    return None


def extract_vendor_name(contract: dict) -> Optional[str]:
    """Extracts the winning supplier/vendor name from an OCDS contract."""
    # Try awards array first
    awards = contract.get("awards", [])
    if awards:
        suppliers = awards[0].get("suppliers", [])
        if suppliers:
            return suppliers[0].get("name")
 
    # Try parties array with role "supplier"
    for party in contract.get("parties", []):
        if "supplier" in party.get("roles", []):
            return party.get("name")
 
    return None
 
 
# ─────────────────────────────────────────────
# OCDS FETCH FUNCTIONS
# ─────────────────────────────────────────────
 
def fetch_ocds_records(
    country_code: str,
    buyer_id: Optional[str],
    start_date: date,
    end_date: date,
    use_cache: bool = True,
    max_records: int = 500,
) -> list[dict]:
    """
    Fetches OCDS procurement records for a country/buyer in a date range.
 
    Returns a list of normalized contract dicts, each containing:
      contract_id, title, description, category, amount_usd, currency,
      award_date, vendor_name, buyer_name, country_code, raw (original OCDS)
 
    Tries the World Bank OCDS API first; falls back to country-specific
    portals if available; falls back to empty list if all sources fail.
    The pipeline is designed to handle procurement being unavailable —
    it's the weakest data source in terms of real-time coverage.
    """
    cache_key = _procurement_cache_key(country_code, buyer_id, start_date, end_date)
 
    if use_cache:
        cached = _read_procurement_cache(cache_key)
        if cached is not None:
            log.debug(f"Procurement cache hit: {country_code} {start_date}→{end_date}")
            return cached
 
    records = []
 
    # Try World Bank OCDS (most broadly available)
    try:
        records = _fetch_world_bank_ocds(country_code, buyer_id, start_date, end_date, max_records)
    except Exception as e:
        log.warning(f"World Bank OCDS failed for {country_code}: {e}")
 
    # Country-specific fallback for Ukraine (best coverage)
    if not records and country_code == "UA":
        try:
            records = _fetch_prozorro(buyer_id, start_date, end_date, max_records)
        except Exception as e:
            log.warning(f"Prozorro API failed: {e}")
 
    normalized = [_normalize_ocds_record(r, country_code) for r in records]
    normalized = [r for r in normalized if r is not None]
 
    if use_cache:
        _write_procurement_cache(cache_key, normalized)
 
    log.info(f"Procurement: {len(normalized)} records for {country_code} "
             f"{start_date}→{end_date}")
    return normalized
 
 
def _fetch_world_bank_ocds(
    country_code: str,
    buyer_id: Optional[str],
    start_date: date,
    end_date: date,
    max_records: int,
) -> list[dict]:
    """Fetches from World Bank OCDS dataset API."""
    params = {
        "where": f"countrycode='{country_code}' AND "
                 f"releasedate>='{start_date.isoformat()}' AND "
                 f"releasedate<='{end_date.isoformat()}'",
        "limit":  min(max_records, 100),
        "offset": 0,
    }
    if buyer_id:
        params["where"] += f" AND buyerid='{buyer_id}'"
 
    resp = requests.get(
        OCDS_ENDPOINTS["world_bank"],
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])
 
 
def _fetch_prozorro(
    buyer_id: Optional[str],
    start_date: date,
    end_date: date,
    max_records: int,
) -> list[dict]:
    """Fetches from Ukraine's Prozorro OpenProcurement API."""
    params = {
        "opt_fields": "tenderID,title,value,procuringEntity,dateModified,status",
        "limit":       min(max_records, 100),
        "offset":      0,
        "dateModified[gte]": start_date.isoformat(),
        "dateModified[lte]": end_date.isoformat(),
    }
    resp = requests.get(OCDS_ENDPOINTS["ukraine"], params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])
 
 
def _normalize_ocds_record(raw: dict, country_code: str) -> Optional[dict]:
    """
    Converts a raw OCDS record (from any source) into our standard format.
    Returns None if the record is missing critical fields.
    """
    award_date = extract_award_date(raw)
    amount_usd = extract_amount_usd(raw)
 
    # Skip records with no usable date or amount
    if award_date is None or amount_usd is None:
        return None
 
    return {
        "contract_id":  raw.get("ocid") or raw.get("id") or raw.get("tenderID", ""),
        "title":        raw.get("title") or raw.get("tender", {}).get("title", ""),
        "description":  raw.get("description") or raw.get("tender", {}).get("description", ""),
        "category":     categorize_contract(raw),
        "amount_usd":   amount_usd,
        "currency":     (raw.get("value") or {}).get("currency", "USD"),
        "award_date":   award_date,
        "vendor_name":  extract_vendor_name(raw),
        "buyer_name":   _extract_buyer_name(raw),
        "country_code": country_code,
        "raw":          raw,
    }
 
 
def _extract_buyer_name(raw: dict) -> Optional[str]:
    """Extracts buyer/procuring entity name from an OCDS record."""
    buyer = raw.get("buyer") or raw.get("procuringEntity", {})
    return buyer.get("name") if buyer else None
 
 
# ─────────────────────────────────────────────
# SPEND TIMESERIES
# ─────────────────────────────────────────────
 
def get_spend_timeseries(
    records: list[dict],
    category: str,
    group_by: str = "month",
) -> list[dict]:
    """
    Aggregates contract amounts into a monthly or weekly spend timeseries
    for a specific category.
 
    Args:
        records:   Normalized OCDS records (from fetch_ocds_records).
        category:  Category to filter on ("MILITARY", "MEDICAL", etc.)
        group_by:  "month" or "week"
 
    Returns:
        List of dicts: { period_start, period_label, total_usd, contract_count }
        Sorted by period_start ascending.
 
    The resulting timeseries is what the anomaly detector uses as input.
    """
    filtered = [r for r in records if r["category"] == category and r["amount_usd"] is not None]
 
    buckets: dict[str, dict] = {}
 
    for record in filtered:
        d = record["award_date"]
        if isinstance(d, str):
            d = datetime.fromisoformat(d).date()
 
        if group_by == "month":
            key = f"{d.year}-{d.month:02d}"
            period_start = date(d.year, d.month, 1)
        else:  # week
            monday = d - timedelta(days=d.weekday())
            key = monday.isoformat()
            period_start = monday
 
        if key not in buckets:
            buckets[key] = {
                "period_start":  period_start,
                "period_label":  key,
                "total_usd":     0.0,
                "contract_count": 0,
            }
        buckets[key]["total_usd"]      += record["amount_usd"]
        buckets[key]["contract_count"] += 1
 
    return sorted(buckets.values(), key=lambda x: x["period_start"])
 
 
def get_new_vendors(
    records: list[dict],
    category: str,
    baseline_records: list[dict],
) -> list[dict]:
    """
    Identifies vendors appearing in current records that weren't present
    in the baseline period for a given category.
 
    A new vendor in MILITARY or MEDICAL procurement is a notable signal —
    it suggests rapid contract-letting outside normal supply chains, which
    often indicates emergency procurement or operational escalation.
 
    Returns list of new vendor dicts with their contract details.
    """
    baseline_vendors = {
        r["vendor_name"] for r in baseline_records
        if r["category"] == category and r["vendor_name"]
    }
    current_vendors_contracts = [
        r for r in records
        if r["category"] == category
        and r["vendor_name"]
        and r["vendor_name"] not in baseline_vendors
    ]
    return current_vendors_contracts