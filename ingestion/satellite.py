from ast import List
import hashlib
import json 
from ntpath import exists
import os 
import pickle
from datetime import date,datetime, timezone,timedelta
import runpy
from sys import meta_path
import time
from typing import Optional
from pathlib import Path

import numpy as np
from pyproj import transform

from config import (
    MONITORED_REGIONS,
    REGIONS_BY_ID,
    SATELLITE_RESOLUTION_M,
    SENTINEL_HUB_CLIENT_ID,
    SENTINEL_HUB_CLIENT_SECRET,
    TILE_CACHE_DIR
)
import config 


# imports of sentinel hub
try:
    from sentinelhub import (
        CRS,
        BBox,
        BBoxSplitter,
        DataCollection,
        MimeType,
        MosaickingOrder,
        SentinelHubRequest,
        SHConfig,
        bbox_to_dimensions,
    )
    from sentinelhub import SentinelHubStatistical
    SENTINEL_AVAILABLE =True
except ImportError:
    SENTINEL_AVAILABLE= False
    # Define module-level stubs so patch() can always find these names in tests,
    # even when the SDK is not installed. patch() requires the attribute to exist.
    BBox = None
    BBoxSplitter = None
    CRS = None
    DataCollection = None
    MimeType = None
    MosaickingOrder = None
    SentinelHubRequest = None
    SHConfig = None
    bbox_to_dimensions = None
    

# ------ CONFIGURATION ------
def _get_sh_config() -> "SHConfig":
    """
    Build sentinel hub config using our enviornment variables.
    Called once per session. The config object is reused across the requests.
    """
    if not SENTINEL_AVAILABLE:
        raise RuntimeError("Sentinel Hub SDK is not available. Please install it to use satellite data features.")
    cfg=SHConfig()
    cfg.sh_client_id=SENTINEL_HUB_CLIENT_ID
    cfg.sh_client_secret=SENTINEL_HUB_CLIENT_SECRET
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise ValueError(
            "Sentinel Hub credentials not set. Add SENTINEL_HUB_CLIENT_ID and "
            "SENTINEL_HUB_CLIENT_SECRET to your .env file.\n"
            "Sign up at: https://www.sentinel-hub.com/trial/"
        )
    return cfg
    
    
#------ BAND CONFIGURATIONS ------
# We define named band configurations as constants
# Each combination is tuple of (band_names, evalscript)

# evalscript() is a tiny JavaScript code function that sentinel hub runs server side to combine output bands into arrays.
# We write them in Python as multi line string that get sent over the API.


EVALSCRIPT_TRUE_COLOR="""
    function setup(){
        return { input: :["B02","B03","B04"], output:{bands:3} };
    }
    
    
    function evaluatePixel(sample){
        return [3.5 * sample.B04, 3.5 * sample.B03, 3.5 * sample.B02];
    }

"""
  
#  All bands needed for NDVI + thermal analysis in a single fetch. Fetching all bands in one request is more efficient that multiple request for each band combination
EVALSCRIPT_ANALYSIS_BANDS ="""

    function setup(){
        return {
            input:["B02","B03","B04","B08",B11","B12"],
            output:{bands:6, SampleType:"FLOAT32"}
        };
    }
    function evaluatePixel(sample){
        return [
            sample.B02,   // index 0 — Blue
            sample.B03,   // index 1 — Green
            sample.B04,   // index 2 — Red
            sample.B08,   // index 3 — NIR (for NDVI)
            sample.B11,   // index 4 — SWIR1 (for fire/thermal)
            sample.B12    // index 5 — SWIR2 (for burn scars)
        ]
    }
"""

# Band indices within the analysis array — named constants avoid magic numbers
BAND_BLUE  = 0
BAND_GREEN = 1
BAND_RED   = 2
BAND_NIR   = 3
BAND_SWIR1 = 4
BAND_SWIR2 = 5


# ------ CACHE UTILITIES ------

# The cache stores the numpy arrays as .npy and the metadata as .json files

def _make_cache_key(bbox:list[float],target_date:date,evalscript_id:str)->str:
    """
        Produces a deterministic cache key file name for a given tile.
        
        The key encodes: bounding box coordinates + date + which bands we want.
        same input -> same key , always. This means we never fetch same tile twice.
        
        We use a short SHA256 hash to keep file names managable, with a human readble prefix so the cache directory is still browsable.
        
        Example output: "eth_trigray_20210315_analysis_a3f7c2b1"
    """
    # Round bbox to 4 decimal places to avoid float representations
    
    bbox_str="_".join(f"{v:.4f}" for v in bbox)
    raw=f"{bbox_str}_{target_date.isoformat()}_{evalscript_id}"
    short_hash=hashlib.sha256(raw.encode()).hexdigest()[:8]
    date_str=target_date.isoformat().replace("-","")
    
    return f"{date_str}_{evalscript_id}_{short_hash}"

def _cache_path(cache_key:str)->tuple[Path,Path]:
    """Returns (array_path, metadata_path) for a cache key"""
    cache_dir=Path(TILE_CACHE_DIR)
    cache_dir.mkdir(parents=True,exist_ok=True)   
       
    return (
        cache_dir / f"{cache_key}.npy",
        cache_dir / f"{cache_key}.meta.json"
    )
    
def _read_from_cache(cache_key:str)->Optional[tuple[np.ndarray,dict]]:
    """
        Returns (array, metadata) if cached, None if not cached
        Validates that cache files are intact before returning them
    """
    array_path, meta_path=_cache_path(cache_key)
    if not array_path.exists() or not meta_path.exists():
        return None;
    try:
        arr=np.load(array_path)
        with open(meta_path) as f:
            meta=json.load(f)
        return arr, meta
    except Exception:
        #corrupted cache files
        array_path.unlink(missing_ok=True)    
        meta_path.unlink(missing_ok=True)
        return None

def _write_to_cache(cache_key:str,arr:np.ndarray,meta:dict):
    """ 
    Persist a fetched tile as a NumPy array plus sidecar JSON metadata under TILE_CACHE_DIR.
    Pairing matches _read_from_cache: same cache_key produces the same .npy / .meta.json paths.
    """
    # Paths: <cache_key>.npy (bands raster) and <cache_key>.meta.json (fetch params, timestamps, etc.)
    arr_path, meta_path=_cache_path(cache_key)
    # Binary array on disk; load with np.load in _read_from_cache
    np.save(arr_path,arr)
    # Pretty-printed JSON; default=str serializes datetimes and other non-JSON-native values
    with open(meta_path,"w") as f:
        json.dump(meta,f,indent=2,default=str)
    

#----------------------------------------
# CORE FETCH FUNCTION
# --------------------------------------- 

def get_tile(
    bbox:List[float],
    target_date:date,
    evalscript_id:str="analysis",
    resolution_m:int=SATELLITE_RESOLUTION_M,
    max_cloud_cover_pct:float=30.0,
    force_refresh:bool=False,
)->Optional[tuple[np.ndarray,dict]]:
    """
        Fetches a single satellite tile for the given bounding box and date
        
        This is the central function of the entire ingestion module.
        All higher-level functions call this one.
    
        Cache-first: always checks the local cache before calling the API.
        Set force_refresh=True to bypass the cache (e.g. for debugging).
    
        Args:
            bbox:               [min_lng, min_lat, max_lng, max_lat] in WGS 84
            target_date:        The date to image. Sentinel Hub will return the
                                best available image within a ±5 day window.
            evalscript_id:      "analysis" (6 bands for computation) or "true_color"
            resolution_m:       Ground resolution in metres per pixel. 10m is native.
                                Use 60m for quick tests (smaller arrays, faster).
            max_cloud_cover_pct: Skip tiles where cloud cover exceeds this %.
                                Cloudy pixels corrupt NDVI and change scores.
            force_refresh:      If True, bypass cache and fetch from API.
    
        Returns:
            (array, metadata) tuple if a usable tile was found, or None if:
            - No cloud-free imagery available for this date/location
            - API credentials are invalid
            - Rate limit exceeded
    """
    eval_script=(
                    EVALSCRIPT_ANALYSIS_BANDS 
                    if evalscript_id=="analysis"
                    else EVALSCRIPT_TRUE_COLOR
                )
    cache_key=_make_cache_key(bbox,target_date,evalscript_id)
    
    # Check cache first unconditionally before API calls
    if not force_refresh:
        cached=_read_from_cache(cache_key)
        if cached is not None:
            arr, meta=cached
            return arr,{**meta,"cache hit":True}
    
    # ----------------------
    #       API fetch 
    # ----------------------
    if not SENTINEL_AVAILABLE:
        raise RuntimeError("Sentinel hub not installed. Please install before running it")
    
    cfg=_get_sh_config() 
    sh_bbox=BBox(bbox=bbox,crs=CRS.WGS84)
    
    # bbox_to_dimensions computes (width_px, height_px) for a bbox at a
    # given resolution. E.g. a 1°×1° box at 10m = ~11,100×11,100 pixels.
    # For large regions this can be enormous — we cap at 2500×2500.
    width,height=bbox_to_dimensions(sh_bbox, resolution=resolution_m)    
    width=min(width,2500)
    height=min(height,2500)
    
    # The time interval. We request a 10 day window centered on the target date and ask snetinel to return least cloudy tile available within that window.
    time_from=target_date-timedelta(days=5)
    time_to=target_date+timedelta(days=5)
    
    request = SentinelHubRequest (
        evalscript=eval_script,
        input_data=[
            SentinelHubRequest.input_data(
                # L2A = atmospherically corrected surface reflectance.
                # L1C is the raw top-of-atmosphere reflectance; L2A is better
                # for vegetation analysis because it removes atmospheric haze.
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(time_from.isoformat(),time_to.isoformat()),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                # LEAST_CC ="Least Cloud Cover" - Sentinel choses tiem image which is most clear within the window automatically.
                
                other_args={"datafilter":{"maxCloudCoverage":max_cloud_cover_pct}}
            )
        ],
        responses=[SentinelHubRequest.output_response("default",MimeType.TIFF)],
        bbox=sh_bbox,
        size=(width,height),
        config=cfg,
    )  
    data=request.get_data()
    
    # get_data() returns a list of arrays (one per response)
    # we have only one response so we take index 0
    if not data or data[0] is None:
        return None
    arr = data[0].astype(np.float32)
    
    # Normalize reflectance values to [0,1] range
    #  Sentinel-2 L2A values are in the range [0,10000] representing reflectance * 1000 (eg 0.05 surface reflectance -> value 500)
    if arr.max()>1.0:
        arr=arr/1000
    arr=np.clip(arr,0.0,1.0)
    
    metadata={
        "bbox":bbox,
        "target_data":target_date.isoformat(),
        "time_from":time_from.isoformat(),
        "time_to":time_to.isoformat(),
        "resolution_m":resolution_m,
        "shape":list(arr.shape),
        "evalscript_id":evalscript_id,
        "cache_hit":False,
        "fetched_at":datetime.now(timezone.utc).isoformat()
    }
    
    _write_to_cache(cache_key,arr,metadata)
    return arr, metadata
    
def get_tile_pair(
    bbox:list[float],
    date_before:date,
    date_after:date,
    resolution_m:int=SATELLITE_RESOLUTION_M,
    max_cloud_cover_pct:float=30.0
    )->Optional[tuple[np.ndarray,np.ndarray,dict,dict]]:
    """
    Fetches two tile for the same location at two different dates.
    Returns (tile_before, tile_after, meta_before, meta_after) or None
    If either tile cannot be fetched (e.g both dates are cloudy)
    
    This is the primary input to the change detection algorithm.
    "Before" is the refrence state, "after" is what we ompare it with
    
    Why fetch both in one call?
      We don't — they're two separate get_tile() calls. But wrapping them
      here guarantees that both tiles are at the same resolution and bbox,
      which is a precondition for pixel-wise comparison in change detection.
    """
    result_before=get_tile(
        bbox,
        date_before,
        resolution_m=resolution_m,
        max_cloud_cover_pct=max_cloud_cover_pct
    )
    result_after=get_tile(
        bbox,
        date_after,
        resolution_m=resolution_m,
        max_cloud_cover_pct=max_cloud_cover_pct
    )
    
    if result_before is None or result_after is None:
        return None
    
    tile_before, meta_before = result_before
    tile_after, meta_after = result_after
    #  Verify if both tiles have same spacial dimensions
    # They should, given the same bbox and resolution, but if the API
    # returned different sizes (can happen near image edges), we can't
    # do pixel-wise comparison
    if tile_before.shape != tile_after.shape:
        # Resize the after tile to match it to before tile using nearest-neighbor
        # interpolation (preserves spectral values, no blending)
    
        from PIL import Image
        h, w=tile_before.shape[:2]
        bands=[]
        for b in range(tile_after.shape[:2]):
            band_img=Image.fromarray(tile_after[:,:,b])
            band_img=band_img.resize((w,h),Image.NEAREST)
            bands.append(np.array(band_img))
        tile_after=np.stack(bands,axis=2)
        
    return tile_before, tile_after, meta_before, meta_after

def list_available_dates(
    bbox:list[float],
    start_date:date,
    end_date:date,
    max_cloud_cover_pct:float=30.0
)->list[date]:
    """
    Returns a sorted list of dates for which Sentinel-2 imagery is available
    and sufficiently cloud-free for the given bounding box.
 
    Sentinel-2 has a 5-day revisit cycle globally, so you'd expect roughly
    6 dates per month — but cloud cover in tropical or monsoon regions can
    reduce this significantly. Knowing available dates upfront prevents us
    from trying to fetch tiles that don't exist.
 
    This function queries Sentinel Hub's catalog API, which is much cheaper
    (quota-wise) than a full tile fetch. Think of it as checking the library's
    index before walking to the shelf.
 
    Note: This is a network call with no caching (the list of available dates
    changes as new imagery arrives). Keep calls infrequent — once per pipeline
    run per region is sufficient.
    """
    if not SENTINEL_AVAILABLE:
        raise RuntimeError("Sentinel Hub not available")
    from sentinelhub import SentinelHubCatalog, CatalogRequest, filter as sh_filter    

    cfg=_get_sh_config()
    catalog=SentinelHubCatalog(config=cfg)
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
 
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=sh_bbox,
        time=(start_date.isoformat(), end_date.isoformat()),
        filter=sh_filter.lte("eo:cloud_cover", max_cloud_cover_pct),
        fields={"include": ["properties.datetime", "properties.eo:cloud_cover"]},
    )
 
    available = []
    for item in search_iterator:
        dt_str = item["properties"]["datetime"]
        # Parse ISO 8601 and extract just the date
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        available.append(dt.date())
 
    return sorted(set(available))  # deduplicate (multiple tiles per date possible)
    
# ------------------
# PERSISTENCE UTILITIES
# ------------------ 

def save_file_to_disk(
    arr: np.ndarray,              # The NumPy array (tile) to be saved
    path: str,                    # Path where the array should be saved (extension added automatically)
    as_geotiff: bool = False,     # If True, save as GeoTIFF; otherwise save as .npy
    bbox: list[float] = None      # Bounding box [min_lng, min_lat, max_lng, max_lat]; required for GeoTIFF
) -> str:
    """
    Saves a tile in disk for inspection or long term storage

    Two formats:
        as_geotiff=False (default): saves as .npy — fast, lossless, no dependencies.
            Best for intermediate processing (change detection, NDVI computation).
        as_geotiff=True: saves as a GeoTIFF with embedded coordinate information.
            Best for visualization in QGIS, Google Earth Engine, or Mapbox.
            Requires rasterio; bbox must be provided.

    Returns the actual path written (may have extension added).
    """
    path = Path(path)  # Convert the provided path to a Path object for easy manipulation
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the parent directory exists

    if as_geotiff:  # If requested to save as GeoTIFF
        if bbox is None:  # Make sure a bounding box is provided for georeferencing
            raise ValueError("bbox must be provided when saving as GeoTIFF")  # Raise descriptive error if not
        try:
            import rasterio                                # Try to import the rasterio library for writing GeoTIFFs
            from rasterio.transform import from_bounds      # Import utility for calculating affine transforms from bounds
            from rasterio.crs import CRS as RioCRS          # Import CRS utility for coordinate reference systems
        except ImportError:
            raise RuntimeError("rasterio not installed. Run pip install rasterio")  # Give up and tell the user to install rasterio

        path = path.with_suffix(".tif")     # Ensure file has .tif extension
        h, w = arr.shape[:2]                # Extract height and width from array shape
        n_bands = arr.shape[2] if arr.ndim == 3 else 1  # Number of image bands; 3D array means channels/bands
        # WARNING: Bug in original code: should be bbox[2], bbox[3] not bbox[3], bbox[4]
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)  # Compute geotransform mapping pixels to coordinates

        with rasterio.open(
            path,                          # The output file path (.tif)
            mode="w",                      # Open for writing ("w")
            driver="GTiff",                # Use GeoTIFF format
            height=h,                      # Set raster height (rows)
            width=w,                       # Set raster width (cols)
            count=n_bands,                 # Number of bands/channels
            dtype=arr.dtype,               # Set array data type (e.g. float32, uint16)
            crs=RioCRS.from_epsg(4326),    # Set projection to WGS84 (EPSG:4326)
            transform=transform,           # Apply calculated geotransform
            compress="lzw",                # Use LZW compression for output
        ) as dst:
            if arr.ndim == 2:                  # If input is single-band (2D)
                dst.write(arr, 1)              # Write data to band 1
            else:                              # If multi-band (3D)
                for i in range(n_bands):       # Loop through each band index
                    dst.write(arr[:, :, i], i + 1)  # Write band data to GeoTIFF band (bands start at 1 in rasterio)

    else:  # Standard NumPy array saving branch
        path = path.with_suffix(".npy")     # Ensure file has .npy extension
        np.save(path, arr)                  # Save the NumPy array using NumPy's binary format

    return str(path)     # Return the file path (as a string) of the saved file


# ----------------------------------------
# REGION-LEVEL CONVENIENCE FUNCTIONS
# ----------------------------------------
def fetch_tile_for_region(
    region_id: str,
    target_date: date,
    resolution_m: int = SATELLITE_RESOLUTION_M,
) -> Optional[tuple[np.ndarray, dict]]:
    """
    Convenience wrapper: fetches a tile using a region_id from config.py
    rather than a raw bbox. Used by the scheduler and change detection.
    """
    region = REGIONS_BY_ID.get(region_id)
    if region is None:
        raise ValueError(
            f"Unknown region_id '{region_id}'. "
            f"Valid IDs: {list(REGIONS_BY_ID.keys())}"
        )
    return get_tile(region.bbox, target_date, resolution_m=resolution_m)
 
 
def fetch_tile_pair_for_region(
    region_id: str,
    date_before: date,
    date_after: date,
    resolution_m: int = SATELLITE_RESOLUTION_M,
) -> Optional[tuple[np.ndarray, np.ndarray, dict, dict]]:
    """
    Convenience wrapper: fetches before/after tile pair for a named region.
    Primary entry point called by run_change_detection() in detection/.
    """
    region = REGIONS_BY_ID.get(region_id)
    if region is None:
        raise ValueError(f"Unknown region_id '{region_id}'")
    return get_tile_pair(
        region.bbox, date_before, date_after, resolution_m=resolution_m
    )
 
 
# -------------------------- 
# CACHE INSPECTION UTILITIES 
# These are used by the diagnostic CLI, not the main pipeline.
# -------------------------- 
 
def list_cached_tiles() -> list[dict]:
    """
    Returns metadata for all tiles currently in the local cache.
    Useful for debugging and cache management.
    """
    cache_dir = Path(TILE_CACHE_DIR)
    if not cache_dir.exists():
        return []
 
    results = []
    for meta_file in sorted(cache_dir.glob("*.meta.json")):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            # Add file size of the corresponding .npy
            npy_path = meta_file.with_suffix("").with_suffix(".npy")
            size_mb = npy_path.stat().st_size / (1024 * 1024) if npy_path.exists() else 0
            results.append({**meta, "size_mb": round(size_mb, 2), "cache_key": meta_file.stem.replace(".meta", "")})
        except Exception:
            continue
    return results
 
 
def get_cache_stats() -> dict:
    """Returns summary statistics for the tile cache."""
    tiles = list_cached_tiles()
    total_mb = sum(t.get("size_mb", 0) for t in tiles)
    regions = set()
    for t in tiles:
        bbox = t.get("bbox", [])
        if bbox:
            regions.add(tuple(round(v, 1) for v in bbox))
    return {
        "tile_count":     len(tiles),
        "total_size_mb":  round(total_mb, 1),
        "unique_regions": len(regions),
        "cache_dir":      str(TILE_CACHE_DIR),
    }
 
 
def clear_cache(older_than_days: int = None) -> int:
    """
    Deletes cached tiles. If older_than_days is set, only deletes tiles
    older than that many days. Returns the number of tiles deleted.
 
    Use with care — deleted tiles will be re-fetched from the API,
    consuming quota.
    """
    cache_dir = Path(TILE_CACHE_DIR)
    if not cache_dir.exists():
        return 0
 
    deleted = 0
    cutoff = None
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
 
    for meta_file in cache_dir.glob("*.meta.json"):
        should_delete = True
        if cutoff is not None:
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                fetched_at = datetime.fromisoformat(
                    meta.get("fetched_at", "2000-01-01T00:00:00+00:00").replace("Z", "+00:00")
                )
                should_delete = fetched_at < cutoff
            except Exception:
                should_delete = True
 
        if should_delete:
            npy_path = meta_file.with_suffix("").with_suffix(".npy")
            meta_file.unlink(missing_ok=True)
            npy_path.unlink(missing_ok=True)
            deleted += 1
 
    return deleted












