"""
scripts/visualize_tiles.py — Before/After Tile Visualizer

A diagnostic CLI script for inspecting cached satellite tiles side by side.
Run this after the pipeline has fetched tiles to verify the data looks right
before running change detection.

Usage:
    python scripts/visualize_tiles.py --region eth_tigray \\
        --before 2021-01-01 --after 2021-04-01

    python scripts/visualize_tiles.py --cache-stats
    python scripts/visualize_tiles.py --list-cache

The output is a side-by-side matplotlib figure saved to:
    data/tile_cache/viz_{region}_{before}_{after}.png

WHY THIS EXISTS
───────────────
Before you trust any automated change score, you need to visually verify
that the tiles look correct. Common problems this catches:
  - Cloud cover the API's filter missed (cloud artifacts look like change)
  - Wrong region coordinates (fetching ocean instead of land)
  - Season-induced NDVI change (dry vs wet season) vs real land use change
  - No-data areas (black patches from satellite scan gaps)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from config import REGIONS_BY_ID, TILE_CACHE_DIR
from ingestion.satellite import (
    fetch_tile_pair_for_region,
    get_cache_stats,
    list_cached_tiles,
    BAND_NIR, BAND_RED, BAND_GREEN, BAND_SWIR1,
)


def compute_ndvi_display(arr: np.ndarray) -> np.ndarray:
    """
    Computes NDVI from an analysis band array and returns it as a
    display-ready single-channel float array in [-1, 1].

    NDVI = (NIR - Red) / (NIR + Red)

    Values interpretation:
      > 0.6  : Dense healthy vegetation (forest, crops)
      0.2–0.6: Sparse vegetation, grassland
      0.0–0.2: Bare soil, rock
      < 0.0  : Water, snow, clouds (negative reflectance difference)
    """
    nir = arr[:, :, BAND_NIR].astype(np.float32)
    red = arr[:, :, BAND_RED].astype(np.float32)
    denom = nir + red
    # Avoid division by zero in no-data pixels
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return ndvi


def make_true_color(arr: np.ndarray) -> np.ndarray:
    """
    Extracts R, G, B bands and gamma-corrects for display.
    Returns (H, W, 3) uint8 array suitable for imshow.
    """
    rgb = arr[:, :, [BAND_RED, BAND_GREEN, 0]]  # 0 = Blue
    # Gamma correction: boosts mid-range values for better visual contrast
    rgb = np.power(np.clip(rgb, 0, 1), 0.5)
    return (rgb * 255).astype(np.uint8)


def visualize_pair(
    tile_before: np.ndarray,
    tile_after: np.ndarray,
    meta_before: dict,
    meta_after: dict,
    region_id: str,
    output_path: str = None,
) -> str:
    """
    Creates a 2×2 figure comparing before/after tiles:
      Top-left:  True color before
      Top-right: True color after
      Bot-left:  NDVI before (green = vegetation)
      Bot-right: NDVI after  (change visible as color shift)

    Returns the path where the figure was saved.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise RuntimeError("matplotlib not installed. Run: pip install matplotlib")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Satellite Tile Comparison — {region_id}\n"
        f"Before: {meta_before['target_date']}   →   After: {meta_after['target_date']}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # True color images
    tc_before = make_true_color(tile_before)
    tc_after  = make_true_color(tile_after)
    axes[0, 0].imshow(tc_before)
    axes[0, 0].set_title(f"True Color — {meta_before['target_date']}", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tc_after)
    axes[0, 1].set_title(f"True Color — {meta_after['target_date']}", fontsize=10)
    axes[0, 1].axis("off")

    # NDVI maps — use a diverging colormap:
    # Red (negative NDVI) → Yellow (zero) → Dark Green (high NDVI)
    ndvi_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ndvi", ["#8B0000", "#FF4500", "#FFFF00", "#90EE90", "#006400"]
    )
    ndvi_before = compute_ndvi_display(tile_before)
    ndvi_after  = compute_ndvi_display(tile_after)

    im0 = axes[1, 0].imshow(ndvi_before, cmap=ndvi_cmap, vmin=-0.5, vmax=0.8)
    axes[1, 0].set_title(f"NDVI — {meta_before['target_date']}", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.03, label="NDVI")

    im1 = axes[1, 1].imshow(ndvi_after, cmap=ndvi_cmap, vmin=-0.5, vmax=0.8)
    axes[1, 1].set_title(f"NDVI — {meta_after['target_date']}", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.03, label="NDVI")

    # NDVI difference overlay (inset in bottom-right corner)
    ndvi_diff = ndvi_after - ndvi_before
    diff_range = max(abs(ndvi_diff.min()), abs(ndvi_diff.max()))
    if diff_range > 0:
        diff_ax = fig.add_axes([0.52, 0.08, 0.18, 0.18])
        diff_im = diff_ax.imshow(
            ndvi_diff, cmap="RdYlGn",
            vmin=-diff_range, vmax=diff_range
        )
        diff_ax.set_title("NDVI Δ", fontsize=8)
        diff_ax.axis("off")

    # Metadata annotation
    cache_labels = []
    if meta_before.get("cache_hit"):
        cache_labels.append("before: cached")
    if meta_after.get("cache_hit"):
        cache_labels.append("after: cached")

    fig.text(
        0.5, 0.01,
        f"Resolution: {meta_before.get('resolution_m', '?')}m/px  |  "
        f"Shape: {tile_before.shape[1]}×{tile_before.shape[0]}px  |  "
        + ("  |  ".join(cache_labels) if cache_labels else ""),
        ha="center", fontsize=8, color="gray"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output_path is None:
        cache_dir = Path(TILE_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            cache_dir / f"viz_{region_id}_{meta_before['target_date']}_{meta_after['target_date']}.png"
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Witness — Satellite Tile Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a before/after pair for Tigray
  python scripts/visualize_tiles.py --region eth_tigray \\
      --before 2021-01-15 --after 2021-04-15

  # Use lower resolution for a quick test (faster, smaller files)
  python scripts/visualize_tiles.py --region bra_amazon_arc \\
      --before 2020-07-01 --after 2021-07-01 --resolution 60

  # Show cache statistics
  python scripts/visualize_tiles.py --cache-stats

  # List all cached tiles
  python scripts/visualize_tiles.py --list-cache
        """
    )
    parser.add_argument("--region",     type=str, help="Region ID from config.py")
    parser.add_argument("--before",     type=str, help="Before date (YYYY-MM-DD)")
    parser.add_argument("--after",      type=str, help="After date (YYYY-MM-DD)")
    parser.add_argument("--resolution", type=int, default=60,
                        help="Resolution in metres/pixel (default: 60 for quick tests)")
    parser.add_argument("--output",     type=str, help="Output PNG path")
    parser.add_argument("--cache-stats",  action="store_true", help="Show cache statistics")
    parser.add_argument("--list-cache",   action="store_true", help="List all cached tiles")
    parser.add_argument("--force-fetch",  action="store_true",
                        help="Bypass cache and re-fetch from API")

    args = parser.parse_args()

    if args.cache_stats:
        stats = get_cache_stats()
        print("\n── Tile Cache Statistics ──────────────────")
        for k, v in stats.items():
            print(f"  {k:<20} {v}")
        print()
        return

    if args.list_cache:
        tiles = list_cached_tiles()
        if not tiles:
            print("Cache is empty.")
            return
        print(f"\n── Cached Tiles ({len(tiles)}) ──────────────────────")
        for t in tiles:
            print(f"  {t.get('target_date', '?')}  {t.get('evalscript_id', '?'):<12}  "
                  f"{t.get('size_mb', 0):.1f} MB  bbox={t.get('bbox', [])}")
        print()
        return

    if not all([args.region, args.before, args.after]):
        parser.print_help()
        sys.exit(1)

    if args.region not in REGIONS_BY_ID:
        print(f"ERROR: Unknown region '{args.region}'.")
        print(f"Valid regions: {', '.join(REGIONS_BY_ID.keys())}")
        sys.exit(1)

    date_before = date.fromisoformat(args.before)
    date_after  = date.fromisoformat(args.after)

    print(f"\n→ Fetching tile pair for region '{args.region}'")
    print(f"  Before: {date_before}   After: {date_after}")
    print(f"  Resolution: {args.resolution}m/px")
    print(f"  Cache: {'bypassed' if args.force_fetch else 'enabled'}")

    result = fetch_tile_pair_for_region(
        region_id=args.region,
        date_before=date_before,
        date_after=date_after,
        resolution_m=args.resolution,
    )

    if result is None:
        print("\n✗ Could not fetch tile pair — no cloud-free imagery available.")
        print("  Try different dates or increase --resolution for a larger search area.")
        sys.exit(1)

    tile_before, tile_after, meta_before, meta_after = result

    print(f"\n✓ Tiles fetched.")
    print(f"  Before: shape={tile_before.shape}, "
          f"cache={'hit' if meta_before.get('cache_hit') else 'miss'}")
    print(f"  After:  shape={tile_after.shape}, "
          f"cache={'hit' if meta_after.get('cache_hit') else 'miss'}")

    print(f"\n→ Generating visualization...")
    out_path = visualize_pair(
        tile_before, tile_after, meta_before, meta_after,
        region_id=args.region,
        output_path=args.output,
    )
    print(f"✓ Saved to: {out_path}")


if __name__ == "__main__":
    main()