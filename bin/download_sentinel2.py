#!/usr/bin/env python3

"""
Download coincident Sentinel-2 imagery for sea ice classification.

Fetches Sentinel-2 L2A scenes from Microsoft Planetary Computer that overlap
with ICESat-2 ATL03 tracks in the specified region and date range.

Usage:
    python download_sentinel2.py --region ross_sea \
                                  --start-date 2019-11-01 \
                                  --end-date 2019-11-30 \
                                  --output sentinel2_scenes.tar.gz
"""

import argparse
import logging
import os
import sys
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Predefined regions with bounding boxes [min_lon, min_lat, max_lon, max_lat]
REGIONS = {
    'ross_sea': (-180, -78, -150, -60),
    'weddell_sea': (-60, -78, 0, -60),
    'beaufort_sea': (-160, 68, -120, 80),
    'arctic_ocean': (-180, 65, 180, 90),
    'southern_ocean': (-180, -78, 180, -60),
}


def download_sentinel2(region, start_date, end_date, output_file, max_cloud_cover=30, max_scenes=10):
    """
    Download Sentinel-2 scenes from Planetary Computer.

    Args:
        region: Region name or bounding box tuple
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        output_file: Output tar.gz file path
        max_cloud_cover: Maximum cloud cover percentage (default: 30)
    """
    import numpy as np
    import planetary_computer
    import pystac_client
    import rasterio

    # Get bounding box
    if isinstance(region, str):
        if region not in REGIONS:
            raise ValueError(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")
        bbox = REGIONS[region]
    else:
        bbox = region

    logger.info(f"Searching Sentinel-2 scenes for bbox={bbox}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Max cloud cover: {max_cloud_cover}%")

    # Connect to Planetary Computer STAC API
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for Sentinel-2 L2A scenes
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    items = list(search.items())
    logger.info(f"Found {len(items)} Sentinel-2 scenes")

    if not items:
        logger.error("No Sentinel-2 scenes found for the specified criteria")
        sys.exit(1)

    # Sort by cloud cover to prefer clearest scenes
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))

    # Download scenes to temporary directory
    tmp_dir = Path("sentinel2_tmp")
    tmp_dir.mkdir(exist_ok=True)

    downloaded = []
    bands = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

    for i, item in enumerate(items[:max_scenes]):
        scene_id = item.id
        cloud_cover = item.properties.get("eo:cloud_cover", -1)
        logger.info(f"Downloading scene {i+1}/{min(len(items), max_scenes)}: {scene_id} (cloud: {cloud_cover:.1f}%)")

        scene_dir = tmp_dir / scene_id
        scene_dir.mkdir(exist_ok=True)

        for band_name in bands:
            if band_name not in item.assets:
                logger.warning(f"Band {band_name} not found in {scene_id}")
                continue

            asset = item.assets[band_name]
            href = asset.href
            out_path = scene_dir / f"{band_name}.tif"

            try:
                with rasterio.open(href) as src:
                    data = src.read()
                    profile = src.profile.copy()

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data)

                logger.info(f"  Saved {band_name}: {out_path}")
            except Exception as e:
                logger.warning(f"  Failed to download {band_name} from {scene_id}: {e}")
                continue

        downloaded.append(scene_dir)

    # Package into tar.gz
    logger.info(f"Packaging {len(downloaded)} scenes into {output_file}")
    with tarfile.open(output_file, "w:gz") as tar:
        for scene_dir in downloaded:
            tar.add(scene_dir, arcname=scene_dir.name)

    # Cleanup temporary files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Sentinel-2 data saved to {output_file}")

    print(f"\n{'='*70}")
    print("SENTINEL-2 DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Scenes downloaded: {len(downloaded)}")
    print(f"Bands: {', '.join(bands)}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 imagery from Planetary Computer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for Ross Sea region
  %(prog)s --region ross_sea --start-date 2019-11-01 --end-date 2019-11-30

  # With low cloud cover threshold
  %(prog)s --region ross_sea --start-date 2019-11-01 --max-cloud-cover 10
        """
    )

    parser.add_argument("--region", type=str, required=True,
                        help="Region name (ross_sea, weddell_sea, beaufort_sea, arctic_ocean, southern_ocean)")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD), defaults to start_date + 30 days")
    parser.add_argument("--max-cloud-cover", type=float, default=30,
                        help="Maximum cloud cover percentage (default: 30)")
    parser.add_argument("--max-scenes", type=int, default=10,
                        help="Maximum number of scenes to download (default: 10)")
    parser.add_argument("--output", type=str, default="sentinel2_scenes.tar.gz",
                        help="Output tar.gz file (default: sentinel2_scenes.tar.gz)")

    args = parser.parse_args()

    if not args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = start + timedelta(days=30)
        args.end_date = end.strftime("%Y-%m-%d")

    try:
        download_sentinel2(
            region=args.region,
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=args.output,
            max_cloud_cover=args.max_cloud_cover,
            max_scenes=args.max_scenes,
        )
        logger.info("Sentinel-2 download completed successfully")
    except Exception as e:
        logger.error(f"Failed to download Sentinel-2 data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
