#!/usr/bin/env python3

"""
Download Sentinel-2 imagery coincident with ICESat-2 ATL03 passes.

Follows Sec. III.A.3 / Table I of Iqrah et al. (IPDPSW 2025): for each ATL03
granule, Sentinel-2 L2A scenes are selected that overlap the track's bounding
box AND were acquired within a short time window of the ICESat-2 overpass
(the paper uses an 80-minute window; Table I pairs are 8-48 minutes apart).
Scenes are ranked by time difference, not by cloud cover.

Per scene, two Planetary Computer assets are saved:
  * ``visual`` - the 8-bit true-color image (TCI). The authors' color-based
    segmentation ([5], Sec. 3.2) operates on RGB imagery in HSV space, so the
    8-bit TCI is the matching input.
  * ``SCL``    - the L2A Scene Classification Layer, used by auto_label.py to
    mask cloud and cloud-shadow pixels.

Granule overpass times come from the track JSON written by download_atl03.py
(``atl03_bbox.json``: per-granule bbox, start_utc, end_utc). Without that file
the script falls back to a plain date-range search, which does NOT reproduce
the paper's coincidence constraint and logs a warning.

Usage:
    python download_sentinel2.py --bbox-file atl03_bbox.json \\
                                  --output sentinel2_scenes.tar.gz
"""

import argparse
import json
import logging
import shutil
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Predefined regions with bounding boxes [min_lon, min_lat, max_lon, max_lat]
REGIONS = {
    'ross_sea': (-180, -78, -140, -70),  # Iqrah et al. 2025, Sec. III.A.1
    'weddell_sea': (-60, -78, 0, -60),
    'beaufort_sea': (-160, 68, -120, 80),
    'arctic_ocean': (-180, 65, 180, 90),
    'southern_ocean': (-180, -78, 180, -60),
}

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
ASSETS = ["visual", "SCL"]
MAX_TIME_DIFF_MIN = 80      # paper Sec. III.A.3: "up to an 80-minute temporal window"
MAX_RETRIES = 4
RETRY_BASE_S = 5


def _parse_utc(s):
    """Parse an ISO-8601 UTC string (with or without trailing Z) to an aware datetime."""
    s = s.rstrip('Z')
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _with_retries(fn, what):
    """
    Call fn() with exponential backoff on transient failures.

    Args:
        fn: Zero-argument callable
        what: Description for log messages

    Returns:
        fn()'s return value

    Raises:
        The last exception if every attempt fails
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_S * (2 ** (attempt - 1))
            logger.warning(f"  {what}: attempt {attempt}/{MAX_RETRIES} failed ({e}); retrying in {delay}s")
            time.sleep(delay)


def _open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _search(catalog, bbox, start, end, max_cloud_cover):
    """Search Sentinel-2 L2A items in a bbox and UTC time window."""
    def do_search():
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        return list(search.items())
    return _with_retries(do_search, "STAC search")


def _download_scene(item, scene_dir, meta):
    """
    Save the TCI and SCL assets of a STAC item plus a metadata JSON.

    Args:
        item: pystac Item
        scene_dir: Destination directory
        meta: Dict written to meta.json (extended with CRS/transform here)

    Returns:
        True if every asset was saved
    """
    import rasterio

    scene_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for asset_name in ASSETS:
        if asset_name not in item.assets:
            logger.warning(f"  Asset {asset_name} not found in {item.id}")
            ok = False
            continue
        href = item.assets[asset_name].href
        out_path = scene_dir / f"{asset_name}.tif"

        def fetch(href=href, out_path=out_path, asset_name=asset_name):
            with rasterio.open(href) as src:
                data = src.read()
                profile = src.profile.copy()
                if asset_name == "visual":
                    meta['epsg'] = src.crs.to_epsg()
                    meta['transform'] = list(src.transform)[:6]
                    meta['shape'] = [src.height, src.width]
            profile.update(driver='GTiff', compress='deflate')
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(data)
            return out_path.stat().st_size

        try:
            size = _with_retries(fetch, f"{item.id}/{asset_name}")
            logger.info(f"  Saved {asset_name}: {out_path.name} ({size / 1e6:.1f} MB)")
        except Exception as e:
            logger.error(f"  Failed to download {asset_name} from {item.id}: {e}")
            ok = False

    with open(scene_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    return ok


def download_sentinel2(track_info, output_file, max_cloud_cover=30,
                       max_scenes_per_granule=10, max_time_diff_min=MAX_TIME_DIFF_MIN,
                       fallback_bbox=None, fallback_dates=None):
    """
    Download Sentinel-2 scenes coincident with each ATL03 granule.

    Args:
        track_info: Parsed atl03_bbox.json (may lack 'granules' for legacy files)
        output_file: Output tar.gz path
        max_cloud_cover: Maximum scene cloud cover percentage
        max_scenes_per_granule: Cap on scenes kept per granule
        max_time_diff_min: Maximum |S2 - IS2| acquisition difference in minutes
        fallback_bbox: bbox used when no per-granule info exists
        fallback_dates: (start, end) date strings for the fallback search
    """
    catalog = _open_catalog()
    tmp_dir = Path("sentinel2_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    granules = [g for g in (track_info or {}).get('granules', []) if 'start_utc' in g]
    downloaded = []
    n_granules_with_scenes = 0

    if granules:
        logger.info(
            f"{len(granules)} granule(s) with overpass times; searching S2 within "
            f"+/-{max_time_diff_min} min, cloud cover < {max_cloud_cover}%"
        )
        for g in granules:
            bbox = (g['min_lon'], g['min_lat'], g['max_lon'], g['max_lat'])
            t_start = _parse_utc(g['start_utc'])
            t_end = _parse_utc(g['end_utc'])
            t_mid = t_start + (t_end - t_start) / 2
            window = timedelta(minutes=max_time_diff_min)
            logger.info(f"{g['granule']}: IS2 {t_start.isoformat()} .. {t_end.isoformat()}, bbox={tuple(round(v, 2) for v in bbox)}")

            items = _search(catalog, bbox, t_start - window, t_end + window, max_cloud_cover)
            scored = []
            for item in items:
                if item.datetime is None:
                    continue
                diff_min = abs((item.datetime - t_mid).total_seconds()) / 60.0
                if diff_min <= max_time_diff_min:
                    scored.append((diff_min, item))
            scored.sort(key=lambda x: x[0])
            logger.info(f"  {len(items)} scene(s) in window, {len(scored)} within {max_time_diff_min} min")
            if not scored:
                logger.warning(f"  {g['granule']}: no coincident Sentinel-2 scene")
                continue
            n_granules_with_scenes += 1

            for diff_min, item in scored[:max_scenes_per_granule]:
                cloud = item.properties.get("eo:cloud_cover", -1)
                logger.info(f"  {item.id}: dt={diff_min:.1f} min, cloud={cloud:.1f}%")
                meta = {
                    'scene_id': item.id,
                    'granule': g['granule'],
                    'source_file': g.get('source_file', ''),
                    's2_datetime_utc': item.datetime.astimezone(timezone.utc).isoformat(),
                    'is2_start_utc': t_start.isoformat(),
                    'is2_end_utc': t_end.isoformat(),
                    'time_diff_min': diff_min,
                    'cloud_cover': cloud,
                    's2_tile': item.properties.get('s2:mgrs_tile', ''),
                }
                scene_dir = tmp_dir / f"{g['granule']}__{item.id}"
                if _download_scene(item, scene_dir, meta):
                    downloaded.append(scene_dir)
                else:
                    logger.warning(f"  {item.id}: incomplete download, scene skipped")
                    shutil.rmtree(scene_dir, ignore_errors=True)
    else:
        if fallback_bbox is None or fallback_dates is None:
            logger.error("Track JSON has no granule overpass times and no fallback bbox/dates given")
            _write_tar(output_file, [])
            sys.exit(1)
        logger.warning(
            "No per-granule overpass times available: falling back to a date-range "
            "search. This does NOT enforce the paper's 80-minute IS2/S2 coincidence."
        )
        start = datetime.strptime(fallback_dates[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(fallback_dates[1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        items = _search(catalog, fallback_bbox, start, end, max_cloud_cover)
        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        logger.info(f"Found {len(items)} scenes; keeping up to {max_scenes_per_granule}")
        for item in items[:max_scenes_per_granule]:
            meta = {'scene_id': item.id, 'granule': None,
                    's2_datetime_utc': item.datetime.astimezone(timezone.utc).isoformat() if item.datetime else None,
                    'cloud_cover': item.properties.get("eo:cloud_cover", -1),
                    's2_tile': item.properties.get('s2:mgrs_tile', '')}
            scene_dir = tmp_dir / item.id
            if _download_scene(item, scene_dir, meta):
                downloaded.append(scene_dir)
        n_granules_with_scenes = 1 if downloaded else 0

    _write_tar(output_file, downloaded)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*70}")
    print("SENTINEL-2 DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Granules with coincident scenes: {n_granules_with_scenes}/{max(len(granules), 1)}")
    print(f"Scenes downloaded: {len(downloaded)}")
    print(f"Assets: {', '.join(ASSETS)}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")

    if not downloaded:
        # Required source: the declared output exists (empty archive) so the
        # job is not held on stage-out, but the DAG sees the failure.
        logger.error("No Sentinel-2 scenes downloaded for any granule")
        sys.exit(1)


def _write_tar(output_file, scene_dirs):
    """Package scene directories into a tar.gz (empty archive if none)."""
    logger.info(f"Packaging {len(scene_dirs)} scene(s) into {output_file}")
    with tarfile.open(output_file, "w:gz") as tar:
        for scene_dir in scene_dirs:
            tar.add(scene_dir, arcname=scene_dir.name)


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 imagery coincident with ATL03 passes (Planetary Computer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Coincident scenes for every granule in the track JSON (workflow use)
  %(prog)s --bbox-file atl03_bbox.json --output sentinel2_scenes.tar.gz

  # Standalone date-range search (no IS2 coincidence constraint)
  %(prog)s --region ross_sea --start-date 2019-11-01 --end-date 2019-11-30
        """
    )

    parser.add_argument("--bbox-file", type=str, default=None,
                        help="Track JSON from download_atl03.py with per-granule bbox and overpass times")
    parser.add_argument("--region", type=str, default=None,
                        help="Region name for fallback search (ross_sea, weddell_sea, beaufort_sea, arctic_ocean, southern_ocean)")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Fallback bounding box as min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Fallback start date (YYYY-MM-DD); ignored when granule times are available")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Fallback end date (YYYY-MM-DD), defaults to start_date + 30 days")
    parser.add_argument("--max-time-diff-min", type=float, default=MAX_TIME_DIFF_MIN,
                        help=f"Max |S2 - IS2| acquisition difference in minutes (default: {MAX_TIME_DIFF_MIN}, paper Sec. III.A.3)")
    parser.add_argument("--max-cloud-cover", type=float, default=30,
                        help="Maximum scene cloud cover percentage (default: 30)")
    parser.add_argument("--max-scenes", type=int, default=10,
                        help="Maximum scenes per granule (default: 10)")
    parser.add_argument("--output", type=str, default="sentinel2_scenes.tar.gz",
                        help="Output tar.gz file (default: sentinel2_scenes.tar.gz)")

    args = parser.parse_args()

    track_info = None
    fallback_bbox = None
    if args.bbox_file:
        with open(args.bbox_file) as f:
            track_info = json.load(f)
        fallback_bbox = (track_info["min_lon"], track_info["min_lat"],
                         track_info["max_lon"], track_info["max_lat"])
        logger.info(f"Loaded track info from {args.bbox_file}")
    elif args.bbox:
        fallback_bbox = tuple(float(x) for x in args.bbox.split(","))
        if len(fallback_bbox) != 4:
            parser.error("--bbox must have 4 comma-separated values: min_lon,min_lat,max_lon,max_lat")
    elif args.region:
        if args.region not in REGIONS:
            parser.error(f"Unknown region: {args.region}. Available: {list(REGIONS.keys())}")
        fallback_bbox = REGIONS[args.region]
    else:
        parser.error("Either --bbox-file, --bbox, or --region is required")

    fallback_dates = None
    if args.start_date:
        end_date = args.end_date
        if not end_date:
            end_date = (datetime.strptime(args.start_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        fallback_dates = (args.start_date, end_date)

    try:
        download_sentinel2(
            track_info=track_info,
            output_file=args.output,
            max_cloud_cover=args.max_cloud_cover,
            max_scenes_per_granule=args.max_scenes,
            max_time_diff_min=args.max_time_diff_min,
            fallback_bbox=fallback_bbox,
            fallback_dates=fallback_dates,
        )
        logger.info("Sentinel-2 download completed successfully")
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Failed to download Sentinel-2 data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
