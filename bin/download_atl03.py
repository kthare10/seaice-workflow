#!/usr/bin/env python3

"""
Download ICESat-2 ATL03 photon-level data from NASA Earthdata.

Uses the earthaccess library to authenticate and download ATL03 HDF5 granules
for a specified region and date range. Supports strong beams (gt1l, gt2l, gt3l).

Usage:
    python download_atl03.py --region ross_sea \
                              --start-date 2019-11-01 \
                              --end-date 2019-11-30 \
                              --output atl03_data.h5
"""

import argparse
import logging
import sys
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


def _write_track_bbox(h5_path, bbox_path="atl03_bbox.json", pad_deg=0.5):
    """
    Compute the bounding box of all photon tracks in an ATL03 HDF5 file
    and write it to a JSON file for use by downstream jobs (e.g. S2 download).

    Args:
        h5_path: Path to the merged ATL03 HDF5 file
        bbox_path: Output JSON path (default: atl03_bbox.json)
        pad_deg: Degrees of padding around the track (default: 0.5)
    """
    import json

    import h5py
    import numpy as np

    min_lat, max_lat = 90.0, -90.0
    min_lon, max_lon = 180.0, -180.0

    with h5py.File(h5_path, 'r') as h5:
        for gname in h5.keys():
            if not gname.startswith('granule_'):
                continue
            for beam in ['gt1l', 'gt2l', 'gt3l']:
                lat_path = f"{gname}/{beam}/lat_ph"
                lon_path = f"{gname}/{beam}/lon_ph"
                if lat_path in h5 and lon_path in h5:
                    lat = h5[lat_path][:]
                    lon = h5[lon_path][:]
                    min_lat = min(min_lat, float(np.min(lat)))
                    max_lat = max(max_lat, float(np.max(lat)))
                    min_lon = min(min_lon, float(np.min(lon)))
                    max_lon = max(max_lon, float(np.max(lon)))

    bbox = {
        "min_lon": round(min_lon - pad_deg, 4),
        "min_lat": round(min_lat - pad_deg, 4),
        "max_lon": round(max_lon + pad_deg, 4),
        "max_lat": round(max_lat + pad_deg, 4),
    }

    with open(bbox_path, 'w') as f:
        json.dump(bbox, f, indent=2)

    logger.info(f"Track bounding box: {bbox}")
    logger.info(f"Saved to {bbox_path}")


def download_atl03(region, start_date, end_date, output_file, granule_id=None, max_granules=None):
    """
    Download ATL03 granules from NASA Earthdata.

    Args:
        region: Region name or bounding box tuple
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        output_file: Output HDF5 file path
        granule_id: Optional specific granule ID to download
    """
    import os
    import time
    from pathlib import Path

    import earthaccess
    import h5py
    import numpy as np
    import requests
    from requests.exceptions import ConnectionError, ChunkedEncodingError, SSLError

    # Authenticate with NASA Earthdata
    # Prefer pre-generated bearer token (EARTHDATA_TOKEN) to bypass login endpoint
    # (urs.earthdata.nasa.gov may be unreachable from some networks like FABRIC)
    logger.info("Authenticating with NASA Earthdata...")
    token = os.environ.get("EARTHDATA_TOKEN")
    use_token = False
    if token:
        logger.info("Using pre-generated EARTHDATA_TOKEN (bypassing login endpoint)")
        use_token = True
    elif os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        logger.info("Using EARTHDATA_USERNAME/PASSWORD login")
        earthaccess.login(strategy="environment")
    else:
        raise RuntimeError(
            "NASA Earthdata credentials not set. Provide either:\n"
            "  - EARTHDATA_TOKEN (pre-generated bearer token), or\n"
            "  - EARTHDATA_USERNAME and EARTHDATA_PASSWORD\n"
            "Register at https://urs.earthdata.nasa.gov/"
        )

    # Get bounding box
    if isinstance(region, str):
        if region not in REGIONS:
            raise ValueError(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")
        bbox = REGIONS[region]
    else:
        bbox = region

    logger.info(f"Searching ATL03 granules for region bbox={bbox}")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Search for ATL03 granules (CMR search works without login)
    results = earthaccess.search_data(
        short_name="ATL03",
        bounding_box=bbox,
        temporal=(start_date, end_date),
    )

    if granule_id:
        results = [r for r in results if granule_id in str(r)]

    logger.info(f"Found {len(results)} ATL03 granules")

    if max_granules and len(results) > max_granules:
        logger.info(f"Limiting to {max_granules} granules (--max-granules)")
        results = results[:max_granules]

    if not results:
        logger.error("No ATL03 granules found for the specified criteria")
        sys.exit(1)

    # Download granules
    if use_token:
        # Direct download with bearer token — bypasses earthaccess.download()
        # which requires a full login session via urs.earthdata.nasa.gov
        logger.info("Downloading with bearer token (direct HTTPS)...")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        max_retries = 5
        base_delay = 5
        downloaded_files = []
        for i, granule in enumerate(results):
            urls = granule.data_links(access="external")
            if not urls:
                logger.warning(f"No download URL for granule {i}")
                continue
            url = urls[0]
            filename = Path(url).name
            # Skip already-downloaded granules (resume after partial failure)
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                logger.info(f"  [{i+1}/{len(results)}] Skipping {filename} (already exists)")
                downloaded_files.append(filename)
                continue

            logger.info(f"  [{i+1}/{len(results)}] Downloading {filename}...")

            for attempt in range(1, max_retries + 1):
                try:
                    resp = session.get(url, stream=True, timeout=300)
                    if resp.status_code != 200:
                        logger.warning(f"  [{i+1}/{len(results)}] HTTP {resp.status_code} for {url}")
                        break
                    with open(filename, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_files.append(filename)
                    logger.info(f"  [{i+1}/{len(results)}] Saved {filename} ({os.path.getsize(filename) / 1e6:.1f} MB)")
                    break
                except (SSLError, ConnectionError, ChunkedEncodingError) as e:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"  [{i+1}/{len(results)}] Attempt {attempt}/{max_retries} "
                        f"failed: {e}. Retrying in {delay}s..."
                    )
                    # Drop stale connections before retrying
                    session.close()
                    session = requests.Session()
                    session.headers.update({"Authorization": f"Bearer {token}"})
                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        logger.error(f"  [{i+1}/{len(results)}] All {max_retries} attempts failed for {filename}")
    else:
        downloaded_files = earthaccess.download(results, local_path=".")

    logger.info(f"Downloaded {len(downloaded_files)} files")

    # Merge into a single output HDF5 file
    strong_beams = ['gt1l', 'gt2l', 'gt3l']

    with h5py.File(output_file, 'w') as out_h5:
        granule_count = 0
        for fpath in downloaded_files:
            fpath = str(fpath)
            if not fpath.endswith('.h5'):
                continue
            logger.info(f"Processing granule: {fpath}")
            try:
                with h5py.File(fpath, 'r') as in_h5:
                    granule_grp = out_h5.create_group(f"granule_{granule_count:04d}")
                    # Copy metadata
                    if 'ancillary_data' in in_h5:
                        in_h5.copy('ancillary_data', granule_grp)
                    if 'orbit_info' in in_h5:
                        in_h5.copy('orbit_info', granule_grp)

                    for beam in strong_beams:
                        if beam not in in_h5:
                            continue
                        beam_grp = in_h5[beam]
                        if 'heights' not in beam_grp:
                            continue

                        out_beam = granule_grp.create_group(beam)
                        heights = beam_grp['heights']

                        # Copy photon-level data
                        for key in ['h_ph', 'lat_ph', 'lon_ph', 'signal_conf_ph',
                                    'delta_time', 'dist_ph_along']:
                            if key in heights:
                                out_beam.create_dataset(key, data=heights[key][:])

                        # Copy geolocation data if available
                        if 'geolocation' in beam_grp:
                            geo = beam_grp['geolocation']
                            out_geo = out_beam.create_group('geolocation')
                            for key in ['segment_id', 'segment_dist_x',
                                        'segment_ph_cnt', 'reference_photon_lat',
                                        'reference_photon_lon']:
                                if key in geo:
                                    out_geo.create_dataset(key, data=geo[key][:])

                    granule_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {fpath}: {e}")
                continue

        out_h5.attrs['n_granules'] = granule_count
        out_h5.attrs['region'] = region if isinstance(region, str) else str(bbox)
        out_h5.attrs['start_date'] = start_date
        out_h5.attrs['end_date'] = end_date

    logger.info(f"Merged {granule_count} granules into {output_file}")

    # Write track bounding box JSON so downstream jobs can use it
    _write_track_bbox(output_file)

    # Summary
    with h5py.File(output_file, 'r') as h5:
        total_photons = 0
        for gname in h5.keys():
            if not gname.startswith('granule_'):
                continue
            for beam in strong_beams:
                path = f"{gname}/{beam}/h_ph"
                if path in h5:
                    total_photons += h5[path].shape[0]
        logger.info(f"Total photons across all beams: {total_photons:,}")

    print(f"\n{'='*70}")
    print("ATL03 DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Granules: {granule_count}")
    print(f"Total photons: {total_photons:,}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download ICESat-2 ATL03 data from NASA Earthdata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for Ross Sea region
  %(prog)s --region ross_sea --start-date 2019-11-01 --end-date 2019-11-30

  # Download specific granule
  %(prog)s --region ross_sea --start-date 2019-11-01 --granule-id ATL03_20191101
        """
    )

    parser.add_argument("--region", type=str, required=True,
                        help="Region name (ross_sea, weddell_sea, beaufort_sea, arctic_ocean, southern_ocean)")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD), defaults to start_date + 30 days")
    parser.add_argument("--granule-id", type=str, default=None,
                        help="Specific ATL03 granule ID (optional)")
    parser.add_argument("--max-granules", type=int, default=None,
                        help="Maximum number of granules to download (default: all)")
    parser.add_argument("--output", type=str, default="atl03_data.h5",
                        help="Output HDF5 file (default: atl03_data.h5)")

    args = parser.parse_args()

    if not args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = start + timedelta(days=30)
        args.end_date = end.strftime("%Y-%m-%d")

    try:
        download_atl03(
            region=args.region,
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=args.output,
            granule_id=args.granule_id,
            max_granules=args.max_granules,
        )
        logger.info("ATL03 download completed successfully")
    except Exception as e:
        logger.error(f"Failed to download ATL03 data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
