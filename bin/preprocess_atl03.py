#!/usr/bin/env python3

"""
Preprocess ICESat-2 ATL03 photon data into 2m along-track segments.

Reads ATL03 HDF5 data, filters for high-confidence signal photons on sea ice,
resamples to 2m along-track bins, and computes statistical features per segment.

Based on: "Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data" (Iqrah et al., IPDPSW 2025)

Usage:
    python preprocess_atl03.py --input atl03_data.h5 \
                                --output atl03_preprocessed.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STRONG_BEAMS = ['gt1l', 'gt2l', 'gt3l']
BIN_SIZE_M = 2.0  # 2-meter along-track resolution
MIN_SIGNAL_CONF = 3  # High-confidence signal photons on sea ice surface type


def extract_beam_photons(h5_file, granule_key, beam):
    """
    Extract photon data from a single beam in a granule.

    Args:
        h5_file: Open h5py File object
        granule_key: Granule group name
        beam: Beam name (e.g., 'gt1l')

    Returns:
        DataFrame with photon-level data, or None if beam is not available
    """
    import h5py
    import numpy as np
    import pandas as pd

    beam_path = f"{granule_key}/{beam}"
    if beam_path not in h5_file:
        return None

    beam_grp = h5_file[beam_path]
    required_keys = ['h_ph', 'lat_ph', 'lon_ph', 'signal_conf_ph']
    for key in required_keys:
        if key not in beam_grp:
            logger.warning(f"Missing {key} in {beam_path}")
            return None

    h_ph = beam_grp['h_ph'][:]
    lat_ph = beam_grp['lat_ph'][:]
    lon_ph = beam_grp['lon_ph'][:]
    signal_conf = beam_grp['signal_conf_ph'][:]

    # Handle multi-dimensional signal_conf (photons x surface_types)
    # ATL03 surface types: 0=land, 1=ocean, 2=sea_ice, 3=land_ice, 4=inland_water
    if signal_conf.ndim == 2:
        signal_conf = signal_conf[:, 2]  # Sea ice surface type
    elif signal_conf.ndim == 1:
        pass  # Already 1D
    else:
        logger.warning(f"Unexpected signal_conf shape: {signal_conf.shape}")
        return None

    # Get along-track distance if available
    if 'dist_ph_along' in beam_grp:
        dist_along = beam_grp['dist_ph_along'][:]
    else:
        # Compute approximate along-track distance from lat/lon
        dlat = np.diff(lat_ph, prepend=lat_ph[0])
        dlon = np.diff(lon_ph, prepend=lon_ph[0])
        step_dist = np.sqrt((dlat * 111000)**2 + (dlon * 111000 * np.cos(np.radians(lat_ph)))**2)
        dist_along = np.cumsum(step_dist)

    df = pd.DataFrame({
        'lat': lat_ph,
        'lon': lon_ph,
        'h_ph': h_ph,
        'signal_conf': signal_conf,
        'dist_along': dist_along,
        'beam': beam,
        'granule': granule_key,
    })

    return df


def filter_signal_photons(df, min_conf=MIN_SIGNAL_CONF):
    """
    Filter for high-confidence signal photons.

    Args:
        df: DataFrame with photon data
        min_conf: Minimum signal confidence threshold

    Returns:
        Filtered DataFrame
    """
    import pandas as pd

    n_before = len(df)
    df = df[df['signal_conf'] >= min_conf].copy()
    n_after = len(df)
    logger.info(f"Filtered {n_before} -> {n_after} photons (signal_conf >= {min_conf})")
    return df


def resample_to_segments(df, bin_size=BIN_SIZE_M):
    """
    Resample photon data into along-track segments of specified size.

    Computes per-segment: mean/median/std height, photon count, background rate.

    Args:
        df: DataFrame with filtered photon data
        bin_size: Along-track bin size in meters

    Returns:
        DataFrame with segment-level features
    """
    import numpy as np
    import pandas as pd

    if df.empty:
        return pd.DataFrame()

    # Create along-track bins
    dist_min = df['dist_along'].min()
    dist_max = df['dist_along'].max()
    bins = np.arange(dist_min, dist_max + bin_size, bin_size)

    df = df.copy()
    df['bin_idx'] = np.digitize(df['dist_along'], bins) - 1

    segments = []
    for bin_idx, group in df.groupby('bin_idx'):
        if bin_idx < 0 or bin_idx >= len(bins) - 1:
            continue

        h_values = group['h_ph'].values

        # First-photon bias correction: remove lowest photon if > 5 photons
        if len(h_values) > 5:
            h_values = np.sort(h_values)[1:]  # Remove lowest (first-return bias)

        segment = {
            'lat': group['lat'].mean(),
            'lon': group['lon'].mean(),
            'along_track_dist': bins[bin_idx] + bin_size / 2,
            'mean_h': np.mean(h_values),
            'median_h': np.median(h_values),
            'std_h': np.std(h_values) if len(h_values) > 1 else 0.0,
            'photon_count': len(group),
            'bg_rate': _estimate_background_rate(group),
            'beam': group['beam'].iloc[0],
            'granule': group['granule'].iloc[0],
        }
        segments.append(segment)

    return pd.DataFrame(segments)


def _estimate_background_rate(photon_group):
    """
    Estimate background photon rate for a segment.

    Uses the spread of photon heights to estimate noise level.
    Higher spread relative to signal indicates more background noise.

    Args:
        photon_group: DataFrame group for a single segment

    Returns:
        Estimated background rate (photons per meter height range)
    """
    import numpy as np

    h_values = photon_group['h_ph'].values
    if len(h_values) < 2:
        return 0.0

    h_range = np.ptp(h_values)
    if h_range == 0:
        return 0.0

    # Count photons outside 2-sigma of the median as background
    median_h = np.median(h_values)
    mad = np.median(np.abs(h_values - median_h))
    sigma = 1.4826 * mad  # MAD to std conversion

    if sigma == 0:
        return 0.0

    bg_mask = np.abs(h_values - median_h) > 2 * sigma
    bg_count = np.sum(bg_mask)

    return bg_count / h_range


def preprocess_atl03(input_file, output_file):
    """
    Full preprocessing pipeline: extract, filter, resample.

    Args:
        input_file: Input HDF5 file with ATL03 data
        output_file: Output CSV file
    """
    import h5py
    import numpy as np
    import pandas as pd

    logger.info(f"Reading ATL03 data from {input_file}")

    all_segments = []

    with h5py.File(input_file, 'r') as h5:
        granule_keys = [k for k in h5.keys() if k.startswith('granule_')]
        logger.info(f"Found {len(granule_keys)} granules")

        for granule_key in granule_keys:
            for beam in STRONG_BEAMS:
                logger.info(f"Processing {granule_key}/{beam}")

                # Extract photons
                photons = extract_beam_photons(h5, granule_key, beam)
                if photons is None or photons.empty:
                    logger.warning(f"No data for {granule_key}/{beam}")
                    continue

                logger.info(f"  Raw photons: {len(photons):,}")

                # Filter signal photons
                photons = filter_signal_photons(photons)
                if photons.empty:
                    continue

                # Resample to 2m segments
                segments = resample_to_segments(photons)
                if not segments.empty:
                    all_segments.append(segments)
                    logger.info(f"  Segments: {len(segments):,}")

    if not all_segments:
        logger.error("No segments produced from any beam/granule")
        sys.exit(1)

    result = pd.concat(all_segments, ignore_index=True)
    result.to_csv(output_file, index=False)
    logger.info(f"Preprocessed data saved to {output_file}")

    print(f"\n{'='*70}")
    print("ATL03 PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total segments: {len(result):,}")
    print(f"Height range: {result['mean_h'].min():.2f} to {result['mean_h'].max():.2f} m")
    print(f"Mean photon count per segment: {result['photon_count'].mean():.1f}")
    print(f"Beams: {result['beam'].nunique()}")
    print(f"Granules: {result['granule'].nunique()}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ICESat-2 ATL03 data into 2m segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input atl03_data.h5 --output atl03_preprocessed.csv
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input ATL03 HDF5 file")
    parser.add_argument("--output", type=str, default="atl03_preprocessed.csv",
                        help="Output CSV file (default: atl03_preprocessed.csv)")

    args = parser.parse_args()

    try:
        preprocess_atl03(args.input, args.output)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Failed to preprocess ATL03 data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
