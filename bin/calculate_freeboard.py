#!/usr/bin/env python3

"""
Calculate sea ice freeboard from classified ATL03 segments.

Implements a 10km sliding window approach to detect local sea surface height
from open water segments, then computes freeboard as the elevation difference
between ice segments and the interpolated sea surface.

Based on: "Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data" (Iqrah et al., IPDPSW 2025)

Usage:
    python calculate_freeboard.py --input classification_results.csv \
                                   --output freeboard_results.csv
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

# Constants
WINDOW_RADIUS_M = 5000  # 5km radius = 10km sliding window
CLASS_OPEN_WATER = 2
CLASS_THICK_ICE = 0
CLASS_THIN_ICE = 1


def compute_local_sea_surface(df, window_radius=WINDOW_RADIUS_M):
    """
    Compute local sea surface height using a sliding window approach.

    For each position along the track, finds open water segments within
    the window radius and computes a weighted mean sea surface height.

    Args:
        df: DataFrame with 'along_track_dist', 'mean_h', 'predicted_class'
        window_radius: Half-width of sliding window in meters

    Returns:
        Array of interpolated sea surface heights for each segment
    """
    import numpy as np
    from scipy.interpolate import interp1d

    dist = df['along_track_dist'].values
    heights = df['mean_h'].values
    classes = df['predicted_class'].values

    # Find open water segments
    ow_mask = classes == CLASS_OPEN_WATER
    ow_dist = dist[ow_mask]
    ow_heights = heights[ow_mask]

    logger.info(f"Open water segments: {np.sum(ow_mask):,} / {len(df):,}")

    if len(ow_dist) == 0:
        logger.warning("No open water segments found. Using minimum height as sea surface estimate.")
        return np.full(len(df), np.percentile(heights, 5))

    # Compute sea surface height at each open water location
    # Using NASA formula: weighted mean with uncertainty-based weights
    sea_surface = np.full(len(df), np.nan)

    for i in range(len(df)):
        d = dist[i]
        # Find open water segments within window
        window_mask = np.abs(ow_dist - d) <= window_radius
        window_heights = ow_heights[window_mask]
        window_dists = ow_dist[window_mask]

        if len(window_heights) == 0:
            continue

        # Distance-based weights (closer segments weighted more)
        distances = np.abs(window_dists - d) + 1.0  # Avoid division by zero
        weights = 1.0 / distances

        # Weighted mean sea surface height
        sea_surface[i] = np.average(window_heights, weights=weights)

    # Interpolate gaps where no open water was found
    valid = ~np.isnan(sea_surface)
    if np.sum(valid) < 2:
        logger.warning("Insufficient open water for interpolation, using percentile estimate")
        return np.full(len(df), np.percentile(heights, 5))

    # Linear interpolation
    interp_func = interp1d(
        dist[valid], sea_surface[valid],
        kind='linear',
        bounds_error=False,
        fill_value=(sea_surface[valid][0], sea_surface[valid][-1])
    )
    sea_surface_interp = interp_func(dist)

    n_interpolated = np.sum(~valid)
    logger.info(f"Interpolated sea surface for {n_interpolated:,} segments")

    return sea_surface_interp


def calculate_freeboard(input_file, output_file, window_radius=WINDOW_RADIUS_M):
    """
    Calculate freeboard for all classified segments.

    Freeboard = segment_elevation - local_sea_surface_height

    Args:
        input_file: Path to classification results CSV
        output_file: Path to output freeboard CSV
        window_radius: Sliding window half-width in meters
    """
    import numpy as np
    import pandas as pd

    logger.info(f"Loading classification results from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total segments: {len(df):,}")

    # Verify required columns
    required = ['along_track_dist', 'mean_h', 'predicted_class']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)

    # Sort by along-track distance for each beam/granule
    df = df.sort_values(['granule', 'beam', 'along_track_dist']).reset_index(drop=True)

    # Compute freeboard per beam/granule track
    all_results = []
    groups = df.groupby(['granule', 'beam'])

    for (granule, beam), group in groups:
        logger.info(f"Computing freeboard for {granule}/{beam} ({len(group):,} segments)")

        group = group.copy()

        # Compute local sea surface
        sea_surface = compute_local_sea_surface(group, window_radius=window_radius)
        group['sea_surface_h'] = sea_surface

        # Compute freeboard
        group['freeboard'] = group['mean_h'] - sea_surface

        # Set open water freeboard to 0 (by definition)
        group.loc[group['predicted_class'] == CLASS_OPEN_WATER, 'freeboard'] = 0.0

        all_results.append(group)

    result = pd.concat(all_results, ignore_index=True)

    # Save results
    result.to_csv(output_file, index=False)
    logger.info(f"Freeboard results saved to {output_file}")

    # Summary statistics
    ice_mask = result['predicted_class'].isin([CLASS_THICK_ICE, CLASS_THIN_ICE])
    ice_freeboard = result.loc[ice_mask, 'freeboard']

    thick_fb = result.loc[result['predicted_class'] == CLASS_THICK_ICE, 'freeboard']
    thin_fb = result.loc[result['predicted_class'] == CLASS_THIN_ICE, 'freeboard']

    print(f"\n{'='*70}")
    print("FREEBOARD CALCULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total segments: {len(result):,}")
    print(f"Window radius: {window_radius/1000:.1f} km")
    print(f"\nAll ice freeboard:")
    print(f"  Mean: {ice_freeboard.mean():.3f} m")
    print(f"  Median: {ice_freeboard.median():.3f} m")
    print(f"  Std: {ice_freeboard.std():.3f} m")
    if len(thick_fb) > 0:
        print(f"\nThick ice freeboard:")
        print(f"  Mean: {thick_fb.mean():.3f} m, Median: {thick_fb.median():.3f} m")
    if len(thin_fb) > 0:
        print(f"\nThin ice freeboard:")
        print(f"  Mean: {thin_fb.mean():.3f} m, Median: {thin_fb.median():.3f} m")
    print(f"\nOutput: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate sea ice freeboard from classified ATL03 segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input classification_results.csv --output freeboard_results.csv

  # Custom window radius
  %(prog)s --input classification_results.csv --output freeboard_results.csv --window-radius 7500
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input classification results CSV file")
    parser.add_argument("--output", type=str, default="freeboard_results.csv",
                        help="Output freeboard CSV (default: freeboard_results.csv)")
    parser.add_argument("--window-radius", type=float, default=WINDOW_RADIUS_M,
                        help=f"Sliding window half-width in meters (default: {WINDOW_RADIUS_M})")

    args = parser.parse_args()

    try:
        calculate_freeboard(args.input, args.output, window_radius=args.window_radius)
        logger.info("Freeboard calculation completed successfully")
    except Exception as e:
        logger.error(f"Failed to calculate freeboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
