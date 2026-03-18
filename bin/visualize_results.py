#!/usr/bin/env python3

"""
Visualize sea ice classification and freeboard results.

Generates classification maps, freeboard profiles along track, and summary
statistics for the sea ice analysis pipeline.

Usage:
    python visualize_results.py --classification-input classification_results.csv \
                                 --freeboard-input freeboard_results.csv \
                                 --classification-map-output classification_map.png \
                                 --freeboard-profile-output freeboard_profile.png \
                                 --summary-output summary_statistics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'Thick Ice', 1: 'Thin Ice', 2: 'Open Water'}
CLASS_COLORS = {0: '#1f77b4', 1: '#87ceeb', 2: '#2ca02c'}


def plot_classification_map(df, output_file):
    """
    Plot geographic classification map of sea ice types.

    Args:
        df: DataFrame with lat, lon, predicted_class columns
        output_file: Output PNG file path
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    for cls_val in sorted(df['predicted_class'].unique()):
        mask = df['predicted_class'] == cls_val
        ax.scatter(
            df.loc[mask, 'lon'],
            df.loc[mask, 'lat'],
            c=CLASS_COLORS.get(cls_val, 'gray'),
            label=CLASS_NAMES.get(cls_val, f'Class {cls_val}'),
            s=1,
            alpha=0.6,
        )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Sea Ice Classification from ICESat-2 ATL03')
    ax.legend(markerscale=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Classification map saved to {output_file}")


def plot_freeboard_profile(df, output_file):
    """
    Plot freeboard profile along track.

    Args:
        df: DataFrame with along_track_dist, freeboard, predicted_class, sea_surface_h
        output_file: Output PNG file path
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top panel: elevation and sea surface
    ax1 = axes[0]
    for cls_val in sorted(df['predicted_class'].unique()):
        mask = df['predicted_class'] == cls_val
        ax1.scatter(
            df.loc[mask, 'along_track_dist'] / 1000,
            df.loc[mask, 'mean_h'],
            c=CLASS_COLORS.get(cls_val, 'gray'),
            label=CLASS_NAMES.get(cls_val, f'Class {cls_val}'),
            s=0.5,
            alpha=0.5,
        )

    if 'sea_surface_h' in df.columns:
        ax1.plot(
            df['along_track_dist'] / 1000,
            df['sea_surface_h'],
            'r-',
            linewidth=1.5,
            label='Sea Surface',
            alpha=0.8,
        )

    ax1.set_ylabel('Elevation (m)')
    ax1.set_title('Along-Track Elevation Profile with Sea Ice Classification')
    ax1.legend(markerscale=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: freeboard
    ax2 = axes[1]
    ice_mask = df['predicted_class'].isin([0, 1])
    if ice_mask.any() and 'freeboard' in df.columns:
        for cls_val in [0, 1]:
            mask = df['predicted_class'] == cls_val
            ax2.scatter(
                df.loc[mask, 'along_track_dist'] / 1000,
                df.loc[mask, 'freeboard'],
                c=CLASS_COLORS.get(cls_val, 'gray'),
                label=CLASS_NAMES.get(cls_val, f'Class {cls_val}'),
                s=0.5,
                alpha=0.5,
            )

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Sea Level')
    ax2.set_xlabel('Along-Track Distance (km)')
    ax2.set_ylabel('Freeboard (m)')
    ax2.set_title('Sea Ice Freeboard Profile')
    ax2.legend(markerscale=10, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Freeboard profile saved to {output_file}")


def compute_summary_statistics(cls_df, fb_df):
    """
    Compute summary statistics for the analysis.

    Args:
        cls_df: Classification results DataFrame
        fb_df: Freeboard results DataFrame

    Returns:
        Dictionary of summary statistics
    """
    stats = {
        'total_segments': int(len(cls_df)),
        'classification': {},
        'freeboard': {},
        'geographic_extent': {
            'min_lat': float(cls_df['lat'].min()),
            'max_lat': float(cls_df['lat'].max()),
            'min_lon': float(cls_df['lon'].min()),
            'max_lon': float(cls_df['lon'].max()),
        },
        'track_length_km': float(
            (cls_df['along_track_dist'].max() - cls_df['along_track_dist'].min()) / 1000
        ),
    }

    # Classification statistics
    for cls_val in sorted(cls_df['predicted_class'].unique()):
        mask = cls_df['predicted_class'] == cls_val
        count = int(mask.sum())
        stats['classification'][CLASS_NAMES.get(cls_val, f'class_{cls_val}')] = {
            'count': count,
            'percentage': float(100 * count / len(cls_df)),
        }

    if 'prediction_prob' in cls_df.columns:
        stats['classification']['mean_confidence'] = float(cls_df['prediction_prob'].mean())

    # Freeboard statistics
    if 'freeboard' in fb_df.columns:
        ice_mask = fb_df['predicted_class'].isin([0, 1])
        ice_fb = fb_df.loc[ice_mask, 'freeboard']

        if len(ice_fb) > 0:
            stats['freeboard']['all_ice'] = {
                'mean': float(ice_fb.mean()),
                'median': float(ice_fb.median()),
                'std': float(ice_fb.std()),
                'min': float(ice_fb.min()),
                'max': float(ice_fb.max()),
            }

        for cls_val, cls_name in [(0, 'thick_ice'), (1, 'thin_ice')]:
            mask = fb_df['predicted_class'] == cls_val
            fb = fb_df.loc[mask, 'freeboard']
            if len(fb) > 0:
                stats['freeboard'][cls_name] = {
                    'mean': float(fb.mean()),
                    'median': float(fb.median()),
                    'std': float(fb.std()),
                    'count': int(len(fb)),
                }

    return stats


def visualize_results(classification_input, freeboard_input,
                      classification_map_output, freeboard_profile_output,
                      summary_output):
    """
    Generate all visualizations and summary statistics.

    Args:
        classification_input: Path to classification results CSV
        freeboard_input: Path to freeboard results CSV
        classification_map_output: Output PNG for classification map
        freeboard_profile_output: Output PNG for freeboard profile
        summary_output: Output JSON for summary statistics
    """
    import numpy as np
    import pandas as pd

    logger.info(f"Loading classification results from {classification_input}")
    cls_df = pd.read_csv(classification_input)
    logger.info(f"Classification segments: {len(cls_df):,}")

    logger.info(f"Loading freeboard results from {freeboard_input}")
    fb_df = pd.read_csv(freeboard_input)
    logger.info(f"Freeboard segments: {len(fb_df):,}")

    # Generate classification map
    logger.info("Generating classification map...")
    plot_classification_map(cls_df, classification_map_output)

    # Generate freeboard profile
    logger.info("Generating freeboard profile...")
    plot_freeboard_profile(fb_df, freeboard_profile_output)

    # Compute and save summary statistics
    logger.info("Computing summary statistics...")
    stats = compute_summary_statistics(cls_df, fb_df)

    with open(summary_output, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Summary statistics saved to {summary_output}")

    # Print summary
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total segments: {stats['total_segments']:,}")
    print(f"Track length: {stats['track_length_km']:.1f} km")
    print(f"\nClassification:")
    for cls_name, cls_stats in stats['classification'].items():
        if isinstance(cls_stats, dict):
            print(f"  {cls_name}: {cls_stats['count']:,} ({cls_stats['percentage']:.1f}%)")
    if 'all_ice' in stats['freeboard']:
        fb = stats['freeboard']['all_ice']
        print(f"\nFreeboard (all ice):")
        print(f"  Mean: {fb['mean']:.3f} m, Median: {fb['median']:.3f} m, Std: {fb['std']:.3f} m")
    print(f"\nOutputs:")
    print(f"  Classification map: {classification_map_output}")
    print(f"  Freeboard profile: {freeboard_profile_output}")
    print(f"  Summary statistics: {summary_output}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize sea ice classification and freeboard results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --classification-input classification_results.csv \\
           --freeboard-input freeboard_results.csv \\
           --classification-map-output classification_map.png \\
           --freeboard-profile-output freeboard_profile.png \\
           --summary-output summary_statistics.json
        """
    )

    parser.add_argument("--classification-input", type=str, required=True,
                        help="Input classification results CSV")
    parser.add_argument("--freeboard-input", type=str, required=True,
                        help="Input freeboard results CSV")
    parser.add_argument("--classification-map-output", type=str,
                        default="classification_map.png",
                        help="Output classification map PNG")
    parser.add_argument("--freeboard-profile-output", type=str,
                        default="freeboard_profile.png",
                        help="Output freeboard profile PNG")
    parser.add_argument("--summary-output", type=str,
                        default="summary_statistics.json",
                        help="Output summary statistics JSON")

    args = parser.parse_args()

    try:
        visualize_results(
            classification_input=args.classification_input,
            freeboard_input=args.freeboard_input,
            classification_map_output=args.classification_map_output,
            freeboard_profile_output=args.freeboard_profile_output,
            summary_output=args.summary_output,
        )
        logger.info("Visualization completed successfully")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
