#!/usr/bin/env python3

"""
Visualize sea ice classification and freeboard results.

Generates publication-quality figures matching the style of Iqrah et al.
(IPDPSW 2025): classification along-track profiles, confusion matrices,
sea surface detection, freeboard profiles, freeboard histograms, and
point density comparisons.

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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'Thick Ice', 1: 'Thin Ice', 2: 'Open Water'}
# Paper colors: thick ice = blue, thin ice = green, open water = orange
CLASS_COLORS = {0: '#1f77b4', 1: '#2ca02c', 2: '#ff7f0e'}


def plot_classification_map(df, output_file):
    """
    Generate a multi-panel classification figure (paper Figs. 4, 6a):

    (a) ATL03 elevation vs along-track longitude colored by ice type
    (b) Geographic classification map (lat vs lon)
    (c) Confusion matrix heatmap (if ground-truth labels are available)

    Args:
        df: DataFrame with lat, lon, mean_h, predicted_class columns
        output_file: Output PNG file path
    """
    import numpy as np

    has_truth = 'label' in df.columns or 'true_class' in df.columns
    truth_col = 'label' if 'label' in df.columns else 'true_class' if 'true_class' in df.columns else None

    ncols = 3 if has_truth else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

    # Custom legend handles
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[k],
               markersize=8, label=CLASS_NAMES[k])
        for k in sorted(CLASS_NAMES.keys())
    ]

    # --- Panel (a): Elevation vs longitude colored by ice type (Fig. 6a) ---
    ax_elev = axes[0]
    for cls_val in sorted(df['predicted_class'].unique()):
        mask = df['predicted_class'] == cls_val
        ax_elev.scatter(
            df.loc[mask, 'lon'],
            df.loc[mask, 'mean_h'],
            c=CLASS_COLORS.get(cls_val, 'gray'),
            s=0.3,
            alpha=0.7,
            rasterized=True,
        )
    ax_elev.set_xlabel('Along Track Long (degree)', fontsize=11)
    ax_elev.set_ylabel('ATL03 Elevation (m)', fontsize=11)
    ax_elev.set_title('ATL03 Classification', fontsize=12, fontweight='bold')
    ax_elev.legend(handles=legend_handles, loc='upper right', fontsize=9,
                   markerscale=1.2, framealpha=0.9)
    ax_elev.grid(True, alpha=0.3)
    ax_elev.set_ylim(-0.5, max(3.5, df['mean_h'].quantile(0.995) + 0.3))
    ax_elev.text(-0.02, 1.05, '(a)', transform=ax_elev.transAxes,
                 fontsize=13, fontweight='bold', va='bottom')

    # --- Panel (b): Geographic map (lat vs lon) ---
    ax_geo = axes[1]
    for cls_val in sorted(df['predicted_class'].unique()):
        mask = df['predicted_class'] == cls_val
        ax_geo.scatter(
            df.loc[mask, 'lon'],
            df.loc[mask, 'lat'],
            c=CLASS_COLORS.get(cls_val, 'gray'),
            s=0.3,
            alpha=0.6,
            rasterized=True,
        )
    ax_geo.set_xlabel('Longitude', fontsize=11)
    ax_geo.set_ylabel('Latitude', fontsize=11)
    ax_geo.set_title('Geographic Classification Map', fontsize=12, fontweight='bold')
    ax_geo.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  markerscale=1.2, framealpha=0.9)
    ax_geo.grid(True, alpha=0.3)
    ax_geo.text(-0.02, 1.05, '(b)', transform=ax_geo.transAxes,
                fontsize=13, fontweight='bold', va='bottom')

    # --- Panel (c): Confusion matrix (Fig. 4) ---
    if has_truth and truth_col:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        ax_cm = axes[2]

        y_true = df[truth_col].values
        y_pred = df['predicted_class'].values
        labels = sorted(set(y_true) | set(y_pred))

        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
        # Normalize to percentages per row
        cm_pct = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_pct = 100.0 * cm_pct / row_sums

        im = ax_cm.imshow(cm_pct, interpolation='nearest', cmap='YlOrRd',
                          vmin=0, vmax=100)
        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_pct[i, j]
                color = 'white' if val > 60 else 'black'
                ax_cm.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=11, fontweight='bold', color=color)

        tick_labels = [CLASS_NAMES.get(l, f'Class {l}') for l in labels]
        pred_labels = [f'Predicted {CLASS_NAMES.get(l, f"Class {l}")}' for l in labels]
        ax_cm.set_xticks(range(len(labels)))
        ax_cm.set_xticklabels(pred_labels, fontsize=9, rotation=20, ha='right')
        ax_cm.set_yticks(range(len(labels)))
        ax_cm.set_yticklabels(tick_labels, fontsize=9)
        ax_cm.set_title('Confusion Matrix (Percentages)', fontsize=12, fontweight='bold')
        ax_cm.text(-0.02, 1.05, '(c)', transform=ax_cm.transAxes,
                   fontsize=13, fontweight='bold', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Classification map saved to {output_file}")


def plot_freeboard_profile(df, output_file):
    """
    Generate a multi-panel freeboard figure (paper Figs. 8a, 10a, 10c, 10d):

    (a) Sea surface detection along track
    (b) Freeboard vs along-track longitude
    (c) Freeboard value distribution histogram
    (d) Point density along track

    Args:
        df: DataFrame with lon, mean_h, freeboard, predicted_class, sea_surface_h columns
        output_file: Output PNG file path
    """
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[k],
               markersize=8, label=CLASS_NAMES[k])
        for k in sorted(CLASS_NAMES.keys())
    ]

    # --- Panel (a): Sea surface detection along track (Fig. 8) ---
    ax_ss = axes[0, 0]

    has_along_track = 'along_track_dist' in df.columns
    x_col = 'along_track_dist' if has_along_track else 'lon'
    x_label = 'Along Track Distance (m)' if has_along_track else 'Along Track Long (degree)'
    x_scale = 1.0

    # Plot elevation colored by class
    for cls_val in sorted(df['predicted_class'].unique()):
        mask = df['predicted_class'] == cls_val
        ax_ss.scatter(
            df.loc[mask, x_col] * x_scale,
            df.loc[mask, 'mean_h'],
            c=CLASS_COLORS.get(cls_val, 'gray'),
            s=0.3, alpha=0.5, rasterized=True,
        )

    # Overlay sea surface line
    if 'sea_surface_h' in df.columns:
        ss = df.sort_values(x_col)
        ax_ss.plot(
            ss[x_col] * x_scale, ss['sea_surface_h'],
            color='red', linewidth=1.2, label='Sea Surface', alpha=0.9,
        )

    ax_ss.set_xlabel(x_label, fontsize=11)
    ax_ss.set_ylabel('Elevation (m)', fontsize=11)
    ax_ss.set_title('Elevation Profile with Sea Surface Detection', fontsize=12,
                     fontweight='bold')

    ss_handles = legend_handles.copy()
    if 'sea_surface_h' in df.columns:
        ss_handles.append(Line2D([0], [0], color='red', linewidth=1.5,
                                 label='Sea Surface'))
    ax_ss.legend(handles=ss_handles, loc='upper right', fontsize=9, framealpha=0.9)
    ax_ss.grid(True, alpha=0.3)
    ax_ss.set_ylim(-0.5, max(3.5, df['mean_h'].quantile(0.995) + 0.3))
    ax_ss.text(-0.02, 1.05, '(a)', transform=ax_ss.transAxes,
               fontsize=13, fontweight='bold', va='bottom')

    # --- Panel (b): Freeboard vs along-track longitude (Fig. 10a) ---
    ax_fb = axes[0, 1]
    if 'freeboard' in df.columns:
        ice_mask = df['predicted_class'].isin([0, 1])
        ice_df = df[ice_mask]
        ax_fb.scatter(
            ice_df['lon'], ice_df['freeboard'],
            c='#2ca02c', s=0.5, alpha=0.6, label='Freeboard 2m ATL03',
            rasterized=True,
        )
        ax_fb.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_fb.set_ylim(-0.5, max(3.5, ice_df['freeboard'].quantile(0.995) + 0.3))
    ax_fb.set_xlabel('Along Track Long (degree)', fontsize=11)
    ax_fb.set_ylabel('Freeboard (m)', fontsize=11)
    ax_fb.set_title('Freeboard from ATL03', fontsize=12, fontweight='bold')
    ax_fb.legend(loc='upper right', fontsize=9, markerscale=6, framealpha=0.9)
    ax_fb.grid(True, alpha=0.3)
    ax_fb.text(-0.02, 1.05, '(b)', transform=ax_fb.transAxes,
               fontsize=13, fontweight='bold', va='bottom')

    # --- Panel (c): Freeboard distribution histogram (Fig. 10c) ---
    ax_hist = axes[1, 0]
    if 'freeboard' in df.columns:
        ice_mask = df['predicted_class'].isin([0, 1])
        fb_vals = df.loc[ice_mask, 'freeboard'].dropna()
        fb_vals = fb_vals[(fb_vals >= -0.5) & (fb_vals <= 3.5)]

        ax_hist.hist(fb_vals, bins=50, color='#1f77b4', edgecolor='black',
                     linewidth=0.5, alpha=0.8, label='Freeboard 2m ATL03')

        # Overlay KDE-like step line
        counts, bin_edges = np.histogram(fb_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax_hist.plot(bin_centers, counts, color='#1f77b4', linewidth=1.5)

        ax_hist.axvline(x=fb_vals.median(), color='red', linestyle='--',
                        linewidth=1.2, alpha=0.8,
                        label=f'Median = {fb_vals.median():.3f} m')
    ax_hist.set_xlabel('Freeboard (m)', fontsize=11)
    ax_hist.set_ylabel('No. of Data Points', fontsize=11)
    ax_hist.set_title('Freeboard Distribution', fontsize=12, fontweight='bold')
    ax_hist.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3, axis='y')
    ax_hist.text(-0.02, 1.05, '(c)', transform=ax_hist.transAxes,
                 fontsize=13, fontweight='bold', va='bottom')

    # --- Panel (d): Point density along track (Fig. 10d) ---
    ax_den = axes[1, 1]
    if len(df) > 0:
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        n_bins = min(100, max(20, len(df) // 500))
        bin_edges = np.linspace(lon_min, lon_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        counts_all = np.histogram(df['lon'], bins=bin_edges)[0]
        ax_den.bar(bin_centers, counts_all, width=(lon_max - lon_min) / n_bins * 0.9,
                   color='#1f77b4', edgecolor='black', linewidth=0.3,
                   alpha=0.8, label='ATL03 2m segments')

    ax_den.set_xlabel('Along Track Long (degree)', fontsize=11)
    ax_den.set_ylabel('No. of Data Points', fontsize=11)
    ax_den.set_title('Point Density Along Track', fontsize=12, fontweight='bold')
    ax_den.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_den.grid(True, alpha=0.3, axis='y')
    ax_den.text(-0.02, 1.05, '(d)', transform=ax_den.transAxes,
                fontsize=13, fontweight='bold', va='bottom')

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
        ) if 'along_track_dist' in cls_df.columns else 0.0,
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

    # Per-class accuracy (if ground-truth available)
    truth_col = 'label' if 'label' in cls_df.columns else 'true_class' if 'true_class' in cls_df.columns else None
    if truth_col:
        overall_acc = float((cls_df[truth_col] == cls_df['predicted_class']).mean())
        stats['classification']['accuracy'] = overall_acc
        for cls_val in sorted(cls_df['predicted_class'].unique()):
            mask = cls_df[truth_col] == cls_val
            if mask.sum() > 0:
                cls_acc = float((cls_df.loc[mask, 'predicted_class'] == cls_val).mean())
                stats['classification'][CLASS_NAMES.get(cls_val, f'class_{cls_val}')]['accuracy'] = cls_acc

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
    if stats['track_length_km'] > 0:
        print(f"Track length: {stats['track_length_km']:.1f} km")
    print(f"\nClassification:")
    for cls_name, cls_stats in stats['classification'].items():
        if isinstance(cls_stats, dict):
            line = f"  {cls_name}: {cls_stats['count']:,} ({cls_stats['percentage']:.1f}%)"
            if 'accuracy' in cls_stats:
                line += f"  [accuracy: {cls_stats['accuracy']:.2%}]"
            print(line)
    if 'accuracy' in stats['classification']:
        print(f"  Overall accuracy: {stats['classification']['accuracy']:.2%}")
    if 'all_ice' in stats['freeboard']:
        fb = stats['freeboard']['all_ice']
        print(f"\nFreeboard (all ice):")
        print(f"  Mean: {fb['mean']:.3f} m, Median: {fb['median']:.3f} m, Std: {fb['std']:.3f} m")
    for ice_type in ['thick_ice', 'thin_ice']:
        if ice_type in stats['freeboard']:
            fb = stats['freeboard'][ice_type]
            print(f"  {ice_type}: mean={fb['mean']:.3f} m, median={fb['median']:.3f} m, n={fb['count']:,}")
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
