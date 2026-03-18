#!/usr/bin/env python3

"""
Auto-label ATL03 segments using coincident Sentinel-2 imagery.

Co-registers Sentinel-2 scenes with ATL03 tracks in Antarctic Polar Stereographic
projection (EPSG:3976), applies color-based segmentation to classify S2 pixels
into thick ice, thin ice, and open water, then overlays labels onto nearest
ATL03 2m segments.

Based on: "Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data" (Iqrah et al., IPDPSW 2025)

Usage:
    python auto_label.py --atl03-input atl03_preprocessed.csv \
                          --sentinel2-input sentinel2_scenes.tar.gz \
                          --output labeled_data.csv
"""

import argparse
import logging
import os
import sys
import tarfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sea ice classes
CLASS_THICK_ICE = 0
CLASS_THIN_ICE = 1
CLASS_OPEN_WATER = 2
CLASS_NAMES = {0: 'thick_ice', 1: 'thin_ice', 2: 'open_water'}

# Antarctic Polar Stereographic
CRS_ANTARCTIC = "EPSG:3976"
CRS_WGS84 = "EPSG:4326"


def classify_sentinel2_scene(scene_dir):
    """
    Classify a Sentinel-2 scene into sea ice types using band ratios.

    Uses a color-based segmentation approach:
    - Open water: low NIR reflectance (B08 < threshold)
    - Thin ice: moderate reflectance, higher blue/red ratio
    - Thick ice: high reflectance across all bands

    Args:
        scene_dir: Path to directory with band TIF files

    Returns:
        Tuple of (classification_array, transform, crs) or None
    """
    import numpy as np
    import rasterio

    scene_dir = Path(scene_dir)
    band_files = {
        'B02': scene_dir / 'B02.tif',  # Blue
        'B03': scene_dir / 'B03.tif',  # Green
        'B04': scene_dir / 'B04.tif',  # Red
        'B08': scene_dir / 'B08.tif',  # NIR
    }

    for name, path in band_files.items():
        if not path.exists():
            logger.warning(f"Missing band file: {path}")
            return None

    # Read bands
    with rasterio.open(band_files['B02']) as src:
        blue = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        shape = blue.shape

    with rasterio.open(band_files['B03']) as src:
        green = src.read(1).astype(np.float32)
    with rasterio.open(band_files['B04']) as src:
        red = src.read(1).astype(np.float32)
    with rasterio.open(band_files['B08']) as src:
        nir = src.read(1).astype(np.float32)

    # Normalize to [0, 1] (S2 L2A values are typically 0-10000)
    scale = 10000.0
    blue /= scale
    green /= scale
    red /= scale
    nir /= scale

    # Classification thresholds (empirically derived for polar sea ice)
    classification = np.full(shape, CLASS_THICK_ICE, dtype=np.int8)

    # Open water: low NIR reflectance
    water_mask = nir < 0.1
    classification[water_mask] = CLASS_OPEN_WATER

    # Thin ice: moderate reflectance, blue-to-red ratio > 1
    blue_red_ratio = np.where(red > 0.01, blue / red, 1.0)
    thin_ice_mask = (~water_mask) & (nir < 0.4) & (blue_red_ratio > 1.05)
    classification[thin_ice_mask] = CLASS_THIN_ICE

    # Everything else is thick ice (high reflectance)

    logger.info(f"Scene classification: "
                f"thick_ice={np.sum(classification == CLASS_THICK_ICE)}, "
                f"thin_ice={np.sum(classification == CLASS_THIN_ICE)}, "
                f"open_water={np.sum(classification == CLASS_OPEN_WATER)}")

    return classification, transform, crs


def reproject_atl03_to_scene_crs(atl03_df, scene_crs):
    """
    Reproject ATL03 lat/lon to the scene's CRS.

    Args:
        atl03_df: DataFrame with 'lat' and 'lon' columns
        scene_crs: Target CRS

    Returns:
        Arrays of (x, y) in scene CRS
    """
    import pyproj

    transformer = pyproj.Transformer.from_crs(CRS_WGS84, scene_crs, always_xy=True)
    x, y = transformer.transform(atl03_df['lon'].values, atl03_df['lat'].values)
    return x, y


def overlay_labels(atl03_df, classification, transform, scene_crs, max_distance=100):
    """
    Overlay S2 classification labels onto ATL03 segments.

    Args:
        atl03_df: DataFrame with ATL03 segment data
        classification: 2D classification array
        transform: Rasterio affine transform
        scene_crs: CRS of the classification raster
        max_distance: Maximum distance (meters) for label assignment

    Returns:
        Array of labels for each ATL03 segment (-1 for no label)
    """
    import numpy as np

    # Reproject ATL03 points to scene CRS
    x_atl03, y_atl03 = reproject_atl03_to_scene_crs(atl03_df, scene_crs)

    labels = np.full(len(atl03_df), -1, dtype=np.int8)

    for i in range(len(atl03_df)):
        # Convert projected coordinates to raster row/col
        col, row = ~transform * (x_atl03[i], y_atl03[i])
        row, col = int(round(row)), int(round(col))

        if 0 <= row < classification.shape[0] and 0 <= col < classification.shape[1]:
            labels[i] = classification[row, col]

    valid = np.sum(labels >= 0)
    logger.info(f"Labeled {valid}/{len(atl03_df)} segments ({100*valid/len(atl03_df):.1f}%)")

    return labels


def auto_label(atl03_input, sentinel2_input, output_file):
    """
    Full auto-labeling pipeline.

    Args:
        atl03_input: Path to preprocessed ATL03 CSV
        sentinel2_input: Path to Sentinel-2 tar.gz archive
        output_file: Output CSV with labeled data
    """
    import numpy as np
    import pandas as pd

    logger.info(f"Reading ATL03 data from {atl03_input}")
    atl03_df = pd.read_csv(atl03_input)
    logger.info(f"ATL03 segments: {len(atl03_df):,}")

    # Extract Sentinel-2 scenes
    logger.info(f"Extracting Sentinel-2 scenes from {sentinel2_input}")
    extract_dir = Path("sentinel2_extracted")
    with tarfile.open(sentinel2_input, "r:gz") as tar:
        tar.extractall(extract_dir)

    scene_dirs = sorted(extract_dir.iterdir())
    logger.info(f"Found {len(scene_dirs)} scenes")

    # Process each scene and collect labels
    all_labels = np.full(len(atl03_df), -1, dtype=np.int8)

    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue

        logger.info(f"Processing scene: {scene_dir.name}")
        result = classify_sentinel2_scene(scene_dir)
        if result is None:
            continue

        classification, transform, crs = result

        # Overlay labels
        scene_labels = overlay_labels(atl03_df, classification, transform, crs)

        # Merge labels (prefer existing labels, fill in missing)
        unlabeled = all_labels < 0
        all_labels[unlabeled] = scene_labels[unlabeled]

    # Cleanup extracted files
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)

    # Add labels to dataframe
    atl03_df['label'] = all_labels

    # Filter to only labeled segments
    labeled_df = atl03_df[atl03_df['label'] >= 0].copy()
    logger.info(f"Total labeled segments: {len(labeled_df):,} / {len(atl03_df):,}")

    if labeled_df.empty:
        logger.error("No segments could be labeled")
        sys.exit(1)

    labeled_df.to_csv(output_file, index=False)
    logger.info(f"Labeled data saved to {output_file}")

    # Summary
    class_counts = labeled_df['label'].value_counts().sort_index()
    print(f"\n{'='*70}")
    print("AUTO-LABELING COMPLETE")
    print(f"{'='*70}")
    print(f"Total labeled segments: {len(labeled_df):,}")
    for label_val, count in class_counts.items():
        pct = 100 * count / len(labeled_df)
        print(f"  {CLASS_NAMES.get(label_val, f'class_{label_val}')}: {count:,} ({pct:.1f}%)")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label ATL03 segments using Sentinel-2 imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --atl03-input atl03_preprocessed.csv \\
           --sentinel2-input sentinel2_scenes.tar.gz \\
           --output labeled_data.csv
        """
    )

    parser.add_argument("--atl03-input", type=str, required=True,
                        help="Input preprocessed ATL03 CSV file")
    parser.add_argument("--sentinel2-input", type=str, required=True,
                        help="Input Sentinel-2 scenes tar.gz archive")
    parser.add_argument("--output", type=str, default="labeled_data.csv",
                        help="Output labeled CSV file (default: labeled_data.csv)")

    args = parser.parse_args()

    try:
        auto_label(args.atl03_input, args.sentinel2_input, args.output)
        logger.info("Auto-labeling completed successfully")
    except Exception as e:
        logger.error(f"Failed to auto-label data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
