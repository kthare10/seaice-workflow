#!/usr/bin/env python3

"""
Auto-label ATL03 segments using coincident Sentinel-2 imagery.

Follows Sec. III.A.3 of Iqrah et al. (IPDPSW 2025) and the color-based
segmentation of Iqrah et al. 2023 (arXiv:2303.12719, Sec. 3.2):

  * Sentinel-2 true-color (8-bit RGB) imagery is converted to HSV and each
    pixel is labeled by the published HSV ranges
        thick ice  (0, 0, 205) - (185, 255, 255)
        thin ice   (0, 0,  31) - (185, 255, 204)
        open water (0, 0,   0) - (185, 255,  30)
    Because the H and S ranges span their full domains, these reduce to
    thresholds on V = max(R, G, B).
  * Cloud and cloud-shadow pixels are masked using the Sentinel-2 L2A Scene
    Classification Layer (SCL). The paper filters thin clouds and shadows
    with an unspecified OpenCV pipeline; the SCL mask is an explicit
    substitute for that step and is documented as such.
  * Both datasets are placed in Antarctic Polar Stereographic (EPSG:3976):
    the label raster is reprojected and the ATL03 segments are transformed.
  * The per-pair Sentinel-2 image shifts of the paper's Table I are applied
    when a scene's ICESat-2 acquisition date matches a Table I entry.
  * Labels are only transferred from a scene to the ATL03 granule it was
    selected for (download_sentinel2.py records the pairing in meta.json);
    when several scenes cover a granule the closest in time wins.

There is no height-based fallback: if no segment receives a label the job
writes an empty output and exits non-zero.

Usage:
    python auto_label.py --atl03-input atl03_preprocessed.csv \\
                          --sentinel2-input sentinel2_scenes.tar.gz \\
                          --output labeled_data.csv
"""

import argparse
import json
import logging
import shutil
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
CLASS_NODATA = -1
CLASS_NAMES = {0: 'thick_ice', 1: 'thin_ice', 2: 'open_water'}

# HSV V-channel bounds from Iqrah et al. 2023, Sec. 3.2 (8-bit, inclusive)
V_OPEN_WATER_MAX = 30
V_THIN_ICE_MIN = 31
V_THIN_ICE_MAX = 204
V_THICK_ICE_MIN = 205

# Sentinel-2 L2A SCL classes treated as unusable: 0 no data, 1 saturated /
# defective, 3 cloud shadow, 8 cloud medium probability, 9 cloud high
# probability, 10 thin cirrus. Class 2 (dark area) is kept because open
# water and leads fall in it.
SCL_MASK_CLASSES = (0, 1, 3, 8, 9, 10)

# Antarctic Polar Stereographic (paper: "EPSG 3976 for both datasets")
CRS_ANTARCTIC = "EPSG:3976"
CRS_WGS84 = "EPSG:4326"
TARGET_RES_M = 10.0

# Paper Table I: Sentinel-2 image shift applied to align with the ICESat-2
# track, keyed by the ICESat-2 acquisition date (UTC). Distance in meters and
# compass direction; the direction is interpreted in the EPSG:3976 grid axes
# (+x east, +y north). Dates absent from the table get no shift.
TABLE_I_SHIFTS = {
    '2019-11-03': (550.0, 'NW'),
    '2019-11-04': (0.0, None),
    '2019-11-13': (200.0, 'W'),
    '2019-11-16': (0.0, None),
    '2019-11-17': (530.0, 'NW'),
    '2019-11-20': (400.0, 'NW'),
    '2019-11-23': (150.0, 'E'),
    '2019-11-26': (350.0, 'SW'),
}
_DIRECTION_VECTORS = {
    'N': (0.0, 1.0), 'S': (0.0, -1.0), 'E': (1.0, 0.0), 'W': (-1.0, 0.0),
    'NE': (0.7071, 0.7071), 'NW': (-0.7071, 0.7071),
    'SE': (0.7071, -0.7071), 'SW': (-0.7071, -0.7071),
}


def shift_for_scene(meta):
    """
    Look up the Table I shift for a scene from its ICESat-2 acquisition date.

    Args:
        meta: Scene metadata dict (needs 'is2_start_utc')

    Returns:
        (dx_m, dy_m) shift to add to the label raster's origin, in EPSG:3976
    """
    is2_time = meta.get('is2_start_utc')
    if not is2_time:
        return 0.0, 0.0
    date = is2_time[:10]
    dist, direction = TABLE_I_SHIFTS.get(date, (0.0, None))
    if not dist or direction is None:
        return 0.0, 0.0
    ux, uy = _DIRECTION_VECTORS[direction]
    logger.info(f"  Table I shift for IS2 {date}: {dist:.0f} m {direction}")
    return dist * ux, dist * uy


def classify_tci(scene_dir):
    """
    Label a scene's pixels from the true-color image using the HSV ranges.

    Args:
        scene_dir: Directory holding visual.tif (and optionally SCL.tif)

    Returns:
        (classification int8 array, source transform, source CRS) or None
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    tci_path = Path(scene_dir) / "visual.tif"
    if not tci_path.exists():
        logger.warning(f"  Missing visual.tif in {scene_dir}")
        return None

    with rasterio.open(tci_path) as src:
        rgb = src.read()[:3]
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)

    if rgb.dtype != np.uint8:
        # The TCI asset is 8-bit; anything else is not the product [5] used
        logger.warning(f"  visual.tif is {rgb.dtype}, expected uint8; scaling to 8-bit")
        rgb = np.clip(rgb.astype(np.float32) / max(float(rgb.max()), 1.0) * 255.0, 0, 255).astype(np.uint8)

    # HSV value channel of an RGB image is max(R, G, B); the published H and
    # S ranges cover their full domains, so V alone decides the class
    v = rgb.max(axis=0)
    nodata = (rgb == 0).all(axis=0)

    classification = np.full(shape, CLASS_NODATA, dtype=np.int8)
    classification[v <= V_OPEN_WATER_MAX] = CLASS_OPEN_WATER
    classification[(v >= V_THIN_ICE_MIN) & (v <= V_THIN_ICE_MAX)] = CLASS_THIN_ICE
    classification[v >= V_THICK_ICE_MIN] = CLASS_THICK_ICE
    classification[nodata] = CLASS_NODATA

    scl_path = Path(scene_dir) / "SCL.tif"
    if scl_path.exists():
        with rasterio.open(scl_path) as src:
            # SCL is 20 m; resample to the 10 m TCI grid (same tile footprint)
            scl = src.read(1, out_shape=shape, resampling=Resampling.nearest)
        cloud_mask = np.isin(scl, SCL_MASK_CLASSES)
        classification[cloud_mask] = CLASS_NODATA
        logger.info(f"  SCL mask: {100.0 * cloud_mask.mean():.1f}% of pixels cloud/shadow/nodata")
    else:
        logger.warning(f"  No SCL.tif in {scene_dir}; clouds and shadows are not masked")

    valid = classification >= 0
    logger.info(
        f"  Labels: thick_ice={np.sum(classification == CLASS_THICK_ICE):,}  "
        f"thin_ice={np.sum(classification == CLASS_THIN_ICE):,}  "
        f"open_water={np.sum(classification == CLASS_OPEN_WATER):,}  "
        f"masked={np.sum(~valid):,}"
    )
    return classification, transform, crs


def reproject_labels_to_3976(classification, transform, crs, shift_xy=(0.0, 0.0)):
    """
    Reproject a label raster to EPSG:3976 at 10 m and apply a Table I shift.

    Args:
        classification: int8 label array in the scene CRS
        transform: Scene affine transform
        crs: Scene CRS
        shift_xy: (dx, dy) in meters added to the raster origin after reprojection

    Returns:
        (labels_3976 array, transform_3976)
    """
    import numpy as np
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.enums import Resampling

    height, width = classification.shape
    left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        crs, CRS_ANTARCTIC, width, height, left, bottom, right, top, resolution=TARGET_RES_M
    )
    dst = np.full((dst_h, dst_w), CLASS_NODATA, dtype=np.int8)
    reproject(
        source=classification, destination=dst,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=CRS_ANTARCTIC,
        src_nodata=CLASS_NODATA, dst_nodata=CLASS_NODATA,
        resampling=Resampling.nearest,
    )
    dx, dy = shift_xy
    if dx or dy:
        dst_transform = dst_transform * Affine.translation(0, 0)
        dst_transform = Affine(dst_transform.a, dst_transform.b, dst_transform.c + dx,
                               dst_transform.d, dst_transform.e, dst_transform.f + dy)
    return dst, dst_transform


def overlay_labels(x, y, labels, transform):
    """
    Read the label under each ATL03 segment position (nearest pixel).

    Args:
        x, y: Segment coordinates in EPSG:3976
        labels: Label raster in EPSG:3976
        transform: Raster affine transform

    Returns:
        int8 array of labels, CLASS_NODATA where off-raster or masked
    """
    import numpy as np

    cols, rows = ~transform * (x, y)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)
    inside = (rows >= 0) & (rows < labels.shape[0]) & (cols >= 0) & (cols < labels.shape[1])
    out = np.full(len(x), CLASS_NODATA, dtype=np.int8)
    out[inside] = labels[rows[inside], cols[inside]]
    return out


def auto_label(atl03_input, sentinel2_input, output_file):
    """
    Full auto-labeling pipeline.

    Args:
        atl03_input: Path to preprocessed ATL03 CSV
        sentinel2_input: Path to Sentinel-2 tar.gz from download_sentinel2.py
        output_file: Output CSV with labeled data
    """
    import numpy as np
    import pandas as pd
    import pyproj

    logger.info(f"Reading ATL03 data from {atl03_input}")
    atl03_df = pd.read_csv(atl03_input)
    logger.info(f"ATL03 segments: {len(atl03_df):,}")
    if 'granule' not in atl03_df.columns:
        atl03_df['granule'] = None

    # Both datasets in EPSG:3976
    transformer = pyproj.Transformer.from_crs(CRS_WGS84, CRS_ANTARCTIC, always_xy=True)
    x_all, y_all = transformer.transform(atl03_df['lon'].values, atl03_df['lat'].values)

    logger.info(f"Extracting Sentinel-2 scenes from {sentinel2_input}")
    extract_dir = Path("sentinel2_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with tarfile.open(sentinel2_input, "r:gz") as tar:
        tar.extractall(extract_dir)

    scenes = []
    for scene_dir in sorted(p for p in extract_dir.iterdir() if p.is_dir()):
        meta_path = scene_dir / "meta.json"
        meta = json.load(open(meta_path)) if meta_path.exists() else {}
        scenes.append((meta.get('time_diff_min', float('inf')), scene_dir, meta))
    # Closest-in-time scenes label first; later scenes only fill gaps
    scenes.sort(key=lambda s: s[0])
    logger.info(f"Found {len(scenes)} scene(s)")

    all_labels = np.full(len(atl03_df), CLASS_NODATA, dtype=np.int8)
    label_source = np.full(len(atl03_df), '', dtype=object)

    for time_diff, scene_dir, meta in scenes:
        logger.info(f"Processing scene: {scene_dir.name} (dt={time_diff} min)")
        result = classify_tci(scene_dir)
        if result is None:
            continue
        classification, transform, crs = result
        labels_3976, transform_3976 = reproject_labels_to_3976(
            classification, transform, crs, shift_for_scene(meta)
        )

        # Restrict to the granule this scene was matched to
        granule = meta.get('granule')
        if granule:
            target = (atl03_df['granule'] == granule).values
        else:
            target = np.ones(len(atl03_df), dtype=bool)
        idx = np.nonzero(target & (all_labels < 0))[0]
        if idx.size == 0:
            logger.info("  No unlabeled segments for this scene's granule")
            continue

        scene_labels = overlay_labels(x_all[idx], y_all[idx], labels_3976, transform_3976)
        got = scene_labels >= 0
        all_labels[idx[got]] = scene_labels[got]
        label_source[idx[got]] = meta.get('scene_id', scene_dir.name)
        logger.info(f"  Labeled {got.sum():,} / {idx.size:,} candidate segments")

    shutil.rmtree(extract_dir, ignore_errors=True)

    atl03_df['label'] = all_labels
    atl03_df['label_scene'] = label_source
    labeled_df = atl03_df[atl03_df['label'] >= 0].copy()
    logger.info(f"Total labeled segments: {len(labeled_df):,} / {len(atl03_df):,}")

    if labeled_df.empty:
        # Required source for training: write the declared output, then fail
        logger.error("No ATL03 segment received a Sentinel-2 label")
        atl03_df.head(0).to_csv(output_file, index=False)
        sys.exit(1)

    labeled_df.to_csv(output_file, index=False)
    logger.info(f"Labeled data saved to {output_file}")

    class_counts = labeled_df['label'].value_counts().sort_index()
    print(f"\n{'='*70}")
    print("AUTO-LABELING COMPLETE")
    print(f"{'='*70}")
    print(f"Total labeled segments: {len(labeled_df):,} of {len(atl03_df):,}")
    for label_val, count in class_counts.items():
        pct = 100 * count / len(labeled_df)
        print(f"  {CLASS_NAMES.get(label_val, f'class_{label_val}')}: {count:,} ({pct:.1f}%)")
    print(f"Scenes used: {labeled_df['label_scene'].nunique()}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label ATL03 segments using coincident Sentinel-2 imagery",
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
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Failed to auto-label data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
