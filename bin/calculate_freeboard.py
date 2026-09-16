#!/usr/bin/env python3

"""
Calculate sea ice freeboard from classified ATL03 segments.

Implements Sec. III.D of Iqrah et al. (IPDPSW 2025):

  * h_f = h_s - h_ref                                                (Eq. 1)
  * the local sea surface is found in 10 km windows (5 km radius) stepped
    5 km along track ("a sliding overlap of 5 km"), per track;
  * within a window, open-water segments are grouped into leads (runs of
    consecutive open-water segments) and each lead's height is the weighted
    mean of Eq. 2,
        h_lead = sum(a_i h_i),  a_i = w_i / sum(w_j),
        w_i = exp(-((h_i - h_min) / sigma_i)^2),
        sigma2_lead = sum(a_i^2 sigma_i^2),
    with h_min the lowest segment height in the lead and sigma_i^2 the error
    variance of the 2 m height estimate. That variance is not defined in the
    paper; here it is the standard error of the segment mean, std_h^2 / N,
    floored at (1 cm)^2;
  * the window's reference height combines its leads by inverse variance
    (Eq. 3),  h_ref = sum(a_i h_lead_i),  a_i = (1/sigma2_lead_i) / sum(...);
  * windows without a lead take the linearly interpolated h_ref of the
    nearest windows that have one. Tracks with no open water at all get no
    freeboard (NaN) rather than an invented sea surface.

Usage:
    python calculate_freeboard.py --input classification_results.csv \\
                                   --output freeboard_results.csv
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WINDOW_RADIUS_M = 5000.0    # 10 km window
WINDOW_STEP_M = 5000.0      # 5 km overlap between successive windows
SEGMENT_M = 2.0
LEAD_MAX_GAP_M = 2.0 * SEGMENT_M   # consecutive open-water segments form one lead
SIGMA_FLOOR_M = 0.01
CLASS_THICK_ICE = 0
CLASS_THIN_ICE = 1
CLASS_OPEN_WATER = 2


def find_leads(x, h, sigma2):
    """
    Group open-water segments into leads and apply Eq. 2 to each.

    Args:
        x: Along-track positions of open-water segments (sorted)
        h: Their heights
        sigma2: Their height error variances

    Returns:
        DataFrame with x_lead, h_lead, sigma2_lead, n_segments
    """
    import numpy as np
    import pandas as pd

    if len(x) == 0:
        return pd.DataFrame(columns=['x_lead', 'h_lead', 'sigma2_lead', 'n_segments'])

    breaks = np.nonzero(np.diff(x) > LEAD_MAX_GAP_M)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(x)]])

    rows = []
    for s, e in zip(starts, ends):
        hi, si = h[s:e], sigma2[s:e]
        h_min = hi.min()
        w = np.exp(-((hi - h_min) ** 2) / si)
        if not np.isfinite(w).all() or w.sum() == 0:
            w = np.ones_like(hi)
        a = w / w.sum()
        rows.append({
            'x_lead': float(x[s:e].mean()),
            'h_lead': float(np.sum(a * hi)),
            'sigma2_lead': float(np.sum(a ** 2 * si)),
            'n_segments': int(e - s),
        })
    return pd.DataFrame(rows)


def local_sea_surface(x, h, cls, sigma2, window_radius=WINDOW_RADIUS_M, step=WINDOW_STEP_M):
    """
    Reference sea surface height for every segment of one track (Eqs. 2-3).

    Args:
        x: Along-track positions (sorted)
        h: Segment heights
        cls: Predicted classes
        sigma2: Height error variances
        window_radius: Window half-width in meters
        step: Window step in meters

    Returns:
        (h_ref per segment, sigma2_ref per segment, n_leads per segment,
         number of windows with a lead, number of windows total)
    """
    import numpy as np

    ow = cls == CLASS_OPEN_WATER
    leads = find_leads(x[ow], h[ow], sigma2[ow])
    logger.info(f"    open-water segments: {int(ow.sum()):,}  leads: {len(leads)}")

    n = len(x)
    if leads.empty:
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=int), 0, 0

    centers = np.arange(x.min(), x.max() + step, step)
    h_ref_w = np.full(len(centers), np.nan)
    s2_ref_w = np.full(len(centers), np.nan)
    n_leads_w = np.zeros(len(centers), dtype=int)

    xl = leads['x_lead'].values
    hl = leads['h_lead'].values
    s2l = np.maximum(leads['sigma2_lead'].values, SIGMA_FLOOR_M ** 2)
    for k, c in enumerate(centers):
        m = np.abs(xl - c) <= window_radius
        if not m.any():
            continue
        inv = 1.0 / s2l[m]
        a = inv / inv.sum()
        h_ref_w[k] = np.sum(a * hl[m])
        s2_ref_w[k] = np.sum(a ** 2 * s2l[m])
        n_leads_w[k] = int(m.sum())

    valid = np.isfinite(h_ref_w)
    n_valid = int(valid.sum())
    if n_valid == 1:
        h_ref = np.full(n, h_ref_w[valid][0])
        s2_ref = np.full(n, s2_ref_w[valid][0])
    else:
        h_ref = np.interp(x, centers[valid], h_ref_w[valid])
        s2_ref = np.interp(x, centers[valid], s2_ref_w[valid])
    # Number of leads in the window nearest each segment
    nearest_w = np.clip(np.round((x - centers[0]) / step).astype(int), 0, len(centers) - 1)
    n_leads = n_leads_w[nearest_w]
    return h_ref, s2_ref, n_leads, n_valid, len(centers)


def calculate_freeboard(input_file, output_file, window_radius=WINDOW_RADIUS_M, step=WINDOW_STEP_M):
    """
    Calculate freeboard for all classified segments, track by track.

    Args:
        input_file: Path to (merged) classification results CSV
        output_file: Path to output freeboard CSV
        window_radius: Sliding window half-width in meters
        step: Window step in meters
    """
    import numpy as np
    import pandas as pd

    logger.info(f"Loading classification results from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total segments: {len(df):,}")

    required = ['along_track_dist', 'mean_h', 'predicted_class', 'std_h', 'photon_count']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)
    for c in ('granule', 'beam'):
        if c not in df.columns:
            df[c] = 'all'

    df = df.sort_values(['granule', 'beam', 'along_track_dist']).reset_index(drop=True)

    results = []
    for (granule, beam), group in df.groupby(['granule', 'beam'], sort=False):
        logger.info(f"Computing freeboard for {granule}/{beam} ({len(group):,} segments)")
        g = group.copy()
        x = g['along_track_dist'].values.astype(float)
        h = g['mean_h'].values.astype(float)
        cls = g['predicted_class'].values
        n_ph = np.maximum(g['photon_count'].values.astype(float), 1.0)
        sigma2 = np.maximum(g['std_h'].values.astype(float) ** 2 / n_ph, SIGMA_FLOOR_M ** 2)

        h_ref, s2_ref, n_leads, n_valid, n_windows = local_sea_surface(
            x, h, cls, sigma2, window_radius, step
        )
        if n_windows == 0:
            logger.warning(f"    {granule}/{beam}: no open water on this track, freeboard undefined")
        else:
            logger.info(f"    windows with leads: {n_valid}/{n_windows}")

        g['sea_surface_h'] = h_ref
        g['sea_surface_sigma'] = np.sqrt(s2_ref)
        g['n_leads_window'] = n_leads
        g['freeboard'] = g['mean_h'] - h_ref
        results.append(g)

    result = pd.concat(results, ignore_index=True)
    result.to_csv(output_file, index=False)
    logger.info(f"Freeboard results saved to {output_file}")

    ice = result['predicted_class'].isin([CLASS_THICK_ICE, CLASS_THIN_ICE]) & result['freeboard'].notna()
    ice_fb = result.loc[ice, 'freeboard']
    thick_fb = result.loc[ice & (result['predicted_class'] == CLASS_THICK_ICE), 'freeboard']
    thin_fb = result.loc[ice & (result['predicted_class'] == CLASS_THIN_ICE), 'freeboard']
    ow_fb = result.loc[(result['predicted_class'] == CLASS_OPEN_WATER) & result['freeboard'].notna(), 'freeboard']

    print(f"\n{'='*70}")
    print("FREEBOARD CALCULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total segments: {len(result):,}   with freeboard: {int(result['freeboard'].notna().sum()):,}")
    print(f"Window: {2 * window_radius / 1000:.0f} km, step {step / 1000:.0f} km")
    if len(ice_fb):
        print(f"\nAll ice freeboard:  mean {ice_fb.mean():.3f} m  median {ice_fb.median():.3f} m  std {ice_fb.std():.3f} m")
    if len(thick_fb):
        print(f"Thick ice:          mean {thick_fb.mean():.3f} m  median {thick_fb.median():.3f} m  n={len(thick_fb):,}")
    if len(thin_fb):
        print(f"Thin ice:           mean {thin_fb.mean():.3f} m  median {thin_fb.median():.3f} m  n={len(thin_fb):,}")
    if len(ow_fb):
        print(f"Open water (check): mean {ow_fb.mean():+.3f} m  std {ow_fb.std():.3f} m")
    print(f"\nOutput: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate sea ice freeboard from classified ATL03 segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input classification_results.csv --output freeboard_results.csv
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input classification results CSV file")
    parser.add_argument("--output", type=str, default="freeboard_results.csv",
                        help="Output freeboard CSV (default: freeboard_results.csv)")
    parser.add_argument("--window-radius", type=float, default=WINDOW_RADIUS_M,
                        help=f"Window half-width in meters (default: {WINDOW_RADIUS_M:.0f})")
    parser.add_argument("--window-step", type=float, default=WINDOW_STEP_M,
                        help=f"Window step in meters (default: {WINDOW_STEP_M:.0f})")

    args = parser.parse_args()

    try:
        calculate_freeboard(args.input, args.output, window_radius=args.window_radius, step=args.window_step)
        logger.info("Freeboard calculation completed successfully")
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate freeboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
