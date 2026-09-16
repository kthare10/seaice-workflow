#!/usr/bin/env python3

"""
Preprocess ICESat-2 ATL03 photon data into 2m along-track segments.

Follows Sec. III.A.2 of Iqrah et al. (IPDPSW 2025):

  * strong beams only (selected upstream by download_atl03.py);
  * high-confidence signal photons for the sea-ice surface type
    (signal_conf_ph[:, 2] == 4);
  * photons flagged as anything other than nominal quality (afterpulse,
    impulse-response, transmit-echo-path) are removed;
  * absolute along-track distance = segment_dist_x + dist_ph_along;
  * 2 m along-track resampling with mean / median / std height, photon
    counts, photon rates and ATL03 background counts / rates;
  * geophysical correction  h_cor = h - MSS - ocean tide - FPB, where MSS is
    ATL03 dem_h where dem_flag == 3 (MSS), tide is geophys_corr/tide_ocean,
    and the first-photon-bias correction is looked up in the granule's CAL-19
    tables following the ATL07 ATBD (Appendix G).

The correction formula is the one that reproduces h_cor_mean in the authors'
labeled data (h_cor = height - mss - tide - fpb_corr, to 4e-6 m).

Output columns match bin/prepare_labeled_csv.py so a model trained in either
mode can be applied in the other.

Usage:
    python preprocess_atl03.py --input atl03_data.h5 \\
                                --output atl03_preprocessed.csv
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

BIN_SIZE_M = 2.0            # 2-meter along-track resolution (paper Sec. III.A.2)
SEA_ICE_SURFACE_TYPE = 2    # signal_conf_ph column: 0 land, 1 ocean, 2 sea ice, 3 land ice, 4 inland water
HIGH_CONF = 4               # signal_conf_ph: 0 noise, 1 buffer, 2 low, 3 medium, 4 high
SIGNAL_CONF_MIN = 2         # low/medium/high count as signal for the photon-rate feature
DEM_FLAG_MSS = 3            # geophys_corr/dem_flag: 0 none, 1 Arctic, 2 global, 3 MSS, 4 Antarctic
SPEED_OF_LIGHT = 299792458.0

# Columns written for every 2 m segment. Shared with prepare_labeled_csv.py.
OUTPUT_COLUMNS = [
    'lat', 'lon', 'along_track_dist', 'delta_time',
    'mean_h', 'median_h', 'std_h',
    'photon_count', 'shots', 'pcnt', 'pcnth', 'd_pcnt',
    'bcnt', 'brate', 'd_brate', 'bg_rate',
    'height_mean_uncor', 'height_med_uncor', 'mss', 'tide_ocean', 'fpb_corr',
    'beam', 'granule',
]


def _photon_segment_index(ph_index_beg, segment_ph_cnt, n_photons):
    """
    Map each photon to its 20 m geolocation segment.

    Args:
        ph_index_beg: 1-based index of the first photon in each segment
        segment_ph_cnt: Photon count per segment
        n_photons: Total photons in the beam

    Returns:
        Integer array (n_photons,) of segment indices, -1 where unassigned
    """
    import numpy as np

    seg_of_ph = np.full(n_photons, -1, dtype=np.int64)
    valid = segment_ph_cnt > 0
    starts = ph_index_beg[valid].astype(np.int64) - 1
    counts = segment_ph_cnt[valid].astype(np.int64)
    seg_ids = np.nonzero(valid)[0]
    for s, c, i in zip(starts, counts, seg_ids):
        end = min(s + c, n_photons)
        if s < n_photons:
            seg_of_ph[s:end] = i
    return seg_of_ph


def _load_fpb_tables(granule_grp, beam):
    """
    Load the CAL-19 first-photon-bias tables and CAL-42 dead time for a beam.

    Args:
        granule_grp: Merged-file granule group (holds a copy of ancillary_data)
        beam: Beam name

    Returns:
        dict with width (ns), strength (events/shot), dead_time (ns),
        ffb_corr (ps) arrays and avg_dead_time_ns, or None if unavailable
    """
    import numpy as np

    fpb_path = f"ancillary_data/calibrations/first_photon_bias/{beam}"
    dt_path = f"ancillary_data/calibrations/dead_time/{beam}"
    if fpb_path not in granule_grp or dt_path not in granule_grp:
        logger.warning(f"  {beam}: CAL-19/CAL-42 tables not found, FPB correction skipped")
        return None

    fpb = granule_grp[fpb_path]
    for key in ('width', 'strength', 'dead_time', 'ffb_corr'):
        if key not in fpb:
            logger.warning(f"  {beam}: CAL-19 table missing '{key}', FPB correction skipped")
            return None

    # CAL-42 dead time is per PCE channel in seconds; channels 1-16 are the
    # strong-beam detectors (ATL03 data dictionary). Use their average, as
    # the ATL07 ATBD does ("average dead time for the active channels").
    dead_time_s = np.asarray(granule_grp[dt_path]['dead_time'][()]).ravel()
    strong_channels = dead_time_s[:16] if dead_time_s.size >= 16 else dead_time_s
    strong_channels = strong_channels[np.isfinite(strong_channels)]
    if strong_channels.size == 0:
        logger.warning(f"  {beam}: no finite CAL-42 dead times, FPB correction skipped")
        return None

    tables = {
        'width': np.asarray(fpb['width'][()]),          # ns, (width, deadtime)
        'strength': np.asarray(fpb['strength'][()]),    # events/shot, (strength, deadtime)
        'dead_time': np.asarray(fpb['dead_time'][()]).ravel(),  # ns
        'ffb_corr': np.asarray(fpb['ffb_corr'][()]),    # ps, (width, strength, deadtime)
        'avg_dead_time_ns': float(np.mean(strong_channels)) * 1e9,
    }
    logger.info(
        f"  {beam}: CAL-19 ffb_corr table {tables['ffb_corr'].shape}, "
        f"avg dead time {tables['avg_dead_time_ns']:.2f} ns"
    )
    return tables


def _fpb_correction_m(tables, width_ns, strength):
    """
    Look up the first-photon-bias correction for a segment (ATL07 ATBD App. G).

    The nearest CAL-19 dead-time slice to the beam's average dead time is
    used; within that slice the correction is interpolated bilinearly in
    (apparent width, strength), clipped to the table range.

    Args:
        tables: Output of _load_fpb_tables
        width_ns: Apparent width array (10%-90% cumulative interval, ns)
        strength: Photon rate array (events/shot)

    Returns:
        Correction array in meters (positive values are subtracted from height)
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    dt_axis = tables['dead_time']
    k = int(np.argmin(np.abs(dt_axis - tables['avg_dead_time_ns'])))

    width_axis = tables['width']
    strength_axis = tables['strength']
    # Axes are stored per dead-time slice, (n_width, n_deadtime) etc.
    if width_axis.ndim == 2:
        width_axis = width_axis[:, k]
    if strength_axis.ndim == 2:
        strength_axis = strength_axis[:, k]
    corr_ps = tables['ffb_corr'][:, :, k]

    # Axes must be strictly increasing for the interpolator
    wi = np.argsort(width_axis)
    si = np.argsort(strength_axis)
    interp = RegularGridInterpolator(
        (width_axis[wi], strength_axis[si]), corr_ps[np.ix_(wi, si)],
        method='linear', bounds_error=False, fill_value=None,
    )
    w = np.clip(width_ns, width_axis.min(), width_axis.max())
    s = np.clip(strength, strength_axis.min(), strength_axis.max())
    corr_ps_seg = interp(np.column_stack([w, s]))

    # Two-way travel time to one-way range
    return corr_ps_seg * 1e-12 * SPEED_OF_LIGHT / 2.0


def process_beam(granule_grp, granule_key, beam):
    """
    Turn one strong beam of one granule into corrected 2 m segments.

    Args:
        granule_grp: Merged-file granule group
        granule_key: Granule group name
        beam: Beam name

    Returns:
        DataFrame of segments with OUTPUT_COLUMNS, or None if the beam is unusable
    """
    import numpy as np
    import pandas as pd

    b = granule_grp[beam]
    beam_type = b.attrs.get('atlas_beam_type', b'strong')
    if isinstance(beam_type, bytes):
        beam_type = beam_type.decode()
    if str(beam_type).lower() != 'strong':
        logger.warning(f"  {beam}: atlas_beam_type={beam_type}, skipping (strong beams only)")
        return None

    required = ['h_ph', 'lat_ph', 'lon_ph', 'signal_conf_ph', 'delta_time', 'dist_ph_along']
    for key in required + ['geolocation', 'geophys_corr', 'bckgrd_atlas']:
        if key not in b:
            logger.warning(f"  {beam}: missing {key}, skipping beam")
            return None
    for key in ['segment_dist_x', 'ph_index_beg', 'segment_ph_cnt']:
        if key not in b['geolocation']:
            logger.warning(f"  {beam}: geolocation missing {key}, skipping beam")
            return None
    for key in ['tide_ocean', 'dem_h', 'dem_flag']:
        if key not in b['geophys_corr']:
            logger.warning(f"  {beam}: geophys_corr missing {key}, skipping beam")
            return None
    for key in ['delta_time', 'bckgrd_rate', 'bckgrd_counts']:
        if key not in b['bckgrd_atlas']:
            logger.warning(f"  {beam}: bckgrd_atlas missing {key}, skipping beam")
            return None

    h_ph = b['h_ph'][:].astype(np.float64)
    lat_ph = b['lat_ph'][:]
    lon_ph = b['lon_ph'][:]
    conf = b['signal_conf_ph'][:]
    delta_time = b['delta_time'][:].astype(np.float64)
    dist_ph_along = b['dist_ph_along'][:].astype(np.float64)
    n_ph = h_ph.size
    logger.info(f"  Raw photons: {n_ph:,}")

    if conf.ndim == 2:
        conf = conf[:, SEA_ICE_SURFACE_TYPE]
    elif conf.ndim != 1:
        logger.warning(f"  {beam}: unexpected signal_conf_ph shape {conf.shape}")
        return None

    quality = b['quality_ph'][:] if 'quality_ph' in b else np.zeros(n_ph, dtype=np.int8)

    # Shot identity for photon-rate features and FPB strength
    if 'pce_mframe_cnt' in b and 'ph_id_pulse' in b:
        shot_id = b['pce_mframe_cnt'][:].astype(np.int64) * 1000 + b['ph_id_pulse'][:].astype(np.int64)
    else:
        logger.warning(f"  {beam}: no pce_mframe_cnt/ph_id_pulse, shots estimated from 10 kHz timing")
        shot_id = np.round(delta_time * 1e4).astype(np.int64)

    # Absolute along-track distance: segment start + offset within segment
    geo = b['geolocation']
    seg_dist_x = geo['segment_dist_x'][:].astype(np.float64)
    seg_of_ph = _photon_segment_index(geo['ph_index_beg'][:], geo['segment_ph_cnt'][:], n_ph)
    assigned = seg_of_ph >= 0
    if not np.all(assigned):
        logger.warning(f"  {beam}: {np.sum(~assigned):,} photons not assigned to a segment, dropped")
    x_atc = np.full(n_ph, np.nan)
    x_atc[assigned] = seg_dist_x[seg_of_ph[assigned]] + dist_ph_along[assigned]

    # Segment-rate corrections, mapped to photons
    gc = b['geophys_corr']
    tide_seg = gc['tide_ocean'][:].astype(np.float64)
    dem_seg = gc['dem_h'][:].astype(np.float64)
    flag_seg = gc['dem_flag'][:]
    mss_seg = np.where(flag_seg == DEM_FLAG_MSS, dem_seg, np.nan)
    n_mss = int(np.sum(flag_seg == DEM_FLAG_MSS))
    logger.info(f"  MSS available (dem_flag==3) on {n_mss:,}/{flag_seg.size:,} geolocation segments")
    # ATL03 fill values are large; treat anything implausible as missing
    tide_seg[np.abs(tide_seg) > 100] = np.nan
    mss_seg[np.abs(mss_seg) > 1000] = np.nan
    tide_ph = np.full(n_ph, np.nan)
    mss_ph = np.full(n_ph, np.nan)
    tide_ph[assigned] = tide_seg[seg_of_ph[assigned]]
    mss_ph[assigned] = mss_seg[seg_of_ph[assigned]]

    # Photon selection: high-confidence sea-ice signal, nominal quality
    is_signal = conf >= SIGNAL_CONF_MIN
    is_high = (conf == HIGH_CONF) & (quality == 0) & assigned & np.isfinite(x_atc)
    logger.info(
        f"  Photons: signal(conf>=2)={np.sum(is_signal):,}  "
        f"high-conf sea ice, nominal quality={np.sum(is_high):,}"
    )
    if np.sum(is_high) == 0:
        logger.warning(f"  {beam}: no high-confidence sea-ice photons")
        return None

    # 2 m bins over the full photon extent (all photons define the shot grid)
    x_valid = x_atc[assigned]
    x0 = np.floor(x_valid.min() / BIN_SIZE_M) * BIN_SIZE_M
    bin_idx_all = np.full(n_ph, -1, dtype=np.int64)
    bin_idx_all[assigned] = np.floor((x_valid - x0) / BIN_SIZE_M).astype(np.int64)

    # Per-bin shot counts and signal counts from ALL photons in the bin, so a
    # pulse with only noise photons still counts as a shot
    bins_all = bin_idx_all[assigned]
    shots_per_bin = (
        pd.DataFrame({'b': bins_all, 's': shot_id[assigned]})
        .drop_duplicates().groupby('b').size()
    )
    signal_per_bin = pd.Series(bins_all[is_signal[assigned]]).value_counts()

    # High-confidence photons define the height statistics
    hb = bin_idx_all[is_high]
    df_ph = pd.DataFrame({
        'b': hb, 'h': h_ph[is_high], 'lat': lat_ph[is_high], 'lon': lon_ph[is_high],
        'x': x_atc[is_high], 't': delta_time[is_high],
        'tide': tide_ph[is_high], 'mss': mss_ph[is_high],
    })
    g = df_ph.groupby('b')
    seg = pd.DataFrame({
        'lat': g['lat'].mean(),
        'lon': g['lon'].mean(),
        'along_track_dist': x0 + (g['x'].mean().index.values + 0.5) * BIN_SIZE_M,
        'delta_time': g['t'].mean(),
        'height_mean_uncor': g['h'].mean(),
        'height_med_uncor': g['h'].median(),
        'std_h': g['h'].std(ddof=0).fillna(0.0),
        'photon_count': g['h'].size(),
        'h_p10': g['h'].quantile(0.10),
        'h_p90': g['h'].quantile(0.90),
        'tide_ocean': g['tide'].mean(),
        'mss': g['mss'].mean(),
    })
    seg['shots'] = shots_per_bin.reindex(seg.index).fillna(1).clip(lower=1).astype(int).values
    n_signal = signal_per_bin.reindex(seg.index).fillna(0).values
    # Photon rates in photons per laser shot (paper: "photon rate", ATL07 photon_rate)
    seg['pcnt'] = n_signal / seg['shots']
    seg['pcnth'] = seg['photon_count'] / seg['shots']

    # Background from ATL03 bckgrd_atlas (50-shot sums), interpolated in time
    bg = b['bckgrd_atlas']
    bg_t = bg['delta_time'][:].astype(np.float64)
    order = np.argsort(bg_t)
    bg_t = bg_t[order]
    bg_rate = bg['bckgrd_rate'][:].astype(np.float64)[order]
    bg_cnt = bg['bckgrd_counts'][:].astype(np.float64)[order]
    seg['brate'] = np.interp(seg['delta_time'].values, bg_t, bg_rate)
    seg['bcnt'] = np.interp(seg['delta_time'].values, bg_t, bg_cnt)
    seg['bg_rate'] = seg['brate']

    # First-photon bias (ATL07 ATBD Appendix G): width = 10%-90% cumulative
    # interval as two-way time, strength = signal photons per shot
    tables = _load_fpb_tables(granule_grp, beam)
    if tables is not None:
        width_ns = (seg['h_p90'] - seg['h_p10']).values * 2.0 / SPEED_OF_LIGHT * 1e9
        seg['fpb_corr'] = _fpb_correction_m(tables, width_ns, seg['pcnt'].values)
    else:
        seg['fpb_corr'] = 0.0

    # Geophysical correction, as in the authors' labeled data
    seg['mean_h'] = seg['height_mean_uncor'] - seg['mss'] - seg['tide_ocean'] - seg['fpb_corr']
    seg['median_h'] = seg['height_med_uncor'] - seg['mss'] - seg['tide_ocean'] - seg['fpb_corr']

    n_before = len(seg)
    seg = seg[np.isfinite(seg['mean_h'])]
    if len(seg) < n_before:
        logger.warning(
            f"  {beam}: dropped {n_before - len(seg):,} segments without MSS/tide "
            "(dem_flag != 3 or fill values)"
        )
    if seg.empty:
        return None

    seg = seg.sort_values('along_track_dist').reset_index(drop=True)
    # Rate-of-change features: first difference along track within the beam
    seg['d_pcnt'] = seg['pcnt'].diff().fillna(0.0)
    seg['d_brate'] = seg['brate'].diff().fillna(0.0)
    seg['beam'] = beam
    seg['granule'] = granule_key

    logger.info(
        f"  Segments: {len(seg):,}  corrected height range "
        f"{seg['mean_h'].min():.2f} to {seg['mean_h'].max():.2f} m  "
        f"FPB {seg['fpb_corr'].min()*100:.1f} to {seg['fpb_corr'].max()*100:.1f} cm"
    )
    return seg[OUTPUT_COLUMNS]


def preprocess_atl03(input_file, output_file):
    """
    Full preprocessing pipeline over every granule and strong beam.

    Args:
        input_file: Merged ATL03 HDF5 from download_atl03.py
        output_file: Output CSV file
    """
    import h5py
    import pandas as pd

    logger.info(f"Reading ATL03 data from {input_file}")
    all_segments = []

    with h5py.File(input_file, 'r') as h5:
        granule_keys = sorted(k for k in h5.keys() if k.startswith('granule_'))
        logger.info(f"Found {len(granule_keys)} granules")

        for granule_key in granule_keys:
            g = h5[granule_key]
            beams = sorted(k for k in g.keys() if k.startswith('gt'))
            for beam in beams:
                logger.info(f"Processing {granule_key}/{beam}")
                try:
                    seg = process_beam(g, granule_key, beam)
                except Exception as e:
                    logger.error(f"  {granule_key}/{beam} failed: {e}")
                    raise
                if seg is not None and not seg.empty:
                    all_segments.append(seg)

    if not all_segments:
        logger.error("No segments produced from any beam/granule")
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_file, index=False)
        sys.exit(1)

    result = pd.concat(all_segments, ignore_index=True)
    result.to_csv(output_file, index=False)
    logger.info(f"Preprocessed data saved to {output_file}")

    print(f"\n{'='*70}")
    print("ATL03 PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total segments: {len(result):,}")
    print(f"Corrected height range: {result['mean_h'].min():.2f} to {result['mean_h'].max():.2f} m")
    print(f"Mean photons per segment: {result['photon_count'].mean():.1f}")
    print(f"Mean photon rate: {result['pcnt'].mean():.2f} photons/shot")
    print(f"Mean background rate: {result['brate'].mean():.3e} counts/s")
    print(f"FPB correction: {result['fpb_corr'].mean()*100:.2f} cm mean")
    print(f"Beams: {result['beam'].nunique()}  Granules: {result['granule'].nunique()}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ICESat-2 ATL03 data into corrected 2m segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input atl03_data.h5 --output atl03_preprocessed.csv
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input merged ATL03 HDF5 file (from download_atl03.py)")
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
