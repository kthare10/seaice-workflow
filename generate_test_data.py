#!/usr/bin/env python3

"""
Generate synthetic test data for the sea ice workflow.

Builds two small synthetic raw ATL03 granules with the full structure the
pipeline reads (photon IDs, 20 m geolocation segments, geophys_corr with an
MSS flag, bckgrd_atlas, CAL-19 / CAL-42 tables, atlas_beam_type attributes,
sc_orient) and a planted surface: a repeating 200 m pattern of thick ice
(0.30 m freeboard), thin ice (0.05 m) and open water (0.00 m) on top of a
sloping mean sea surface plus ocean tide.

The granules are then pushed through the real bin/download_atl03.py merge and
bin/preprocess_atl03.py, so the test files always match the pipeline's current
schema. Labels come from the planted pattern.

Generated files:
  test_data/atl03_data.h5          - Merged ATL03 in the workflow's layout
  test_data/atl03_preprocessed.csv - Output of preprocess_atl03.py
  test_data/labeled_data.csv       - Preprocessed segments + planted labels

Usage:
    python generate_test_data.py
"""

import importlib.util
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDP_EPOCH_GPS = 1198800018.0        # 2018-01-01T00:00:00 GPS, in GPS seconds
TRACK_LENGTH_M = 2000.0
BLOCK_M = 200.0
FREEBOARD_BY_BLOCK = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.05, 4: 0.00}
LABEL_BY_BLOCK = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}     # thick, thick, thick, thin, water
TIDE_M = 0.057


def _load(module_name, rel_path):
    """Import a bin/ script as a module by path."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPT_DIR, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def utc_to_delta(dt):
    """UTC datetime -> ATL03 delta_time (seconds since the ATLAS SDP epoch)."""
    return (dt - datetime(1980, 1, 6)).total_seconds() + 18 - SDP_EPOCH_GPS


def block_of(x, x0):
    return (((x - x0) // BLOCK_M).astype(int)) % 5


def make_granule(path, t0_utc, x0, sc_orient, seed):
    """Write one synthetic raw ATL03 granule with all six beams."""
    import h5py

    r = np.random.default_rng(seed)
    strong_suffix = 'r' if sc_orient == 1 else 'l'

    with h5py.File(path, 'w') as f:
        anc = f.create_group('ancillary_data')
        anc.create_dataset('atlas_sdp_gps_epoch', data=np.array([SDP_EPOCH_GPS]))
        f.create_group('orbit_info').create_dataset('sc_orient', data=np.array([sc_orient], dtype=np.int8))
        cal = anc.create_group('calibrations')

        for beam in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
            strong = beam.endswith(strong_suffix)

            # CAL-19 first-photon-bias tables (ns / events per shot / ps) and CAL-42 dead time
            dt_axis = np.array([2.5, 3.0, 3.5])
            w_axis = np.linspace(0.3, 12.0, 12)
            s_axis = np.linspace(0.5, 25.0, 15)
            W, S, T = np.meshgrid(w_axis, s_axis, dt_axis, indexing='ij')
            fpb = cal.require_group('first_photon_bias').create_group(beam)
            fpb.create_dataset('dead_time', data=dt_axis)
            fpb.create_dataset('width', data=np.repeat(w_axis[:, None], 3, axis=1))
            fpb.create_dataset('strength', data=np.repeat(s_axis[:, None], 3, axis=1))
            fpb.create_dataset('ffb_corr', data=40.0 * S * np.exp(-W / 6.0) * (T / 3.0))
            dtg = cal.require_group('dead_time').create_group(beam)
            dtg.create_dataset('dead_time', data=np.full(20, 3.1e-9))
            dtg.create_dataset('sigma', data=np.full(20, 1e-10))

            # Shots every 0.7 m; photons per shot ~ Poisson(6) strong / (1.5) weak
            n_shots = int(TRACK_LENGTH_M / 0.7)
            shot = np.arange(n_shots)
            x_shot = x0 + shot * 0.7 + r.normal(0, 0.05, n_shots)
            fb = np.vectorize(FREEBOARD_BY_BLOCK.get)(block_of(x_shot, x0))
            mss = -62.0 - 0.002 * (x_shot - x0) / 1000.0
            true_h = mss + TIDE_M + fb

            nph = r.poisson(6.0 if strong else 1.5, n_shots)
            ph_shot = np.repeat(shot, nph)
            h = np.repeat(true_h, nph) + r.normal(0, 0.08, nph.sum())
            conf = np.full(h.size, 4, dtype=np.int8)
            # noise photons (conf 0) and low/medium photons (conf 2/3)
            ph_shot_n = np.repeat(shot, 2)
            h_n = np.repeat(true_h, 2) + r.uniform(-20, 20, 2 * n_shots)
            conf_n = np.zeros(2 * n_shots, dtype=np.int8)
            h_m = true_h + r.normal(0, 0.3, n_shots)
            conf_m = r.choice([2, 3], n_shots).astype(np.int8)
            ph_shot = np.concatenate([ph_shot, ph_shot_n, shot])
            h = np.concatenate([h, h_n, h_m])
            conf1 = np.concatenate([conf, conf_n, conf_m])
            x_ph = x_shot[ph_shot] + r.normal(0, 0.02, h.size)
            n = h.size
            conf5 = np.zeros((n, 5), dtype=np.int8)
            conf5[:, 1] = conf1
            conf5[:, 2] = conf1
            quality = np.zeros(n, dtype=np.int8)
            quality[r.random(n) < 0.01] = 3
            delta_time = (utc_to_delta(t0_utc) + shot * 1e-4)[ph_shot]
            lat = -75.0 - (x_ph - x0) / 111000.0
            lon = -170.0 + (x_ph - x0) / (111000 * np.cos(np.radians(75))) * 0.3

            # 20 m geolocation segments; photons ordered by segment
            n_seg = int(np.ceil(TRACK_LENGTH_M / 20.0)) + 1
            seg_x = x0 + 20.0 * np.arange(n_seg)
            seg_of_ph = np.clip(((x_ph - x0) // 20).astype(int), 0, n_seg - 1)
            order = np.lexsort((ph_shot, seg_of_ph))
            h, conf5, quality, delta_time = h[order], conf5[order], quality[order], delta_time[order]
            x_ph, lat, lon, ph_shot, seg_of_ph = x_ph[order], lat[order], lon[order], ph_shot[order], seg_of_ph[order]
            seg_cnt = np.bincount(seg_of_ph, minlength=n_seg)
            ph_index_beg = np.concatenate([[0], np.cumsum(seg_cnt)[:-1]]) + 1
            ph_index_beg[seg_cnt == 0] = 0

            bg = f.create_group(beam)
            bg.attrs['atlas_beam_type'] = np.bytes_(b'strong' if strong else b'weak')
            hg = bg.create_group('heights')
            hg.create_dataset('h_ph', data=h.astype(np.float32))
            hg.create_dataset('lat_ph', data=lat)
            hg.create_dataset('lon_ph', data=lon)
            hg.create_dataset('signal_conf_ph', data=conf5)
            hg.create_dataset('quality_ph', data=quality)
            hg.create_dataset('delta_time', data=delta_time)
            hg.create_dataset('dist_ph_along', data=(x_ph - seg_x[seg_of_ph]).astype(np.float32))
            hg.create_dataset('pce_mframe_cnt', data=(ph_shot // 200 + 1000).astype(np.uint32))
            hg.create_dataset('ph_id_pulse', data=(ph_shot % 200 + 1).astype(np.uint8))
            gg = bg.create_group('geolocation')
            seg_t = utc_to_delta(t0_utc) + (seg_x - x0) / 0.7 * 1e-4
            gg.create_dataset('segment_id', data=np.arange(n_seg) + 500000)
            gg.create_dataset('segment_dist_x', data=seg_x)
            gg.create_dataset('segment_length', data=np.full(n_seg, 20.0))
            gg.create_dataset('segment_ph_cnt', data=seg_cnt.astype(np.int32))
            gg.create_dataset('ph_index_beg', data=ph_index_beg.astype(np.int64))
            gg.create_dataset('delta_time', data=seg_t)
            gg.create_dataset('reference_photon_lat', data=-75.0 - (seg_x - x0) / 111000.0)
            gg.create_dataset('reference_photon_lon', data=np.full(n_seg, -170.0))
            gc = bg.create_group('geophys_corr')
            gc.create_dataset('delta_time', data=seg_t)
            gc.create_dataset('tide_ocean', data=np.full(n_seg, TIDE_M, dtype=np.float32))
            gc.create_dataset('dem_h', data=(-62.0 - 0.002 * (seg_x - x0) / 1000.0).astype(np.float32))
            gc.create_dataset('dem_flag', data=np.full(n_seg, 3, dtype=np.int8))
            gc.create_dataset('geoid', data=np.full(n_seg, -56.8, dtype=np.float32))
            gc.create_dataset('dac', data=np.full(n_seg, 0.15, dtype=np.float32))
            ba = bg.create_group('bckgrd_atlas')
            nb = n_shots // 50 + 1
            ba.create_dataset('delta_time', data=utc_to_delta(t0_utc) + np.arange(nb) * 50e-4)
            ba.create_dataset('bckgrd_rate', data=(6.4e6 + r.normal(0, 2e5, nb)).astype(np.float32))
            ba.create_dataset('bckgrd_counts', data=(1500 + r.normal(0, 50, nb)).astype(np.int32))


def main():
    import pandas as pd

    test_data_dir = os.path.join(SCRIPT_DIR, 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    download = _load('download_atl03', 'bin/download_atl03.py')
    preprocess = _load('preprocess_atl03', 'bin/preprocess_atl03.py')

    granules = [
        ('ATL03_20191104195311_05940510_006_01.h5', datetime(2019, 11, 4, 19, 53, 11), 28_400_000.0, 1, 1),
        ('ATL03_20191126182014_09290510_006_01.h5', datetime(2019, 11, 26, 18, 20, 14), 28_300_000.0, 0, 2),
    ]
    x0_by_granule = {}

    tmp = tempfile.mkdtemp(prefix='seaice_synth_')
    try:
        raw_files = []
        for i, (name, t0, x0, sc_orient, seed) in enumerate(granules):
            path = os.path.join(tmp, name)
            make_granule(path, t0, x0, sc_orient, seed)
            raw_files.append(path)
            x0_by_granule[f'granule_{i:04d}'] = x0
        print(f"Generated {len(raw_files)} synthetic raw ATL03 granules")

        h5_path = os.path.join(test_data_dir, 'atl03_data.h5')
        cwd = os.getcwd()
        os.chdir(test_data_dir)     # merge writes atl03_bbox.json next to the h5
        try:
            download._merge_granules(raw_files, h5_path, 'ross_sea',
                                     download.REGIONS['ross_sea'], '2019-11-01', '2019-11-30')
        finally:
            os.chdir(cwd)
        print(f"Generated merged ATL03: {h5_path}")

        pre_path = os.path.join(test_data_dir, 'atl03_preprocessed.csv')
        preprocess.preprocess_atl03(h5_path, pre_path)
        print(f"Generated preprocessed CSV: {pre_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Labels from the planted pattern
    df = pd.read_csv(pre_path)
    x0 = df['granule'].map(x0_by_granule).values
    df['label'] = np.vectorize(LABEL_BY_BLOCK.get)(block_of(df['along_track_dist'].values, x0))
    labeled_path = os.path.join(test_data_dir, 'labeled_data.csv')
    df.to_csv(labeled_path, index=False)
    counts = df['label'].value_counts().sort_index().to_dict()
    print(f"Generated labeled CSV: {labeled_path}  ({len(df):,} segments, labels {counts})")

    print("\nTest data generation complete!")
    print(f"Output directory: {test_data_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Failed to generate test data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
