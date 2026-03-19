#!/usr/bin/env python3

"""
Generate synthetic test data for the sea ice workflow.

Creates realistic test data at pipeline boundaries so the workflow can be
tested end-to-end without live APIs or heavy geospatial dependencies.

Generated files:
  test_data/atl03_data.h5         - Synthetic HDF5 mimicking ICESat-2 ATL03
  test_data/atl03_preprocessed.csv - Pre-built preprocessed CSV
  test_data/labeled_data.csv       - Preprocessed CSV with labels

Usage:
    python generate_test_data.py
"""

import os
import sys

import numpy as np


def generate_atl03_h5(output_path):
    """
    Generate synthetic ATL03 HDF5 file with 2 granules x 3 beams.

    Each beam has ~500 photons with heights simulating three ice types:
      - Open water: ~0.0 m
      - Thin ice:   ~0.15 m
      - Thick ice:  ~0.5 m
    """
    import h5py

    np.random.seed(42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    beams = ['gt1l', 'gt2l', 'gt3l']
    n_photons_per_beam = 500

    with h5py.File(output_path, 'w') as f:
        for g_idx in range(2):
            granule_key = f"granule_{g_idx:03d}"

            for beam in beams:
                beam_path = f"{granule_key}/{beam}"
                grp = f.create_group(beam_path)

                # Along-track distance: 0 to 1000 m
                dist_along = np.sort(
                    np.random.uniform(0, 1000, n_photons_per_beam)
                )

                # Simulate three ice types based on position
                # 0-333m: thick ice, 333-666m: thin ice, 666-1000m: open water
                h_ph = np.zeros(n_photons_per_beam)
                # Real ATL03 signal_conf_ph is (photons, 5) for 5 surface types:
                # 0=land, 1=ocean, 2=sea_ice, 3=land_ice, 4=inland_water
                # Sea ice classification uses column 2 (sea_ice surface type)
                # Pipeline reads column 2; we populate all 5 columns
                signal_conf = np.zeros((n_photons_per_beam, 5), dtype=np.int8)

                for i in range(n_photons_per_beam):
                    d = dist_along[i]
                    if d < 333:
                        # Thick ice: mean ~0.5m, std ~0.05m
                        h_ph[i] = np.random.normal(0.5, 0.05)
                        conf = np.random.choice([3, 4], p=[0.3, 0.7])
                    elif d < 666:
                        # Thin ice: mean ~0.15m, std ~0.03m
                        h_ph[i] = np.random.normal(0.15, 0.03)
                        conf = np.random.choice([3, 4], p=[0.4, 0.6])
                    else:
                        # Open water: mean ~0.0m, std ~0.02m
                        h_ph[i] = np.random.normal(0.0, 0.02)
                        conf = np.random.choice([3, 4], p=[0.3, 0.7])
                    signal_conf[i, :] = conf  # Same confidence across all surface types

                # Add some noise photons (low confidence)
                noise_mask = np.random.random(n_photons_per_beam) < 0.1
                noise_conf = np.random.choice([0, 1, 2], size=noise_mask.sum())
                signal_conf[noise_mask, :] = noise_conf[:, np.newaxis]
                h_ph[noise_mask] += np.random.normal(0, 2.0, size=noise_mask.sum())

                # Lat/lon: simulate Antarctic track
                base_lat = -75.0 + g_idx * 0.5
                base_lon = 170.0 + g_idx * 1.0
                lat_ph = base_lat + dist_along / 111000.0
                lon_ph = base_lon + np.random.normal(0, 0.0001, n_photons_per_beam)

                grp.create_dataset('h_ph', data=h_ph)
                grp.create_dataset('lat_ph', data=lat_ph)
                grp.create_dataset('lon_ph', data=lon_ph)
                grp.create_dataset('signal_conf_ph', data=signal_conf)
                grp.create_dataset('dist_ph_along', data=dist_along)

    print(f"Generated ATL03 HDF5: {output_path}")
    print(f"  2 granules x 3 beams x {n_photons_per_beam} photons")


def generate_preprocessed_csv(output_path):
    """
    Generate pre-built preprocessed CSV with segment-level features.

    ~500 rows with columns matching preprocess_atl03.py output.
    """
    np.random.seed(42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    beams = ['gt1l', 'gt2l', 'gt3l']

    for g_idx in range(2):
        granule = f"granule_{g_idx:03d}"
        for beam in beams:
            n_segments = np.random.randint(75, 95)
            base_lat = -75.0 + g_idx * 0.5
            base_lon = 170.0 + g_idx * 1.0

            for s in range(n_segments):
                frac = s / max(n_segments - 1, 1)
                dist = s * 2.0 + 1.0  # 2m segments

                # Height profile: thick ice -> thin ice -> open water
                if frac < 0.33:
                    mean_h = np.random.normal(0.5, 0.05)
                    median_h = mean_h + np.random.normal(0, 0.01)
                    std_h = abs(np.random.normal(0.05, 0.01))
                elif frac < 0.66:
                    mean_h = np.random.normal(0.15, 0.03)
                    median_h = mean_h + np.random.normal(0, 0.005)
                    std_h = abs(np.random.normal(0.03, 0.005))
                else:
                    mean_h = np.random.normal(0.0, 0.02)
                    median_h = mean_h + np.random.normal(0, 0.005)
                    std_h = abs(np.random.normal(0.02, 0.005))

                rows.append({
                    'lat': base_lat + dist / 111000.0,
                    'lon': base_lon + np.random.normal(0, 0.0001),
                    'along_track_dist': dist,
                    'mean_h': mean_h,
                    'median_h': median_h,
                    'std_h': std_h,
                    'photon_count': np.random.randint(5, 30),
                    'bg_rate': abs(np.random.normal(0.5, 0.2)),
                    'beam': beam,
                    'granule': granule,
                })

    # Write CSV
    header = 'lat,lon,along_track_dist,mean_h,median_h,std_h,photon_count,bg_rate,beam,granule'
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            line = (f"{r['lat']:.8f},{r['lon']:.8f},{r['along_track_dist']:.2f},"
                    f"{r['mean_h']:.6f},{r['median_h']:.6f},{r['std_h']:.6f},"
                    f"{r['photon_count']},{r['bg_rate']:.6f},{r['beam']},{r['granule']}")
            f.write(line + '\n')

    print(f"Generated preprocessed CSV: {output_path}")
    print(f"  {len(rows)} segments")


def generate_labeled_csv(output_path):
    """
    Generate labeled CSV (preprocessed + label column).

    Labels are assigned based on height:
      0 = thick ice  (mean_h > 0.3)
      1 = thin ice   (0.05 < mean_h <= 0.3)
      2 = open water (mean_h <= 0.05)
    """
    np.random.seed(42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    beams = ['gt1l', 'gt2l', 'gt3l']

    for g_idx in range(2):
        granule = f"granule_{g_idx:03d}"
        for beam in beams:
            n_segments = np.random.randint(75, 95)
            base_lat = -75.0 + g_idx * 0.5
            base_lon = 170.0 + g_idx * 1.0

            for s in range(n_segments):
                frac = s / max(n_segments - 1, 1)
                dist = s * 2.0 + 1.0

                if frac < 0.33:
                    mean_h = np.random.normal(0.5, 0.05)
                    median_h = mean_h + np.random.normal(0, 0.01)
                    std_h = abs(np.random.normal(0.05, 0.01))
                elif frac < 0.66:
                    mean_h = np.random.normal(0.15, 0.03)
                    median_h = mean_h + np.random.normal(0, 0.005)
                    std_h = abs(np.random.normal(0.03, 0.005))
                else:
                    mean_h = np.random.normal(0.0, 0.02)
                    median_h = mean_h + np.random.normal(0, 0.005)
                    std_h = abs(np.random.normal(0.02, 0.005))

                # Assign label based on height
                if mean_h > 0.3:
                    label = 0  # thick ice
                elif mean_h > 0.05:
                    label = 1  # thin ice
                else:
                    label = 2  # open water

                rows.append({
                    'lat': base_lat + dist / 111000.0,
                    'lon': base_lon + np.random.normal(0, 0.0001),
                    'along_track_dist': dist,
                    'mean_h': mean_h,
                    'median_h': median_h,
                    'std_h': std_h,
                    'photon_count': np.random.randint(5, 30),
                    'bg_rate': abs(np.random.normal(0.5, 0.2)),
                    'beam': beam,
                    'granule': granule,
                    'label': label,
                })

    header = 'lat,lon,along_track_dist,mean_h,median_h,std_h,photon_count,bg_rate,beam,granule,label'
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            line = (f"{r['lat']:.8f},{r['lon']:.8f},{r['along_track_dist']:.2f},"
                    f"{r['mean_h']:.6f},{r['median_h']:.6f},{r['std_h']:.6f},"
                    f"{r['photon_count']},{r['bg_rate']:.6f},{r['beam']},{r['granule']},"
                    f"{r['label']}")
            f.write(line + '\n')

    # Count labels
    label_counts = {}
    for r in rows:
        label_counts[r['label']] = label_counts.get(r['label'], 0) + 1

    print(f"Generated labeled CSV: {output_path}")
    print(f"  {len(rows)} segments")
    print(f"  Labels: {label_counts}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, "test_data")

    print("=" * 60)
    print("GENERATING SYNTHETIC TEST DATA")
    print("=" * 60)

    generate_atl03_h5(os.path.join(test_data_dir, "atl03_data.h5"))
    generate_preprocessed_csv(os.path.join(test_data_dir, "atl03_preprocessed.csv"))
    generate_labeled_csv(os.path.join(test_data_dir, "labeled_data.csv"))

    print("=" * 60)
    print("TEST DATA GENERATION COMPLETE")
    print(f"Output directory: {test_data_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
