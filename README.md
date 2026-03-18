# Sea Ice Classification & Freeboard from ICESat-2 ATL03

A Pegasus workflow for scalable, higher resolution polar sea ice classification
and freeboard calculation from ICESat-2 ATL03 photon-level data.

Based on: *"Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data"* (Iqrah et al., IPDPSW 2025)

## Pipeline Overview

```
download_atl03 ──────┐
                     ├──> preprocess_atl03 ──> auto_label ──> train_model [GPU] ──> classify_seaice [GPU] ──> calculate_freeboard ──> visualize_results
download_sentinel2 ──┘
```

| Stage | Description | Memory | GPU |
|-------|-------------|--------|-----|
| `download_atl03` | Fetch ICESat-2 ATL03 HDF5 granules from NASA Earthdata | 4 GB | No |
| `download_sentinel2` | Fetch coincident Sentinel-2 imagery (parallel) | 4 GB | No |
| `preprocess_atl03` | Filter photons, resample to 2m segments, compute features | 8 GB | No |
| `auto_label` | Co-register S2 with ATL03, classify S2, overlay labels | 4 GB | No |
| `train_model` | Train LSTM or MLP classifier on labeled data | 16 GB | Yes |
| `classify_seaice` | Run inference on full ATL03 dataset | 8 GB | Yes |
| `calculate_freeboard` | Sliding-window sea surface detection + freeboard | 8 GB | No |
| `visualize_results` | Generate maps, profiles, summary statistics | 4 GB | No |

## Quick Start

### Prerequisites

- Python 3.10+
- [Pegasus WMS](https://pegasus.isi.edu/) 5.0+
- HTCondor (for condorpool execution)
- NVIDIA GPU with CUDA drivers on worker nodes (for training/classification)
- NASA Earthdata account (see below)

### NASA Earthdata Credentials

The `download_atl03` stage authenticates with NASA Earthdata using username/password
passed via `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables.

1. Create an account at <https://urs.earthdata.nasa.gov/>
2. Pass credentials to the workflow generator via CLI args or environment variables:

```bash
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

The workflow generator will warn if credentials are not provided.

### Generate and Submit Workflow

```bash
# Generate workflow DAG (credentials from env vars)
python workflow_generator.py --region ross_sea \
                              --start-date 2019-11-01 \
                              --end-date 2019-11-30 \
                              --output workflow.yml

# Or pass credentials explicitly
python workflow_generator.py --region ross_sea \
                              --start-date 2019-11-01 \
                              --end-date 2019-11-30 \
                              --earthdata-username "user" \
                              --earthdata-password "pass" \
                              --output workflow.yml

# Submit to HTCondor
pegasus-plan --submit -s condorpool -o local workflow.yml

# Monitor
pegasus-status <run-dir>
```

### Command-Line Options

```
--region              Region name: ross_sea, weddell_sea, beaufort_sea, arctic_ocean, southern_ocean
--start-date          Start date (YYYY-MM-DD)
--end-date            End date (YYYY-MM-DD), defaults to start_date + 30 days
--earthdata-username  NASA Earthdata username (default: $EARTHDATA_USERNAME)
--earthdata-password  NASA Earthdata password (default: $EARTHDATA_PASSWORD)
--granule-id          Specific ATL03 granule ID (optional)
--model-type          Classifier type: lstm (default) or mlp
-e                    Execution site name (default: condorpool)
-o                    Output workflow file (default: workflow.yml)
```

## GPU Acceleration

The `train_model` and `classify_seaice` stages are GPU-accelerated using
TensorFlow with CUDA. The workflow requests 1 GPU per job via HTCondor
(`request_gpus=1`) and uses Singularity `--nv` for NVIDIA GPU passthrough.

### Requirements

- NVIDIA GPU with CUDA 12.3+ compatible drivers on worker nodes
- Singularity/Apptainer with `--nv` support

The container image (`kthare10/seaice-icesat2:latest`) is built on
`nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04` with `tensorflow[and-cuda]`.
If no GPU is available, TensorFlow falls back to CPU automatically.

## Scientific Details

### ATL03 Preprocessing

- Reads photon heights (`h_ph`), positions (`lat_ph`, `lon_ph`), and signal confidence (`signal_conf_ph`)
- Filters for high-confidence signal photons (confidence >= 3 on sea ice surface)
- Resamples to 2m along-track bins
- Per bin: mean/median/std height, photon count, background rate
- Applies first-photon bias correction

### Auto-Labeling

- Reprojects to EPSG:3976 (Antarctic Polar Stereographic)
- Classifies S2 imagery using band ratios: thick ice (high reflectance), thin ice (moderate, blue/red > 1), open water (low NIR)
- Overlays S2-derived labels onto nearest ATL03 2m segments

### Model Architecture

**LSTM**: 1 LSTM layer (16 units, ELU) + 7 Dense layers -> softmax(3)
**MLP**: Dense(32, ReLU) -> Dense(3, softmax)

Both use focal loss for class imbalance and Adam optimizer (lr=0.003).

### Freeboard Calculation

- 10km sliding window (5km radius)
- Identifies open water segments within window
- Computes local sea surface height using distance-weighted mean
- Linear interpolation where no open water exists
- Freeboard = segment_elevation - local_sea_surface

## Container Image

Build the GPU-enabled Docker image:

```bash
docker build -t kthare10/seaice-icesat2:latest -f Docker/Seaice_Dockerfile .
```

The workflow uses Singularity to pull from Docker Hub at runtime. GPU stages
use `--nv` for NVIDIA device passthrough.

## Output Files

| File | Description |
|------|-------------|
| `atl03_data.h5` | Raw ATL03 photon data (HDF5) |
| `sentinel2_scenes.tar.gz` | Sentinel-2 band imagery |
| `atl03_preprocessed.csv` | 2m segment features |
| `labeled_data.csv` | Auto-labeled training data |
| `model.h5` | Trained classifier weights |
| `training_metrics.json` | Training loss/accuracy history |
| `classification_results.csv` | Per-segment ice type predictions |
| `freeboard_results.csv` | Per-segment freeboard values |
| `classification_map.png` | Geographic classification map |
| `freeboard_profile.png` | Along-track freeboard profile |
| `summary_statistics.json` | Aggregate statistics |

## References

- Iqrah et al., "Scalable Higher Resolution Polar Sea Ice Classification and Freeboard Calculation from ICESat-2 ATL03 Data", IPDPSW 2025
- ICESat-2 ATL03: https://nsidc.org/data/atl03
- Sentinel-2 via Planetary Computer: https://planetarycomputer.microsoft.com
