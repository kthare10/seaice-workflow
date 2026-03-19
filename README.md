# Sea Ice Classification & Freeboard from ICESat-2 ATL03

A Pegasus workflow for scalable, higher resolution polar sea ice classification
and freeboard calculation from ICESat-2 ATL03 photon-level data.

Based on: *"Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data"* (Iqrah et al., IPDPSW 2025)

## Pipeline Overview

```
Full mode (single classify job):
  download_atl03 ──┬──> preprocess_atl03 ──┐
                   │                        ├──> auto_label ──> train_model [GPU] ──> classify_seaice [GPU] ──> calculate_freeboard ──> visualize_results
                   └──> download_sentinel2 ─┘
                        (uses ATL03 track bbox)

Full mode with --max-granules N (parallel classify jobs):
  download_atl03 ──┬──> preprocess_atl03 ──┐                        ┌─ classify_seaice_0 [GPU] ─┐
                   │                        ├──> auto_label ──> train ├─ classify_seaice_1 [GPU] ─├─> merge ──> calculate_freeboard ──> visualize
                   └──> download_sentinel2 ─┘                        └─ classify_seaice_N [GPU] ─┘

Test mode (--test-mode, 2 parallel classify jobs):
  [test_data/atl03_data.h5] ──> preprocess_atl03 ──┐            ┌─ classify_seaice_0 [GPU] ─┐
  [test_data/labeled_data.csv] ────────────────────>├──> train ──├─ classify_seaice_1 [GPU] ─├──> merge ──> calculate_freeboard ──> visualize
                                                                 └───────────────────────────┘
```

| Stage | Description | Memory | GPU |
|-------|-------------|--------|-----|
| `download_atl03` | Fetch ICESat-2 ATL03 HDF5 granules from NASA Earthdata | 4 GB | No |
| `download_sentinel2` | Fetch coincident Sentinel-2 imagery (parallel) | 4 GB | No |
| `preprocess_atl03` | Filter photons, resample to 2m segments, compute features | 8 GB | No |
| `auto_label` | Co-register S2 with ATL03, classify S2, overlay labels | 4 GB | No |
| `train_model` | Train LSTM or MLP classifier on labeled data | 16 GB | Yes |
| `classify_seaice` | Run inference on full ATL03 dataset (parallelized per granule when `--max-granules` is set) | 8 GB | Yes |
| `merge_classifications` | Concatenate per-granule classification CSVs (only in parallel mode) | 4 GB | No |
| `calculate_freeboard` | Sliding-window sea surface detection + freeboard | 8 GB | No |
| `visualize_results` | Generate maps, profiles, summary statistics | 4 GB | No |

## Execution Environments

This workflow requires Pegasus WMS and HTCondor. Two options are available:

### Option A: FABRIC Testbed (Recommended for GPU Workflows)

Deploy a dedicated Pegasus/HTCondor cluster on [FABRIC](https://portal.fabric-testbed.net/) using the automated provisioning notebook.

**Prerequisites:**
- A FABRIC account and active project allocation
- JupyterHub access via the FABRIC portal

**Setup:**

1. Open the **PegasusAI** artifact on FABRIC:
   <https://artifacts.fabric-testbed.net/artifacts/53da4088-a175-4f0c-9e25-a4a371032a39>

2. Download the `.tgz` archive and upload the notebook to the FABRIC JupyterHub, or clone the artifact directly in a FABRIC Jupyter terminal.

3. Run the notebook cells to:
   - Create a FABRIC slice with a submit node and one or more worker nodes across FABRIC sites
   - Configure FABNetv4 networking between all nodes
   - Install HTCondor (Central Manager on submit node, execute daemons on workers)
   - Install Pegasus WMS on the submit node
   - Set up passwordless SSH and hostname resolution (`/etc/hosts`)

4. Once the cluster is running, SSH into the submit node and clone this repository:

   ```bash
   git clone <repo-url> && cd seaice-workflow
   ```

5. Follow the [Generate and Submit Workflow](#generate-and-submit-workflow) instructions below.

> **Note:** FABRIC worker nodes can be provisioned with NVIDIA GPUs (e.g., RTX6000, A30, A40) for the `train_model` and `classify_seaice` stages. Request GPU components in the notebook when creating your slice.

### Option B: ACCESS Pegasus (Hosted Environment)

[ACCESS Pegasus](https://pegasus.access-ci.org/) is a hosted workflow environment — no cluster setup required. A built-in **test pool** lets you get started immediately without an allocation.

**Setup:**

1. Log in at <https://pegasus.access-ci.org/> using your ACCESS credentials (single sign-on).
2. Open a Jupyter notebook or terminal from the Open OnDemand dashboard.
3. Clone this repository:

   ```bash
   git clone <repo-url> && cd seaice-workflow
   ```

4. **To get started quickly**, submit workflows to the built-in test pool — no allocation needed:

   ```bash
   pegasus-plan --submit -s condorpool -o local workflow.yml
   ```

5. **To scale up**, request an [ACCESS allocation](https://allocations.access-ci.org/) and use **HTCondor Annex** to provision pilot jobs on allocated resources (see the [ACCESS Pegasus examples](https://github.com/pegasus-isi/ACCESS-Pegasus-Examples)).
6. Follow the [Generate and Submit Workflow](#generate-and-submit-workflow) instructions below.

> **Note:** The test pool has limited resources and no GPUs. For the GPU-accelerated `train_model` and `classify_seaice` stages, provision GPU nodes via HTCondor Annex with an ACCESS allocation.

## Quick Start

### Prerequisites

- Python 3.10+
- [Pegasus WMS](https://pegasus.isi.edu/) 5.0+
- HTCondor (for condorpool execution)
- NVIDIA GPU with CUDA drivers on worker nodes (for training/classification)
- NASA Earthdata account (see below)

Both execution environments above (FABRIC, ACCESS Pegasus) satisfy these prerequisites automatically.

### NASA Earthdata Credentials

The `download_atl03` stage authenticates with NASA Earthdata. Two methods are supported:

**Option 1: Bearer token (recommended for FABRIC)**

Pre-generate a token from a machine that can reach `urs.earthdata.nasa.gov`:

1. Create an account at <https://urs.earthdata.nasa.gov/>
2. Generate a token at `https://urs.earthdata.nasa.gov/users/<username>/user_tokens`
3. Pass it via `--earthdata-token` or `EARTHDATA_TOKEN` env var:

```bash
export EARTHDATA_TOKEN="your_token_here"
```

This bypasses the login endpoint, which is useful on networks (like FABRIC)
where `urs.earthdata.nasa.gov` is unreachable.

**Option 2: Username/password**

```bash
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

### Generate and Submit Workflow

```bash
# Generate workflow DAG (token from $EARTHDATA_TOKEN)
python workflow_generator.py --region ross_sea \
                              --start-date 2019-11-01 \
                              --end-date 2019-11-30 \
                              --output workflow.yml

# Or pass token explicitly
python workflow_generator.py --region ross_sea \
                              --start-date 2019-11-01 \
                              --end-date 2019-11-30 \
                              --earthdata-token "your_token" \
                              --output workflow.yml

# Submit to HTCondor
pegasus-plan --submit -s condorpool -o local workflow.yml

# Monitor
pegasus-status <run-dir>
```

### Test Mode (No Downloads)

To test the workflow end-to-end without downloading real data, use `--test-mode`.
This skips the download and auto-label jobs and uses pre-generated synthetic data:

```bash
# Generate synthetic test data (one-time setup)
python generate_test_data.py

# Generate workflow using test data
python workflow_generator.py --test-mode --output workflow_test.yml

# Submit
pegasus-plan --submit -s condorpool -o local workflow_test.yml
```

In test mode, `--start-date` and Earthdata credentials are not required.

### Limited Download Mode

To run with real data but limit download volume for faster testing, use
`--max-granules` and/or `--max-scenes`:

```bash
# Download only 2 ATL03 granules and 3 Sentinel-2 scenes
python workflow_generator.py --region ross_sea \
                              --start-date 2019-11-01 \
                              --end-date 2019-11-07 \
                              --max-granules 2 \
                              --max-scenes 3 \
                              --output workflow_limited.yml
```

### Command-Line Options

```
--region              Region name: ross_sea, weddell_sea, beaufort_sea, arctic_ocean, southern_ocean
--start-date          Start date (YYYY-MM-DD). Required unless --test-mode is used.
--end-date            End date (YYYY-MM-DD), defaults to start_date + 30 days
--test-mode           Use synthetic test data (skips downloads and auto-label)
--max-granules        Max ATL03 granules to download (default: all)
--max-scenes          Max Sentinel-2 scenes to download (default: 10)
--earthdata-token     Pre-generated bearer token (default: $EARTHDATA_TOKEN)
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
