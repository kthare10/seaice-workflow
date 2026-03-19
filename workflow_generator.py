#!/usr/bin/env python3

"""
Pegasus workflow generator for sea ice classification and freeboard calculation
from ICESat-2 ATL03 data.

This script generates a Pegasus workflow implementing the pipeline from:
"Scalable Higher Resolution Polar Sea Ice Classification and Freeboard Calculation
from ICESat-2 ATL03 Data" (Iqrah et al., IPDPSW 2025)

Pipeline stages:
1. Download ICESat-2 ATL03 photon data from NASA Earthdata
2. Download coincident Sentinel-2 imagery (uses ATL03 track bbox for spatial filtering)
3. Preprocess ATL03: filter photons, resample to 2m segments, compute features
4. Auto-label: co-register S2 imagery with ATL03 tracks, overlay labels
5. Train LSTM/MLP classifier on labeled data
6. Classify sea ice types on full ATL03 dataset
7. Calculate freeboard using sliding-window sea surface detection
8. Visualize classification maps, freeboard profiles, summary statistics

Usage:
    ./workflow_generator.py --region ross_sea \
                            --start-date 2019-11-01 \
                            --end-date 2019-11-30 \
                            --output workflow.yml

    # With specific granule and MLP model
    ./workflow_generator.py --region ross_sea \
                            --start-date 2019-11-01 \
                            --granule-id ATL03_20191101... \
                            --model-type mlp \
                            --output workflow.yml
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Pegasus imports
from Pegasus.api import *

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeaIceWorkflow:
    """Sea ice classification and freeboard workflow generator."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "seaice"

    def __init__(self, dagfile="workflow.yml"):
        """Initialize workflow."""
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

    def write(self):
        """Write all catalogs and workflow to files."""
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    def create_pegasus_properties(self):
        """Create Pegasus properties configuration."""
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        self.props["pegasus.data.configuration"] = "condorio"

    def create_sites_catalog(self, exec_site_name="condorpool"):
        """Create site catalog."""
        logger.info(f"Creating site catalog for execution site: {exec_site_name}")
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(
                Directory.LOCAL_STORAGE, self.local_storage_dir
            ).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
            .add_pegasus_profile(data_configuration="condorio")
        )

        self.sc.add_sites(local, exec_site)

    def create_transformation_catalog(self, exec_site_name="condorpool"):
        """Create transformation catalog with executables and containers."""
        logger.info("Creating transformation catalog")
        self.tc = TransformationCatalog()

        # CPU container for non-GPU stages
        seaice_container = Container(
            "seaice_container",
            container_type=Container.SINGULARITY,
            image="docker://kthare10/seaice-icesat2:latest",
            image_site="docker_hub",
        )

        # GPU container for training and classification (--nv enables NVIDIA GPU passthrough)
        seaice_gpu_container = Container(
            "seaice_gpu_container",
            container_type=Container.SINGULARITY,
            image="docker://kthare10/seaice-icesat2:latest",
            image_site="docker_hub",
        ).add_env(SINGULARITY_ARGS="--nv")

        # Transformations
        download_atl03 = Transformation(
            "download_atl03",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/download_atl03.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="4 GB")

        download_sentinel2 = Transformation(
            "download_sentinel2",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/download_sentinel2.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="4 GB")

        preprocess_atl03 = Transformation(
            "preprocess_atl03",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/preprocess_atl03.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="8 GB")

        auto_label = Transformation(
            "auto_label",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/auto_label.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="4 GB")

        # GPU-accelerated stages: train_model and classify_seaice
        train_model = Transformation(
            "train_model",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/train_model.py"),
            is_stageable=True,
            container=seaice_gpu_container,
        ).add_pegasus_profile(memory="16 GB"
        ).add_condor_profile(request_gpus=1)

        classify_seaice = Transformation(
            "classify_seaice",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/classify_seaice.py"),
            is_stageable=True,
            container=seaice_gpu_container,
        ).add_pegasus_profile(memory="8 GB"
        ).add_condor_profile(request_gpus=1)

        calculate_freeboard = Transformation(
            "calculate_freeboard",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/calculate_freeboard.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="8 GB")

        visualize_results = Transformation(
            "visualize_results",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/visualize_results.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="4 GB")

        merge_classifications = Transformation(
            "merge_classifications",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/merge_classifications.py"),
            is_stageable=True,
            container=seaice_container,
        ).add_pegasus_profile(memory="4 GB")

        self.tc.add_containers(seaice_container, seaice_gpu_container)
        self.tc.add_transformations(
            download_atl03,
            download_sentinel2,
            preprocess_atl03,
            auto_label,
            train_model,
            classify_seaice,
            calculate_freeboard,
            visualize_results,
            merge_classifications,
        )

    def create_replica_catalog(self, test_mode=False):
        """Create replica catalog."""
        logger.info("Creating replica catalog")
        self.rc = ReplicaCatalog()

        if test_mode:
            # Register synthetic test data so download jobs can be skipped
            test_data_dir = os.path.join(self.wf_dir, "test_data")
            atl03_path = os.path.join(test_data_dir, "atl03_data.h5")
            labeled_path = os.path.join(test_data_dir, "labeled_data.csv")

            if not os.path.exists(atl03_path):
                logger.warning(
                    f"Test data not found at {test_data_dir}. "
                    "Run 'python generate_test_data.py' first."
                )

            self.rc.add_replica("local", "atl03_data.h5", "file://" + atl03_path)
            self.rc.add_replica("local", "labeled_data.csv", "file://" + labeled_path)
            logger.info(f"Registered test data from {test_data_dir}")

    def create_workflow(self, region, start_date, end_date, granule_id=None,
                        model_type="lstm", earthdata_token=None,
                        earthdata_username=None, earthdata_password=None,
                        test_mode=False, max_granules=None, max_scenes=None):
        """Create the workflow DAG."""
        logger.info("Creating workflow DAG")
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # Output files
        atl03_data = File("atl03_data.h5")
        atl03_bbox = File("atl03_bbox.json")
        sentinel2_scenes = File("sentinel2_scenes.tar.gz")
        atl03_preprocessed = File("atl03_preprocessed.csv")
        labeled_data = File("labeled_data.csv")
        model_file = File("model.h5")
        training_metrics = File("training_metrics.json")
        classification_results = File("classification_results.csv")
        freeboard_results = File("freeboard_results.csv")
        classification_map = File("classification_map.png")
        freeboard_profile = File("freeboard_profile.png")
        summary_stats = File("summary_statistics.json")

        if test_mode:
            # In test mode, skip downloads and use pre-generated synthetic data
            # Files are registered in the replica catalog by create_replica_catalog()
            logger.info("TEST MODE: Skipping download jobs, using synthetic test data")
        else:
            # Job 1: Download ATL03 data
            download_atl03_args = [
                "--region", region,
                "--start-date", start_date,
                "--end-date", end_date,
                "--output", atl03_data
            ]
            if granule_id:
                download_atl03_args.extend(["--granule-id", granule_id])
            if max_granules:
                download_atl03_args.extend(["--max-granules", str(max_granules)])

            download_atl03_job = (
                Job(
                    "download_atl03",
                    _id="download_atl03",
                    node_label="download_atl03",
                )
                .add_args(*download_atl03_args)
                .add_outputs(atl03_data, stage_out=True, register_replica=False)
                .add_outputs(atl03_bbox, stage_out=False, register_replica=False)
            )
            if earthdata_token:
                download_atl03_job.add_env(EARTHDATA_TOKEN=earthdata_token)
            elif earthdata_username and earthdata_password:
                download_atl03_job.add_env(
                    EARTHDATA_USERNAME=earthdata_username,
                    EARTHDATA_PASSWORD=earthdata_password,
                )
            self.wf.add_jobs(download_atl03_job)

            # Job 2: Download Sentinel-2 (depends on ATL03 download for track bbox)
            download_sentinel2_args = [
                "--bbox-file", atl03_bbox,
                "--start-date", start_date,
                "--end-date", end_date,
                "--output", sentinel2_scenes,
            ]
            if max_scenes:
                download_sentinel2_args.extend(["--max-scenes", str(max_scenes)])

            download_sentinel2_job = (
                Job(
                    "download_sentinel2",
                    _id="download_sentinel2",
                    node_label="download_sentinel2",
                )
                .add_args(*download_sentinel2_args)
                .add_inputs(atl03_bbox)
                .add_outputs(sentinel2_scenes, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(download_sentinel2_job)

        # Job 3: Preprocess ATL03
        preprocess_job = (
            Job(
                "preprocess_atl03",
                _id="preprocess_atl03",
                node_label="preprocess_atl03",
            )
            .add_args(
                "--input", atl03_data,
                "--output", atl03_preprocessed,
            )
            .add_inputs(atl03_data)
            .add_outputs(atl03_preprocessed, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(preprocess_job)

        if test_mode:
            # In test mode, skip auto_label — labeled_data.csv is provided
            # via the replica catalog from generate_test_data.py
            logger.info("TEST MODE: Skipping auto_label job, using pre-built labeled_data.csv")
        else:
            # Job 4: Auto-label with Sentinel-2
            auto_label_job = (
                Job(
                    "auto_label",
                    _id="auto_label",
                    node_label="auto_label",
                )
                .add_args(
                    "--atl03-input", atl03_preprocessed,
                    "--sentinel2-input", sentinel2_scenes,
                    "--output", labeled_data,
                )
                .add_inputs(atl03_preprocessed, sentinel2_scenes)
                .add_outputs(labeled_data, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(auto_label_job)

        # Job 5: Train model
        train_job = (
            Job(
                "train_model",
                _id="train_model",
                node_label="train_model",
            )
            .add_args(
                "--input", labeled_data,
                "--model-output", model_file,
                "--metrics-output", training_metrics,
                "--model-type", model_type,
            )
            .add_inputs(labeled_data)
            .add_outputs(model_file, stage_out=True, register_replica=False)
            .add_outputs(training_metrics, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(train_job)

        # Job 6: Classify sea ice
        # Determine fan-out width: test_mode -> 2, max_granules set -> that value, else single job
        num_classify_jobs = None
        if test_mode:
            num_classify_jobs = 2
        elif max_granules is not None:
            num_classify_jobs = max_granules

        if num_classify_jobs is not None and num_classify_jobs > 1:
            # Fan-out: one classify job per granule
            per_granule_files = []
            for i in range(num_classify_jobs):
                granule_id_str = f"granule_{i:04d}"
                per_granule_file = File(f"classification_{granule_id_str}.csv")
                per_granule_files.append(per_granule_file)

                classify_job_i = (
                    Job(
                        "classify_seaice",
                        _id=f"classify_seaice_{i}",
                        node_label=f"classify_seaice_{i}",
                    )
                    .add_args(
                        "--input", atl03_preprocessed,
                        "--model", model_file,
                        "--granule", granule_id_str,
                        "--output", per_granule_file,
                    )
                    .add_inputs(atl03_preprocessed, model_file)
                    .add_outputs(per_granule_file, stage_out=False, register_replica=False)
                )
                self.wf.add_jobs(classify_job_i)

            # Fan-in: merge per-granule results
            merge_args = ["--output", classification_results, "--inputs"]
            merge_args.extend(per_granule_files)

            merge_job = (
                Job(
                    "merge_classifications",
                    _id="merge_classifications",
                    node_label="merge_classifications",
                )
                .add_args(*merge_args)
                .add_inputs(*per_granule_files)
                .add_outputs(classification_results, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(merge_job)
        else:
            # Single classify job (backward compatible)
            classify_job = (
                Job(
                    "classify_seaice",
                    _id="classify_seaice",
                    node_label="classify_seaice",
                )
                .add_args(
                    "--input", atl03_preprocessed,
                    "--model", model_file,
                    "--output", classification_results,
                )
                .add_inputs(atl03_preprocessed, model_file)
                .add_outputs(classification_results, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(classify_job)

        # Job 7: Calculate freeboard
        freeboard_job = (
            Job(
                "calculate_freeboard",
                _id="calculate_freeboard",
                node_label="calculate_freeboard",
            )
            .add_args(
                "--input", classification_results,
                "--output", freeboard_results,
            )
            .add_inputs(classification_results)
            .add_outputs(freeboard_results, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(freeboard_job)

        # Job 8: Visualize results
        visualize_job = (
            Job(
                "visualize_results",
                _id="visualize_results",
                node_label="visualize_results",
            )
            .add_args(
                "--classification-input", classification_results,
                "--freeboard-input", freeboard_results,
                "--classification-map-output", classification_map,
                "--freeboard-profile-output", freeboard_profile,
                "--summary-output", summary_stats,
            )
            .add_inputs(classification_results, freeboard_results)
            .add_outputs(classification_map, stage_out=True, register_replica=False)
            .add_outputs(freeboard_profile, stage_out=True, register_replica=False)
            .add_outputs(summary_stats, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(visualize_job)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pegasus workflow for sea ice classification and freeboard from ICESat-2 ATL03",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ross Sea, November 2019
  %(prog)s --region ross_sea --start-date 2019-11-01 --end-date 2019-11-30

  # Weddell Sea with specific granule
  %(prog)s --region weddell_sea --start-date 2019-11-01 --granule-id ATL03_20191101...

  # Use MLP classifier instead of LSTM
  %(prog)s --region ross_sea --start-date 2019-11-01 --model-type mlp

Available regions:
  - ross_sea: Ross Sea, Antarctica
  - weddell_sea: Weddell Sea, Antarctica
  - beaufort_sea: Beaufort Sea, Arctic
  - arctic_ocean: Full Arctic Ocean
  - southern_ocean: Full Southern Ocean
        """
    )

    parser.add_argument(
        "-s",
        "--skip-sites-catalog",
        action="store_true",
        help="Skip site catalog creation",
    )
    parser.add_argument(
        "-e",
        "--execution-site-name",
        metavar="STR",
        type=str,
        default="condorpool",
        help="Execution site name (default: condorpool)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="STR",
        type=str,
        default="workflow.yml",
        help="Output file (default: workflow.yml)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="ross_sea",
        help="Region name (default: ross_sea)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Required unless --test-mode is used."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to start_date + 30 days"
    )
    parser.add_argument(
        "--granule-id",
        type=str,
        default=None,
        help="Specific ATL03 granule ID (optional)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "mlp"],
        default="lstm",
        help="Classifier model type (default: lstm)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use synthetic test data instead of downloading from NASA/Planetary Computer. "
             "Skips download and auto-label jobs. Run 'python generate_test_data.py' first."
    )
    parser.add_argument(
        "--max-granules",
        type=int,
        default=None,
        help="Maximum number of ATL03 granules to download (limits download size)"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of Sentinel-2 scenes to download (limits download size)"
    )
    parser.add_argument(
        "--earthdata-token",
        type=str,
        default=os.environ.get("EARTHDATA_TOKEN"),
        help="NASA Earthdata bearer token (default: $EARTHDATA_TOKEN). "
             "Preferred on networks where urs.earthdata.nasa.gov is unreachable."
    )
    parser.add_argument(
        "--earthdata-username",
        type=str,
        default=os.environ.get("EARTHDATA_USERNAME"),
        help="NASA Earthdata username (default: $EARTHDATA_USERNAME env var)"
    )
    parser.add_argument(
        "--earthdata-password",
        type=str,
        default=os.environ.get("EARTHDATA_PASSWORD"),
        help="NASA Earthdata password (default: $EARTHDATA_PASSWORD env var)"
    )

    args = parser.parse_args()

    # In test mode, start-date is optional
    if not args.test_mode and not args.start_date:
        parser.error("--start-date is required unless --test-mode is used")
    if args.test_mode and not args.start_date:
        args.start_date = "2019-11-01"  # default for test mode

    # Handle default end date
    if not args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = start + timedelta(days=30)
        args.end_date = end.strftime("%Y-%m-%d")

    # Validate region
    valid_regions = ['ross_sea', 'weddell_sea', 'beaufort_sea', 'arctic_ocean',
                     'southern_ocean']
    if args.region not in valid_regions:
        logger.error(f"Invalid region: {args.region}. Valid regions: {valid_regions}")
        sys.exit(1)

    if not args.test_mode and not args.earthdata_token and (not args.earthdata_username or not args.earthdata_password):
        logger.warning(
            "No Earthdata credentials provided. Provide either:\n"
            "  --earthdata-token / $EARTHDATA_TOKEN (preferred for FABRIC), or\n"
            "  --earthdata-username + --earthdata-password / $EARTHDATA_USERNAME + $EARTHDATA_PASSWORD\n"
            "Register at: https://urs.earthdata.nasa.gov/"
        )

    logger.info("=" * 70)
    logger.info("SEA ICE WORKFLOW GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Region: {args.region}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Model type: {args.model_type}")
    if args.granule_id:
        logger.info(f"Granule ID: {args.granule_id}")
    if args.test_mode:
        logger.info("Mode: TEST (synthetic data, no downloads)")
    else:
        logger.info(f"Earthdata auth: {'token' if args.earthdata_token else 'username/password' if args.earthdata_username else 'NOT SET'}")
        if args.max_granules:
            logger.info(f"Max ATL03 granules: {args.max_granules}")
        if args.max_scenes:
            logger.info(f"Max Sentinel-2 scenes: {args.max_scenes}")
    logger.info(f"Execution site: {args.execution_site_name}")
    logger.info(f"Output file: {args.output}")
    logger.info("=" * 70)

    try:
        workflow = SeaIceWorkflow(dagfile=args.output)

        if not args.skip_sites_catalog:
            logger.info("Creating execution sites...")
            workflow.create_sites_catalog(args.execution_site_name)

        logger.info("Creating workflow properties...")
        workflow.create_pegasus_properties()

        logger.info("Creating transformation catalog...")
        workflow.create_transformation_catalog(args.execution_site_name)

        logger.info("Creating replica catalog...")
        workflow.create_replica_catalog(test_mode=args.test_mode)

        logger.info("Creating sea ice workflow DAG...")
        workflow.create_workflow(
            region=args.region,
            start_date=args.start_date,
            end_date=args.end_date,
            granule_id=args.granule_id,
            model_type=args.model_type,
            earthdata_token=args.earthdata_token,
            earthdata_username=args.earthdata_username,
            earthdata_password=args.earthdata_password,
            test_mode=args.test_mode,
            max_granules=args.max_granules,
            max_scenes=args.max_scenes,
        )

        workflow.write()

        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info(f"  1. Review workflow: {args.output}")
        logger.info(f"  2. Submit workflow: pegasus-plan --submit -s {args.execution_site_name} -o local {args.output}")
        logger.info(f"  3. Monitor status: pegasus-status <submit_dir>")
        logger.info("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
