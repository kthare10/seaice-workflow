#!/bin/bash

# Example usage of sea ice workflow

echo "========================================"
echo "SEA ICE WORKFLOW EXAMPLE USAGE"
echo "========================================"

# Build Docker container (optional, if using locally)
echo ""
echo "1. Build Docker container:"
echo "   cd Docker"
echo "   docker build -t kthare10/seaice-icesat2:latest ."
echo "   docker push kthare10/seaice-icesat2:latest"

# Example 1: Ross Sea, November 2019
echo ""
echo "2. Example 1: Ross Sea, November 2019 (LSTM model)"
echo "   python workflow_generator.py \\"
echo "       --region ross_sea \\"
echo "       --start-date 2019-11-01 \\"
echo "       --end-date 2019-11-30 \\"
echo "       --output workflow.yml"

# Example 2: Weddell Sea with specific granule
echo ""
echo "3. Example 2: Weddell Sea with specific granule"
echo "   python workflow_generator.py \\"
echo "       --region weddell_sea \\"
echo "       --start-date 2019-11-01 \\"
echo "       --granule-id ATL03_20191101... \\"
echo "       --output workflow_weddell.yml"

# Example 3: MLP classifier
echo ""
echo "4. Example 3: Use MLP classifier instead of LSTM"
echo "   python workflow_generator.py \\"
echo "       --region ross_sea \\"
echo "       --start-date 2019-11-01 \\"
echo "       --model-type mlp \\"
echo "       --output workflow_mlp.yml"

# Example 4: Arctic region
echo ""
echo "5. Example 4: Beaufort Sea, Arctic"
echo "   python workflow_generator.py \\"
echo "       --region beaufort_sea \\"
echo "       --start-date 2020-03-01 \\"
echo "       --end-date 2020-03-31 \\"
echo "       --output workflow_beaufort.yml"

# Example 5: Custom execution site
echo ""
echo "6. Example 5: Custom execution site"
echo "   python workflow_generator.py \\"
echo "       --region ross_sea \\"
echo "       --start-date 2019-11-01 \\"
echo "       -e condorpool \\"
echo "       --output workflow.yml"

# Example 6: Test mode (no downloads, uses synthetic data)
echo ""
echo "6. Test mode (no downloads, uses synthetic test data):"
echo "   python generate_test_data.py"
echo "   python workflow_generator.py --test-mode --output workflow_test.yml"

# Example 7: Limited download (fewer granules/scenes for faster testing)
echo ""
echo "7. Limited download mode:"
echo "   python workflow_generator.py \\"
echo "       --region ross_sea \\"
echo "       --start-date 2019-11-01 \\"
echo "       --end-date 2019-11-07 \\"
echo "       --max-granules 2 \\"
echo "       --max-scenes 3 \\"
echo "       --output workflow_limited.yml"

# Run functional test
echo ""
echo "8. Run functional test on synthetic data:"
echo "   bash run_test.sh"

# Submit workflow
echo ""
echo "9. Submit workflow to Pegasus:"
echo "   pegasus-plan --submit -s condorpool -o local workflow.yml"

# Monitor workflow
echo ""
echo "10. Monitor workflow status:"
echo "   pegasus-status /path/to/submit/directory"
echo "   pegasus-analyzer /path/to/submit/directory"

# Check outputs
echo ""
echo "11. Output files:"
echo "    atl03_data.h5                  - Downloaded ATL03 photon data"
echo "    sentinel2_scenes.tar.gz        - Downloaded Sentinel-2 imagery"
echo "    atl03_preprocessed.csv         - Preprocessed 2m segments"
echo "    labeled_data.csv               - Auto-labeled training data"
echo "    model.h5                       - Trained classifier model"
echo "    training_metrics.json          - Training performance metrics"
echo "    classification_results.csv     - Sea ice type predictions"
echo "    freeboard_results.csv          - Freeboard calculations"
echo "    classification_map.png         - Geographic classification map"
echo "    freeboard_profile.png          - Along-track freeboard profile"
echo "    summary_statistics.json        - Summary statistics"

echo ""
echo "========================================"
echo "Available regions:"
echo "  - ross_sea: Ross Sea, Antarctica"
echo "  - weddell_sea: Weddell Sea, Antarctica"
echo "  - beaufort_sea: Beaufort Sea, Arctic"
echo "  - arctic_ocean: Full Arctic Ocean"
echo "  - southern_ocean: Full Southern Ocean"
echo "========================================"
echo ""
echo "Model types: lstm (default), mlp"
echo "Data mode: condorio (HTCondor-managed file transfers)"
echo "========================================"
