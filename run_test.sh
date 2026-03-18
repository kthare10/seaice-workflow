#!/bin/bash

# Functional test for the sea ice workflow pipeline.
#
# Runs pipeline steps 2-8 on synthetic test data, skipping steps that
# require live APIs (download_atl03, download_sentinel2, auto_label).
#
# Usage:
#     bash run_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DATA_DIR="${SCRIPT_DIR}/test_data"
TEST_OUTPUT_DIR="${SCRIPT_DIR}/test_output"
BIN_DIR="${SCRIPT_DIR}/bin"

# Use venv python if available, otherwise system python3
if [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi
echo "Using Python: ${PYTHON}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

pass() {
    echo -e "  ${GREEN}PASS${NC}: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo -e "  ${RED}FAIL${NC}: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

echo "========================================"
echo "SEA ICE WORKFLOW FUNCTIONAL TEST"
echo "========================================"
echo ""

# Clean up previous test output
rm -rf "${TEST_OUTPUT_DIR}"
mkdir -p "${TEST_OUTPUT_DIR}"

# ----------------------------------------
# Step 1: Generate test data
# ----------------------------------------
echo "Step 1: Generate synthetic test data"
${PYTHON} "${SCRIPT_DIR}/generate_test_data.py"

if [ -f "${TEST_DATA_DIR}/atl03_data.h5" ]; then
    pass "atl03_data.h5 created"
else
    fail "atl03_data.h5 not created"
fi

if [ -f "${TEST_DATA_DIR}/atl03_preprocessed.csv" ]; then
    pass "atl03_preprocessed.csv created"
else
    fail "atl03_preprocessed.csv not created"
fi

if [ -f "${TEST_DATA_DIR}/labeled_data.csv" ]; then
    pass "labeled_data.csv created"
else
    fail "labeled_data.csv not created"
fi
echo ""

# ----------------------------------------
# Step 2: Preprocess ATL03 (from HDF5)
# ----------------------------------------
echo "Step 2: Preprocess ATL03 data"
${PYTHON} "${BIN_DIR}/preprocess_atl03.py" \
    --input "${TEST_DATA_DIR}/atl03_data.h5" \
    --output "${TEST_OUTPUT_DIR}/atl03_preprocessed.csv"

if [ -f "${TEST_OUTPUT_DIR}/atl03_preprocessed.csv" ]; then
    LINE_COUNT=$(wc -l < "${TEST_OUTPUT_DIR}/atl03_preprocessed.csv" | tr -d ' ')
    if [ "$LINE_COUNT" -gt 10 ]; then
        pass "Preprocessed CSV has ${LINE_COUNT} lines"
    else
        fail "Preprocessed CSV too small (${LINE_COUNT} lines)"
    fi
else
    fail "Preprocessed CSV not created"
fi
echo ""

# ----------------------------------------
# Step 3: Train model (MLP, 5 epochs)
# ----------------------------------------
echo "Step 3: Train MLP model (5 epochs)"
${PYTHON} "${BIN_DIR}/train_model.py" \
    --input "${TEST_DATA_DIR}/labeled_data.csv" \
    --model-output "${TEST_OUTPUT_DIR}/model.h5" \
    --metrics-output "${TEST_OUTPUT_DIR}/training_metrics.json" \
    --model-type mlp \
    --epochs 5

if [ -f "${TEST_OUTPUT_DIR}/model.h5" ]; then
    pass "Model file created"
else
    fail "Model file not created"
fi

if [ -f "${TEST_OUTPUT_DIR}/training_metrics.json" ]; then
    pass "Training metrics created"
else
    fail "Training metrics not created"
fi
echo ""

# ----------------------------------------
# Step 4: Classify sea ice
# ----------------------------------------
echo "Step 4: Classify sea ice types"
${PYTHON} "${BIN_DIR}/classify_seaice.py" \
    --input "${TEST_DATA_DIR}/atl03_preprocessed.csv" \
    --model "${TEST_OUTPUT_DIR}/model.h5" \
    --output "${TEST_OUTPUT_DIR}/classification_results.csv"

if [ -f "${TEST_OUTPUT_DIR}/classification_results.csv" ]; then
    # Check that predicted_class column exists
    if head -1 "${TEST_OUTPUT_DIR}/classification_results.csv" | grep -q "predicted_class"; then
        pass "Classification results with predicted_class column"
    else
        fail "Classification results missing predicted_class column"
    fi
else
    fail "Classification results not created"
fi
echo ""

# ----------------------------------------
# Step 5: Calculate freeboard
# ----------------------------------------
echo "Step 5: Calculate freeboard"
${PYTHON} "${BIN_DIR}/calculate_freeboard.py" \
    --input "${TEST_OUTPUT_DIR}/classification_results.csv" \
    --output "${TEST_OUTPUT_DIR}/freeboard_results.csv"

if [ -f "${TEST_OUTPUT_DIR}/freeboard_results.csv" ]; then
    if head -1 "${TEST_OUTPUT_DIR}/freeboard_results.csv" | grep -q "freeboard"; then
        pass "Freeboard results with freeboard column"
    else
        fail "Freeboard results missing freeboard column"
    fi
else
    fail "Freeboard results not created"
fi
echo ""

# ----------------------------------------
# Step 6: Visualize results
# ----------------------------------------
echo "Step 6: Visualize results"
${PYTHON} "${BIN_DIR}/visualize_results.py" \
    --classification-input "${TEST_OUTPUT_DIR}/classification_results.csv" \
    --freeboard-input "${TEST_OUTPUT_DIR}/freeboard_results.csv" \
    --classification-map-output "${TEST_OUTPUT_DIR}/classification_map.png" \
    --freeboard-profile-output "${TEST_OUTPUT_DIR}/freeboard_profile.png" \
    --summary-output "${TEST_OUTPUT_DIR}/summary_statistics.json"

if [ -f "${TEST_OUTPUT_DIR}/classification_map.png" ]; then
    pass "Classification map PNG created"
else
    fail "Classification map PNG not created"
fi

if [ -f "${TEST_OUTPUT_DIR}/freeboard_profile.png" ]; then
    pass "Freeboard profile PNG created"
else
    fail "Freeboard profile PNG not created"
fi

if [ -f "${TEST_OUTPUT_DIR}/summary_statistics.json" ]; then
    pass "Summary statistics JSON created"
else
    fail "Summary statistics JSON not created"
fi
echo ""

# ----------------------------------------
# Summary
# ----------------------------------------
echo "========================================"
TOTAL=$((PASS_COUNT + FAIL_COUNT))
echo "Results: ${PASS_COUNT}/${TOTAL} checks passed"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}OVERALL: PASS${NC}"
    echo "========================================"
    exit 0
else
    echo -e "${RED}OVERALL: FAIL (${FAIL_COUNT} failures)${NC}"
    echo "========================================"
    exit 1
fi
