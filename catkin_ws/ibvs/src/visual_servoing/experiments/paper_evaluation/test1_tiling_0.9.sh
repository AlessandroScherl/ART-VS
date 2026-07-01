#!/bin/bash

# TEST1 - TILING 0.9: Hollywood 500 evaluation with 0.9 tiling switch threshold
# - Feature stagnation: 10 iterations at 0.5px
# - Tiling switch: 0.9 (10% of initial error)
# - Lambda: 0.3 for DINOv2-308 and DINOv2-518

set -e  # Exit on error

# Parse command line arguments
SAMPLES=500  # Default to 500 samples
if [[ "$1" == "--samples" ]]; then
    SAMPLES=$2
fi

echo "============================================================="
echo "TEST1 - TILING 0.9: Hollywood 500 Evaluation - 0.9 Switch"
echo "============================================================="
echo "Samples per method: $SAMPLES"
echo "Feature stagnation: 10 iterations at 0.5px"
echo "Tiling switch: 0.9 (10% of initial error)"
echo ""

# Get the absolute path of the paper_evaluation directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create timestamped result directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${SCRIPT_DIR}/results/test1_tiling_0.9/${TIMESTAMP}"
mkdir -p ${RESULT_DIR}

# Save experiment metadata
cat > ${RESULT_DIR}/experiment_info.txt << EOF
TEST1 - TILING 0.9: Hollywood 500 - 0.9 Tiling Switch
Date: $(date)
Samples: $SAMPLES
Model: 99 (Hollywood poster)
Methods: SIFT, ORB, AKAZE, DINOv2-308, DINOv2-518, DINOv2-224-7x7, DINOv3-256-6x6, DINOv3-1440, AM-RADIO-256-6x6
Feature Stagnation: Enabled (10 iterations at 0.5px)
Tiling Switch: 0.9 (10% of initial error)
Lambda: 0.3 for DINOv2-308/518, 0.1 for others
EOF

# Set up Python path to find modules
export PYTHONPATH="${SCRIPT_DIR}/../..:$PYTHONPATH"

# Change to experiments directory for correct relative paths
cd ${SCRIPT_DIR}/..

echo "Results will be saved to: ${RESULT_DIR}"
echo ""

# Track timing
START_TIME=$(date +%s)

# Method 1: DINOv2-308 (no tiling - MODE 1)
echo ""
echo "=================================================="
echo "[1/9] Running DINOv2-308 (no tiling, lambda=0.3)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_308_no_tiling
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_dinov2_308_no_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_308_no_tiling/results_dinov2_308_model99.npz \
    --iteration-display-freq 50

# Method 2: DINOv2-518 (no tiling - MODE 1)
echo ""
echo "=================================================="
echo "[2/9] Running DINOv2-518 (no tiling, lambda=0.3)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_518_no_tiling
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_dinov2_518_no_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_518_no_tiling/results_dinov2_518_model99.npz \
    --iteration-display-freq 50

# Method 3: DINOv2-224 (7x7 tiling, 0.9 switch)
echo ""
echo "=================================================="
echo "[3/9] Running DINOv2-224 (7x7 tiling, 0.9 switch)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_224_7x7
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_dinov2_224_7x7_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_224_7x7/results_dinov2_224_7x7_model99.npz \
    --iteration-display-freq 50

# Method 4: DINOv3-256 (6x6 tiling, 0.9 switch)
echo ""
echo "=================================================="
echo "[4/9] Running DINOv3-256 (6x6 tiling, 0.9 switch)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov3_256_6x6
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_dinov3_256_6x6_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov3_256_6x6/results_dinov3_256_6x6_model99.npz \
    --iteration-display-freq 50

# Method 5: DINOv3-1440 (no tiling - MODE 1, max_iter=500)
echo ""
echo "=================================================="
echo "[5/9] Running DINOv3-1440 (no tiling, max_iter=500)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov3_1440_no_tiling
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_dinov3_1440_no_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov3_1440_no_tiling/results_dinov3_1440_model99.npz \
    --iteration-display-freq 50

# Method 6: AM-RADIO-256 (6x6 tiling, 0.9 switch)
echo ""
echo "=================================================="
echo "[6/9] Running AM-RADIO-256 (6x6 tiling, 0.9 switch)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/amradio_256_6x6
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_amradio_256_6x6_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/amradio_256_6x6/results_amradio_256_6x6_model99.npz \
    --iteration-display-freq 50

# Method 7: SIFT
echo ""
echo "=================================================="
echo "[7/9] Running SIFT..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/sift
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_sift_hollywood_500.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/sift/results_sift_model99.npz \
    --iteration-display-freq 50

# Method 8: ORB
echo ""
echo "=================================================="
echo "[8/9] Running ORB..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/orb
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_orb_hollywood_500.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/orb/results_orb_model99.npz \
    --iteration-display-freq 50

# Method 9: AKAZE
echo ""
echo "=================================================="
echo "[9/9] Running AKAZE..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/akaze
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../../configs/test1_tiling_0.9/config_akaze_hollywood_500.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/akaze/results_akaze_model99.npz \
    --iteration-display-freq 50

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=================================================="
echo "All methods completed!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=================================================="

# Run unified evaluation
echo "Running unified evaluation..."
cd ${SCRIPT_DIR}
python3 ../unified_eval.py ${RESULT_DIR}/*/results_*.npz --table > ${RESULT_DIR}/evaluation_summary.txt

# Display summary
echo ""
echo "Evaluation Summary:"
echo "-------------------"
tail -20 ${RESULT_DIR}/evaluation_summary.txt

echo ""
echo "=================================================="
echo "Results saved to: ${RESULT_DIR}"
echo "=================================================="
echo ""
echo "To analyze results:"
echo "  cd experiments/paper_evaluation"
echo "  python3 ../unified_eval.py ${RESULT_DIR}/*/results_*.npz --table"
echo ""