#!/bin/bash

# TEST1: Hollywood 500 evaluation WITH perturbation
# 10 methods: SIFT, ORB, AKAZE, DINOv2-308, DINOv2-518, DINOv2-224-7x7, DINOv2-224-6x6, DINOv3-256-6x6, DINOv3-1440, AM-RADIO-256-6x6

set -e  # Exit on error

# Parse command line arguments
SAMPLES=500  # Default to 500 samples
if [[ "$1" == "--samples" ]]; then
    SAMPLES=$2
fi

echo "=================================================="
echo "TEST1: Hollywood 500 Evaluation - WITH Perturbation"
echo "=================================================="
echo "Samples per method: $SAMPLES"
echo ""

# Get the absolute path of the paper_evaluation directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create timestamped result directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${SCRIPT_DIR}/results/test1_with_perturbation/${TIMESTAMP}"
mkdir -p ${RESULT_DIR}

# Save experiment metadata
cat > ${RESULT_DIR}/experiment_info.txt << EOF
TEST1: Hollywood 500 - WITH Perturbation
Date: $(date)
Samples: $SAMPLES
Model: 99 (Hollywood poster)
Methods: SIFT, ORB, AKAZE, DINOv2-308, DINOv2-518, DINOv2-224-7x7, DINOv2-224-6x6, DINOv3-256-6x6, DINOv3-1440, AM-RADIO-256-6x6
Perturbation: Yes (Random erasing, color jitter, Gaussian noise)
EOF

# Set up Python path to find modules
export PYTHONPATH="${SCRIPT_DIR}/../..:$PYTHONPATH"

# Change to experiments directory for correct relative paths
cd ${SCRIPT_DIR}/..

echo "Results will be saved to: ${RESULT_DIR}"
echo ""
echo "PERTURBATION ENABLED:"
echo "  - Random erasing (50% probability)"
echo "  - Color jittering (±60% brightness, ±40% contrast)"
echo "  - Gaussian noise (σ=0.05)"
echo ""

# Track timing
START_TIME=$(date +%s)

# Method 1: DINOv2-308 (no tiling - MODE 1)
echo ""
echo "=================================================="
echo "[1/9] Running DINOv2-308 (no tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_308_no_tiling
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_dinov2_308_no_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_308_no_tiling/results_dinov2_308_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 2: DINOv2-518 (no tiling - MODE 1)
echo ""
echo "=================================================="
echo "[2/9] Running DINOv2-518 (no tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_518_no_tiling
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_dinov2_518_no_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_518_no_tiling/results_dinov2_518_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 3: DINOv2-224 (7x7 tiling)
echo ""
echo "=================================================="
echo "[3/10] Running DINOv2-224 (7x7 tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_224_7x7
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_dinov2_224_7x7_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_224_7x7/results_dinov2_224_7x7_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 4: DINOv2-224 (6x6 tiling)
echo ""
echo "=================================================="
echo "[4/10] Running DINOv2-224 (6x6 tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov2_224_6x6
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_dinov2_224_6x6_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov2_224_6x6/results_dinov2_224_6x6_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 5: DINOv3-256 (6x6 tiling)
echo ""
echo "=================================================="
echo "[5/10] Running DINOv3-256 (6x6 tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov3_256_6x6
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_dinov3_256_6x6_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov3_256_6x6/results_dinov3_256_6x6_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 6: DINOv3-1440 (no tiling - MODE 1)
echo ""
echo "=================================================="
echo "[6/10] Running DINOv3-1440 (no tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/dinov3_1440_no_tiling
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_dinov3_1440_no_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/dinov3_1440_no_tiling/results_dinov3_1440_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 7: AM-RADIO-256 (6x6 tiling)
echo ""
echo "=================================================="
echo "[7/10] Running AM-RADIO-256 (6x6 tiling, with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/amradio_256_6x6
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_amradio_256_6x6_tiling.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/amradio_256_6x6/results_amradio_256_6x6_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation
    
# Method 8: SIFT
echo ""
echo "=================================================="
echo "[8/10] Running SIFT (with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/sift
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_sift_hollywood_500.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/sift/results_sift_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 9: ORB
echo ""
echo "=================================================="
echo "[9/10] Running ORB (with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/orb
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_orb_hollywood_500.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/orb/results_orb_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Method 10: AKAZE
echo ""
echo "=================================================="
echo "[10/10] Running AKAZE (with perturbation)..."
echo "=================================================="
mkdir -p ${RESULT_DIR}/akaze
python3 run_multi_model_experiment_with_hollywood.py \
    --config ../configs/hollywood_500_evaluation/config_akaze_hollywood_500.yaml \
    --models 99 \
    --samples ${SAMPLES} \
    --output ${RESULT_DIR}/akaze/results_akaze_model99_perturbed.npz \
    --iteration-display-freq 50 \
    --perturbation

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=================================================="
echo "All experiments complete!"
echo "=================================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Run unified evaluation
echo "Running unified evaluation..."
cd ${SCRIPT_DIR}
python3 ../unified_eval.py ${RESULT_DIR}/*/results_*.npz --table > ${RESULT_DIR}/evaluation_summary.txt

# Display summary
echo ""
echo "Evaluation Summary (With Perturbation):"
echo "----------------------------------------"
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
echo "To compare with non-perturbed results:"
echo "  python3 ../unified_eval.py results/test1_no_perturbation/*/results_*.npz results/test1_with_perturbation/*/results_*.npz --compare"