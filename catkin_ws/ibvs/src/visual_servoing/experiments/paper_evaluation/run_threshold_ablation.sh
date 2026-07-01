#!/bin/bash

# ==============================================================================
# THRESHOLD ABLATION STUDY: DINOv3-256-6x6 on Hollywood with Perturbation
# ==============================================================================
# Tests 5 different hybrid_switch_threshold values to find optimal coarse-to-fine
# switching point for the ART-VS method.
#
# Thresholds tested:
#   0.40 (60% error reduction) - Early tiling activation
#   0.30 (70% error reduction)
#   0.20 (80% error reduction) - BASELINE (current default)
#   0.10 (90% error reduction)
#   0.05 (95% error reduction) - Late tiling activation
#
# Usage:
#   ./run_threshold_ablation.sh                  # Full 500 samples
#   ./run_threshold_ablation.sh --samples 10    # Quick test with 10 samples
#
# Output:
#   results/threshold_ablation/YYYYMMDD_HHMMSS/
#     - threshold_0.XX/results_*.npz for each threshold
#     - experiment_info.txt
#     - evaluation_summary.txt
# ==============================================================================

set -e  # Exit on error

# Get script directory (absolute path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default configuration
SAMPLES=500  # Default to 500 samples (same as TEST1)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--samples N]"
            echo ""
            echo "Options:"
            echo "  --samples N    Number of samples per threshold (default: 500)"
            echo ""
            echo "Example:"
            echo "  $0 --samples 10    # Quick test"
            echo "  $0 --samples 500   # Full evaluation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================================================="
echo "THRESHOLD ABLATION STUDY: DINOv3-256-6x6 on Hollywood (Perturbed)"
echo "==================================================================${NC}"
echo ""
echo "Date: $(date)"
echo "Samples per threshold: ${SAMPLES}"
echo "Model: 99 (Hollywood poster)"
echo -e "Perturbation: ${YELLOW}ENABLED${NC}"
echo ""
echo "Thresholds to test:"
echo -e "  ${CYAN}0.40${NC} - Tiling at 60% error reduction (early)"
echo -e "  ${CYAN}0.30${NC} - Tiling at 70% error reduction"
echo -e "  ${CYAN}0.20${NC} - Tiling at 80% error reduction ${YELLOW}(BASELINE)${NC}"
echo -e "  ${CYAN}0.10${NC} - Tiling at 90% error reduction"
echo -e "  ${CYAN}0.05${NC} - Tiling at 95% error reduction (late)"
echo ""

# Create timestamped result directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${SCRIPT_DIR}/results/threshold_ablation/${TIMESTAMP}"
mkdir -p ${RESULT_DIR}

# Save experiment metadata
cat > ${RESULT_DIR}/experiment_info.txt << EOF
THRESHOLD ABLATION STUDY
========================
Date: $(date)
Samples per threshold: ${SAMPLES}
Model: 99 (Hollywood poster)
Method: DINOv3-256-6x6 (ART-VS)
Perturbation: Enabled (Random erasing, color jitter, Gaussian noise)

Thresholds Tested:
------------------
  0.40: Tiling activates at 60% error reduction (early)
  0.30: Tiling activates at 70% error reduction
  0.20: Tiling activates at 80% error reduction (BASELINE)
  0.10: Tiling activates at 90% error reduction
  0.05: Tiling activates at 95% error reduction (late)

Convergence Criteria (same as TEST1):
-------------------------------------
  - Position error < 10% of initial (90% reduction)
  - Rotation error < 10% of initial (90% reduction)
  - Divergence: abort if error > 3x initial
  - Max iterations: 1500

Expected Trade-offs:
--------------------
  Early switching (0.40): More Phase2 iterations, potentially better precision, slower
  Late switching (0.05): More Phase1 iterations, faster, potentially less precise

Key Metrics:
------------
  - Convergence rate
  - Final position/orientation errors
  - tiling_switch_iterations (Phase1/Phase2 boundary)
  - pre_tiling_fps / post_tiling_fps
  - Total iterations
EOF

echo "Results will be saved to: ${RESULT_DIR}"
echo ""

# Check if simulation is running
echo -e "${YELLOW}Checking for running Gazebo simulation...${NC}"
if pgrep -x "gzserver" > /dev/null; then
    echo -e "${GREEN}✓ Gazebo simulation is running${NC}"
else
    echo -e "${RED}✗ WARNING: Gazebo simulation not detected!${NC}"
    echo "Please start simulation in another terminal:"
    echo "  cd catkin_ws/ibvs/src"
    echo "  ./run_ibvs.sh"
    echo ""
    read -p "Press Enter when simulation is ready, or Ctrl+C to cancel..."
fi

# Navigate to experiments directory (relative to script location)
cd ${SCRIPT_DIR}/..
echo "Working directory: $(pwd)"
echo ""

# Track timing
START_TIME=$(date +%s)

# Define thresholds array
declare -a THRESHOLDS=("0.40" "0.30" "0.20" "0.10" "0.05")
declare -a REDUCTIONS=("60%" "70%" "80% (BASELINE)" "90%" "95%")

# Total number of thresholds
TOTAL=${#THRESHOLDS[@]}

echo -e "${GREEN}Starting threshold ablation study with ${TOTAL} thresholds...${NC}"
echo ""

# Run each threshold
for i in "${!THRESHOLDS[@]}"; do
    THRESHOLD=${THRESHOLDS[$i]}
    REDUCTION=${REDUCTIONS[$i]}
    NUM=$((i + 1))

    echo ""
    echo -e "${MAGENTA}=================================================================="
    echo "[${NUM}/${TOTAL}] Running DINOv3-256-6x6 with threshold ${THRESHOLD} (${REDUCTION} reduction)"
    echo -e "==================================================================${NC}"

    mkdir -p ${RESULT_DIR}/threshold_${THRESHOLD}

    # Run experiment
    python3 run_multi_model_experiment_with_hollywood.py \
        --config ../configs/threshold_ablation/config_dinov3_256_6x6_threshold_${THRESHOLD}.yaml \
        --models 99 \
        --samples ${SAMPLES} \
        --output ${RESULT_DIR}/threshold_${THRESHOLD}/results_dinov3_256_threshold_${THRESHOLD}_model99_perturbed.npz \
        --iteration-display-freq 50 \
        --perturbation \
        || echo -e "${RED}✗ Threshold ${THRESHOLD} failed${NC}"

    echo -e "${GREEN}✓ Threshold ${THRESHOLD} completed${NC}"
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo -e "${GREEN}=================================================================="
echo "All thresholds completed!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "==================================================================${NC}"

# List generated files
echo ""
echo "Generated result files:"
echo "-----------------------"
ls -la ${RESULT_DIR}/*/results_*.npz 2>/dev/null || echo "No result files found"
echo ""

# Run unified evaluation
echo -e "${YELLOW}Running unified evaluation...${NC}"
cd ${SCRIPT_DIR}

# Check if unified_eval.py exists
if [ -f "../unified_eval.py" ]; then
    python3 ../unified_eval.py ${RESULT_DIR}/*/results_*.npz --table > ${RESULT_DIR}/evaluation_summary.txt 2>&1 || true

    echo ""
    echo -e "${GREEN}Evaluation Summary:${NC}"
    echo "-------------------"
    cat ${RESULT_DIR}/evaluation_summary.txt
else
    echo -e "${YELLOW}unified_eval.py not found, skipping automatic evaluation${NC}"
fi

# Create summary file
cat >> ${RESULT_DIR}/experiment_info.txt << EOF

================================================================================
EXECUTION COMPLETE
================================================================================
End time: $(date)
Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s

Result files:
$(ls -1 ${RESULT_DIR}/*/results_*.npz 2>/dev/null || echo "No result files generated")

Analysis commands:
------------------
# View evaluation table
python3 unified_eval.py ${RESULT_DIR}/*/results_*.npz --table

# Compare with TEST1 baseline (if available)
python3 unified_eval.py ${RESULT_DIR}/*/results_*.npz results/test1_with_perturbation/*/dinov3_256_6x6/*.npz --compare

# Extract specific metrics
python3 -c "
import numpy as np
import glob
for f in sorted(glob.glob('${RESULT_DIR}/*/results_*.npz')):
    data = np.load(f)
    name = f.split('/')[-2]
    conv = data['convergence_flags'].mean() * 100
    switch = data['tiling_switch_iterations']
    switch_mean = switch[switch > 0].mean() if (switch > 0).any() else -1
    print(f'{name}: Conv={conv:.1f}%, Switch@iter={switch_mean:.0f}')
"
EOF

echo ""
echo -e "${GREEN}=================================================================="
echo "Results saved to: ${RESULT_DIR}"
echo -e "==================================================================${NC}"
echo ""
echo "Next steps:"
echo "-----------"
echo "1. View evaluation table:"
echo "   python3 unified_eval.py ${RESULT_DIR}/*/results_*.npz --table"
echo ""
echo "2. Key questions to answer:"
echo "   - Which threshold achieves highest convergence rate?"
echo "   - How does Phase1/Phase2 split affect final precision?"
echo "   - Is 80% (0.20) truly optimal?"
echo ""

echo -e "${GREEN}✓ Threshold ablation study completed!${NC}"

exit 0
