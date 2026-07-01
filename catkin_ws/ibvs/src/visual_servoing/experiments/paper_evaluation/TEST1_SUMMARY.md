# TEST1 Parameter Study - Summary

This directory contains 4 test variations to study the effect of feature stagnation detection and tiling switch thresholds on Hollywood 500-sample evaluation.

## Test Scripts

### 1. test1_current.sh
- **Purpose**: Baseline with current settings
- **Feature Stagnation**: Enabled (10 iterations at 0.5px)
- **Tiling Switch**: 0.8 (switches to tiling when error reaches 20% of initial)
- **Lambda**: 0.3 for DINOv2-308/518, 0.1 for others
- **Config Directory**: Uses existing `hollywood_500_evaluation/` configs

### 2. test1_no_early_stop.sh
- **Purpose**: Evaluate impact of feature stagnation detection
- **Feature Stagnation**: DISABLED (runs full 1500 iterations, 500 for DINOv3-1440)
- **Tiling Switch**: 0.8 (switches to tiling when error reaches 20% of initial)
- **Lambda**: 0.3 for DINOv2-308/518, 0.1 for others
- **Config Directory**: `test1_no_early_stop/`
- **Expected Outcome**: Longer runtimes but potentially more consistent convergence metrics

### 3. test1_tiling_0.9.sh
- **Purpose**: Evaluate later tiling activation
- **Feature Stagnation**: Enabled (10 iterations at 0.5px)
- **Tiling Switch**: 0.9 (switches to tiling when error reaches 10% of initial - LATER activation)
- **Lambda**: 0.3 for DINOv2-308/518, 0.1 for others
- **Config Directory**: `test1_tiling_0.9/`
- **Expected Outcome**: Tiling activates later, may affect convergence quality/speed

### 4. test1_stagnation_20.sh
- **Purpose**: Evaluate longer stagnation window
- **Feature Stagnation**: Enabled (20 iterations at 0.5px - DOUBLED window)
- **Tiling Switch**: 0.8 (switches to tiling when error reaches 20% of initial)
- **Lambda**: 0.3 for DINOv2-308/518, 0.1 for others
- **Config Directory**: `test1_stagnation_20/`
- **Expected Outcome**: More conservative early stopping, potentially higher accuracy

## Methods Tested (9 total, skipping DINOv2-224 6x6 as requested)

1. **DINOv2-308** (no tiling, lambda=0.3)
2. **DINOv2-518** (no tiling, lambda=0.3)
3. **DINOv2-224** (7x7 tiling)
4. **DINOv3-256** (6x6 tiling)
5. **DINOv3-1440** (no tiling, max_iter=500)
6. **AM-RADIO-256** (6x6 tiling)
7. **SIFT** (classical)
8. **ORB** (classical)
9. **AKAZE** (classical)

## Running the Tests

```bash
# Navigate to the directory
cd experiments/paper_evaluation

# Run tests sequentially (each takes ~4-6 hours with 500 samples)
./test1_current.sh --samples 500
./test1_no_early_stop.sh --samples 500
./test1_tiling_0.9.sh --samples 500
./test1_stagnation_20.sh --samples 500

# Or run with fewer samples for testing
./test1_current.sh --samples 10
```

## Results Structure

Each test creates its own timestamped directory:
- `results/test1_current/<timestamp>/`
- `results/test1_no_early_stop/<timestamp>/`
- `results/test1_tiling_0.9/<timestamp>/`
- `results/test1_stagnation_20/<timestamp>/`

Each contains:
- Individual method results in subdirectories
- `evaluation_summary.txt` with convergence statistics
- `experiment_info.txt` with test metadata

## Key Parameters Explained

### Feature Stagnation Detection
- **Purpose**: Early stopping when feature error stays near zero
- **Threshold**: 0.5 pixels (average feature error)
- **Window**: Number of consecutive iterations below threshold
- **Effect**: Saves computation when features are perfectly aligned

### Tiling Switch Threshold
- **Purpose**: When to switch from low-res to high-res tiling mode
- **Value**: Fraction of initial error remaining
- **0.8 = 20% of initial**: Switch when 80% error reduction achieved
- **0.9 = 10% of initial**: Switch when 90% error reduction achieved
- **Effect**: Later switching may improve stability but increase iterations

### Lambda Values
- **DINOv2-308/518**: 0.3 (as per your previous paper)
- **All others**: 0.1 (standard value)
- **Effect**: Controls servo gain (speed vs stability tradeoff)

## Analysis Commands

After running all tests:

```bash
# Compare convergence rates
python3 ../unified_eval.py results/test1_*/*/results_*.npz --compare

# Generate detailed tables
for dir in results/test1_*/*/; do
    echo "Analyzing: $dir"
    python3 ../unified_eval.py ${dir}results_*.npz --table > ${dir}analysis.txt
done
```

## Expected Insights

1. **Feature Stagnation Impact**: Compare test1_current vs test1_no_early_stop
   - How many iterations saved?
   - Does early stopping affect final accuracy?

2. **Tiling Activation Timing**: Compare test1_current vs test1_tiling_0.9
   - Does later tiling activation improve convergence?
   - Trade-off between speed and accuracy?

3. **Stagnation Window Length**: Compare test1_current vs test1_stagnation_20
   - Does longer window improve final pose accuracy?
   - How many additional iterations needed?

## Notes

- Classical methods (SIFT, ORB, AKAZE) don't use tiling, so tiling threshold changes won't affect them
- DINOv3-1440 uses max_iterations=500 (not 1500) as specified
- All tests use Hollywood poster (model 99) with 500 samples by default
- Results are fully comparable across tests due to identical sampling strategy