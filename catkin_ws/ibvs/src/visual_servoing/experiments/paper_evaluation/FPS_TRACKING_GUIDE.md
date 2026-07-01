# FPS Tracking Guide for TEST1 Parameter Study

## Overview
The enhanced FPS tracking properly separates pre-tiling and post-tiling performance metrics for all tiling modes (MODE 2 and MODE 4). This functionality is now integrated directly into `run_multi_model_experiment_with_hollywood.py`.

## How It Works

### 1. Data Collection
The experiment runner (`run_multi_model_experiment_with_hollywood.py`):
- Tracks iteration times for each sample
- Records when tiling switches from low-res to high-res mode
- Calculates separate FPS for pre-tiling and post-tiling phases

### 2. FPS Calculation
```python
# For non-tiling methods (MODE 1):
- Single FPS value calculated from all iteration times
- post_tiling_fps = 0.0 (no tiling occurs)

# For tiling methods (MODE 2):
- pre_tiling_fps: Average FPS before tiling activation
- post_tiling_fps: Average FPS after tiling activation
- Tiling switch iteration recorded for analysis
```

### 3. Data Storage in NPZ Files
Each result file contains:
- `pre_tiling_fps`: Array of FPS values before tiling (or overall FPS if no tiling)
- `post_tiling_fps`: Array of FPS values after tiling (0.0 if no tiling)
- `tiling_switch_iterations`: When tiling activated for each sample

## Methods and Their FPS Behavior

### Non-Tiling Methods (MODE 1)
- **DINOv2-308**: No tiling, single FPS value (~15-20 Hz)
- **DINOv2-518**: No tiling, single FPS value (~8-12 Hz)
- **DINOv3-1440**: No tiling, single FPS value (~1-2 Hz)
- **SIFT/ORB/AKAZE**: Classical methods, single FPS value

### Tiling Methods (MODE 2)
- **DINOv2-224 (7x7)**: Fast pre-tiling (~30 Hz), slower post-tiling (~5 Hz)
- **DINOv3-256 (6x6)**: Fast pre-tiling (~25 Hz), slower post-tiling (~7 Hz)
- **AM-RADIO-256 (6x6)**: Fast pre-tiling (~25 Hz), slower post-tiling (~7 Hz)

## Analyzing FPS Results

### Extract FPS Data from NPZ
```python
import numpy as np

# Load results
data = np.load('results_dinov2_224_7x7_model99.npz')

# For tiling methods
pre_fps = data['model_99_pre_tiling_fps']
post_fps = data['model_99_post_tiling_fps']
switch_iters = data['model_99_tiling_switch_iterations']

# Calculate statistics
print(f"Pre-tiling FPS: {pre_fps.mean():.1f} ± {pre_fps.std():.1f} Hz")
if post_fps.mean() > 0:
    print(f"Post-tiling FPS: {post_fps.mean():.1f} ± {post_fps.std():.1f} Hz")
    print(f"Slowdown factor: {pre_fps.mean()/post_fps.mean():.2f}x")
```

### Compare Across Test Variations
```python
# Compare how parameter changes affect FPS

# Test1 Current (baseline)
current = np.load('test1_current/*/results_dinov2_224_7x7_model99.npz')

# Test1 Tiling 0.9 (later activation)
late_tiling = np.load('test1_tiling_0.9/*/results_dinov2_224_7x7_model99.npz')

# Later tiling activation means:
# - More iterations at high FPS (pre-tiling)
# - Fewer iterations at low FPS (post-tiling)
# - Overall faster completion time possible
```

## Expected Insights from FPS Analysis

### 1. Impact of Tiling Switch Threshold
- **0.8 threshold (current)**: Switches at 20% error → earlier tiling → more time in slow mode
- **0.9 threshold (test1_tiling_0.9)**: Switches at 10% error → later tiling → more time in fast mode

### 2. Feature Stagnation Impact
- **With stagnation**: May stop before tiling activates → only pre-tiling FPS relevant
- **Without stagnation (test1_no_early_stop)**: Always reaches tiling phase → both FPS values important

### 3. Performance Trade-offs
```
High pre-tiling FPS + Late tiling switch = Faster overall convergence
BUT
May sacrifice final accuracy if tiling needed for precision
```

## Console Output During Experiments
```
Sample 1 FPS: Pre-tiling=28.3 Hz, Post-tiling=5.2 Hz (switched at iter 187)
Sample 2 FPS: Pre-tiling=27.9 Hz, Post-tiling=5.1 Hz (switched at iter 195)
Sample 3 FPS: 28.1 Hz (no tiling)  # Converged before tiling activated
Sample 4 FPS: Pre-tiling=28.0 Hz, Post-tiling=5.3 Hz (switched at iter 201)

FPS Summary:
  Pre-tiling:  28.1 ± 0.2 Hz
  Post-tiling: 5.2 ± 0.1 Hz
  Slowdown factor: 5.40x
```

## Key Observations

1. **Tiling causes 5-6x slowdown** in processing speed
2. **DINOv3-1440** is extremely slow (~1-2 Hz) due to high resolution
3. **Classical methods** (SIFT/ORB) maintain consistent FPS throughout
4. **Optimization opportunity**: Later tiling activation can significantly reduce total time

## Using Results in Your Paper

When reporting results:
1. Show both pre-tiling and post-tiling FPS for MODE 2 methods
2. Report the average tiling switch iteration
3. Calculate the percentage of iterations spent in each mode
4. Consider total time = (pre_iters/pre_fps) + (post_iters/post_fps)

Example table format:
```
Method          | Pre-FPS | Post-FPS | Switch Iter | Total Time
----------------|---------|----------|-------------|------------
DINOv2-224 7x7  | 28.1 Hz | 5.2 Hz   | 193 ± 15   | 45.2s
DINOv3-256 6x6  | 24.5 Hz | 7.1 Hz   | 201 ± 18   | 38.9s
AM-RADIO 6x6    | 25.3 Hz | 6.9 Hz   | 198 ± 12   | 40.1s
```