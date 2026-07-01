#!/usr/bin/env python3
"""
Unified Complete Evaluation Script
===================================
Combines ALL metrics from unified_eval_legacy.py and eval_time_to_convergence.py:
- Legacy convergence criterion (90% relative OR <1cm absolute)
- Time to convergence at multiple thresholds
- Final and best position/rotation errors
- FPS breakdown (pre-tiling, post-tiling)

Usage:
    python3 unified_eval_complete.py results_*.npz --table
    python3 unified_eval_complete.py results_*.npz --thresholds
    python3 unified_eval_complete.py results_*.npz --all

Author: Claude Code
Date: January 2025
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional, Any
import sys


class UnifiedCompleteEvaluator:
    """
    Complete evaluator combining legacy convergence criterion with time-to-convergence.

    Convergence criterion (LEGACY - matches controller):
    - 90% relative reduction in BOTH position AND rotation, OR
    - Absolute threshold: position <1cm AND rotation <1°
    """

    # Desired pose for error calculation (from simulation setup)
    DESIRED_POSITION = np.array([0, 0, 0.61])  # meters
    DESIRED_ORIENTATION = np.array([0, 0.7071068, 0, 0.7071068])  # xyzw quaternion

    def __init__(self, npz_files: List[str], verbose: bool = False):
        """
        Initialize evaluator with NPZ file paths.

        Args:
            npz_files: List of paths to NPZ result files
            verbose: Print detailed progress information
        """
        self.npz_files = npz_files
        self.verbose = verbose
        self.model_results: Dict[str, Dict[str, Any]] = {}

    def calculate_errors_at_iteration(self, position: np.ndarray,
                                       orientation: np.ndarray) -> Tuple[float, float]:
        """
        Calculate position (cm) and rotation (deg) error for a single iteration.

        Args:
            position: 3D position in meters [x, y, z]
            orientation: Quaternion in xyzw format

        Returns:
            Tuple of (position_error_cm, rotation_error_deg)
        """
        # Position error in cm
        pos_err = np.linalg.norm(position - self.DESIRED_POSITION) * 100

        # Rotation error in degrees
        try:
            current_rot = R.from_quat(orientation)
            desired_rot = R.from_quat(self.DESIRED_ORIENTATION)
            rot_diff = current_rot.inv() * desired_rot
            rot_err = rot_diff.magnitude() * (180.0 / np.pi)
        except Exception:
            rot_err = float('inf')

        return pos_err, rot_err

    def calculate_full_error_history(self, position_history: np.ndarray,
                                     orientation_history: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Calculate position and rotation errors for all iterations.

        Args:
            position_history: Array of positions [N, 3]
            orientation_history: Array of quaternions [N, 4]

        Returns:
            Tuple of (position_errors_cm, rotation_errors_deg) lists
        """
        position_errors = []
        rotation_errors = []

        for i in range(len(position_history)):
            pos_err, rot_err = self.calculate_errors_at_iteration(
                position_history[i], orientation_history[i]
            )
            position_errors.append(pos_err)
            rotation_errors.append(rot_err)

        return position_errors, rotation_errors

    def find_legacy_convergence_iteration(self, position_errors: List[float],
                                           rotation_errors: List[float]) -> Optional[int]:
        """
        Find first iteration using LEGACY convergence criterion.

        Converged if EITHER:
        - 90% relative reduction in BOTH position AND rotation
        - Absolute threshold: position <1cm AND rotation <1°

        This matches the criterion used by the controller during experiments.

        Args:
            position_errors: List of position errors per iteration (cm)
            rotation_errors: List of rotation errors per iteration (deg)

        Returns:
            Iteration index where convergence reached, or None if never reached
        """
        if len(position_errors) == 0 or len(rotation_errors) == 0:
            return None

        initial_pos = position_errors[0]
        initial_rot = rotation_errors[0]

        for i, (pos_err, rot_err) in enumerate(zip(position_errors, rotation_errors)):
            # Criterion 1: 90% relative reduction in BOTH
            relative_converged = False
            if initial_pos > 1e-6 and initial_rot > 1e-6:
                relative_converged = (pos_err <= 0.1 * initial_pos) and (rot_err <= 0.1 * initial_rot)

            # Criterion 2: Absolute threshold (<1cm, <1°)
            absolute_converged = (pos_err < 1.0) and (rot_err < 1.0)

            # EITHER criterion triggers convergence
            if relative_converged or absolute_converged:
                return i

        return None

    def find_relative_threshold_iteration(self, position_errors: List[float],
                                          rotation_errors: List[float],
                                          threshold: float) -> Optional[int]:
        """
        Find first iteration where BOTH errors are below threshold of initial.

        Args:
            position_errors: List of position errors per iteration (cm)
            rotation_errors: List of rotation errors per iteration (deg)
            threshold: Fraction of initial error (e.g., 0.1 for 90% reduction)

        Returns:
            Iteration index where threshold reached, or None if never reached
        """
        if len(position_errors) == 0 or len(rotation_errors) == 0:
            return None

        initial_pos = position_errors[0]
        initial_rot = rotation_errors[0]

        # Avoid division by zero for already-converged cases
        if initial_pos < 1e-6 or initial_rot < 1e-6:
            return 0

        target_pos = threshold * initial_pos
        target_rot = threshold * initial_rot

        for i, (pos_err, rot_err) in enumerate(zip(position_errors, rotation_errors)):
            if pos_err <= target_pos and rot_err <= target_rot:
                return i

        return None

    def find_absolute_threshold_iteration(self, position_errors: List[float],
                                          rotation_errors: List[float],
                                          pos_threshold: float,
                                          rot_threshold: float) -> Optional[int]:
        """
        Find first iteration where BOTH errors are below absolute thresholds.

        Args:
            position_errors: List of position errors per iteration (cm)
            rotation_errors: List of rotation errors per iteration (deg)
            pos_threshold: Absolute position threshold in cm
            rot_threshold: Absolute rotation threshold in degrees

        Returns:
            Iteration index where threshold reached, or None if never reached
        """
        if len(position_errors) == 0 or len(rotation_errors) == 0:
            return None

        for i, (pos_err, rot_err) in enumerate(zip(position_errors, rotation_errors)):
            if pos_err <= pos_threshold and rot_err <= rot_threshold:
                return i

        return None

    def find_all_thresholds(self, position_errors: List[float],
                            rotation_errors: List[float]) -> Dict[str, Optional[int]]:
        """
        Find iterations for all threshold types (legacy, relative, and absolute).

        Returns dict with keys: iter_legacy, iter_90, iter_95, iter_99, iter_5cm, iter_2cm, iter_1cm
        """
        return {
            # Legacy convergence (90% relative OR <1cm absolute)
            'iter_legacy': self.find_legacy_convergence_iteration(position_errors, rotation_errors),
            # Relative thresholds (% of initial error)
            'iter_90': self.find_relative_threshold_iteration(position_errors, rotation_errors, 0.10),
            'iter_95': self.find_relative_threshold_iteration(position_errors, rotation_errors, 0.05),
            'iter_99': self.find_relative_threshold_iteration(position_errors, rotation_errors, 0.01),
            # Absolute thresholds (cm for position, degrees for rotation)
            'iter_5cm': self.find_absolute_threshold_iteration(position_errors, rotation_errors, 5.0, 5.0),
            'iter_2cm': self.find_absolute_threshold_iteration(position_errors, rotation_errors, 2.0, 2.0),
            'iter_1cm': self.find_absolute_threshold_iteration(position_errors, rotation_errors, 1.0, 1.0),
        }

    def analyze_sample(self, sample_idx: int,
                       model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single sample: find all convergence thresholds and calculate times.

        Args:
            sample_idx: Index of the sample to analyze
            model_data: Dictionary containing all NPZ data for the model

        Returns:
            Dictionary with analysis results for this sample
        """
        result = {
            'sample_idx': sample_idx,
            'converged_legacy': False,
            'convergence_iter_legacy': None,
            'time_legacy': None,
            'initial_pos_error': 0.0,
            'initial_rot_error': 0.0,
            'final_pos_error': 0.0,
            'final_rot_error': 0.0,
            'best_pos_error': 0.0,
            'best_rot_error': 0.0,
            'tiling_activated': False,
            'tiling_switch_iter': -1,
            # Multi-threshold iterations
            'iter_legacy': None,
            'iter_90': None,
            'iter_95': None,
            'iter_99': None,
            'iter_5cm': None,
            'iter_2cm': None,
            'iter_1cm': None,
            # Multi-threshold times
            'time_90': None,
            'time_95': None,
            'time_99': None,
            'time_5cm': None,
            'time_2cm': None,
            'time_1cm': None,
        }

        try:
            # Get position and orientation histories
            pos_history = model_data['all_position_histories'][sample_idx]
            orient_history = model_data['all_orientation_histories'][sample_idx]
            iter_times = model_data['all_iteration_times'][sample_idx]

            # Get tiling switch iteration
            tiling_switch = model_data.get('tiling_switch_iterations', [])
            if len(tiling_switch) > sample_idx:
                tiling_switch_iter = int(tiling_switch[sample_idx])
            else:
                tiling_switch_iter = -1

            result['tiling_switch_iter'] = tiling_switch_iter
            result['tiling_activated'] = tiling_switch_iter > 0

            # Calculate error history
            pos_errors, rot_errors = self.calculate_full_error_history(
                pos_history, orient_history
            )

            # Store initial, final, and best errors
            if len(pos_errors) > 0:
                result['initial_pos_error'] = pos_errors[0]
                result['final_pos_error'] = pos_errors[-1]
                result['best_pos_error'] = min(pos_errors)
            if len(rot_errors) > 0:
                result['initial_rot_error'] = rot_errors[0]
                result['final_rot_error'] = rot_errors[-1]
                result['best_rot_error'] = min(rot_errors)

            # Find ALL threshold iterations
            thresholds = self.find_all_thresholds(pos_errors, rot_errors)
            for key, iter_val in thresholds.items():
                result[key] = iter_val
                # Calculate time to each threshold
                if iter_val is not None:
                    result[key.replace('iter_', 'time_')] = np.sum(iter_times[:iter_val + 1])

            # Legacy convergence tracking
            result['convergence_iter_legacy'] = result['iter_legacy']
            result['converged_legacy'] = result['iter_legacy'] is not None

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error analyzing sample {sample_idx}: {e}")

        return result

    def _normalize_npz_data(self, raw_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Normalize NPZ data to handle both model-prefixed and non-prefixed formats.

        Args:
            raw_data: Raw dictionary from NPZ file

        Returns:
            Tuple of (normalized_data, model_prefix) where model_prefix is empty for non-prefixed format
        """
        keys = list(raw_data.keys())

        # Check for model-prefixed format (e.g., "model_99_all_position_histories")
        model_prefixes = set()
        for key in keys:
            if key.startswith('model_') and '_all_position_histories' in key:
                # Extract prefix like "model_99"
                prefix = key.split('_all_position_histories')[0]
                model_prefixes.add(prefix)

        if model_prefixes:
            # Model-prefixed format - use first model found
            prefix = sorted(model_prefixes)[0]  # Use first if multiple
            if self.verbose:
                print(f"  Detected model-prefixed format: {prefix}")

            # Create normalized dict by stripping prefix
            normalized = {}
            prefix_with_underscore = prefix + '_'
            for key, value in raw_data.items():
                if key.startswith(prefix_with_underscore):
                    new_key = key[len(prefix_with_underscore):]
                    normalized[new_key] = value

            return normalized, prefix
        else:
            # Original non-prefixed format
            return dict(raw_data), ""

    def analyze_model(self, npz_path: str) -> Dict[str, Any]:
        """
        Analyze all samples for one model/method.

        Args:
            npz_path: Path to NPZ result file

        Returns:
            Dictionary with aggregated statistics for this model
        """
        # Extract model name from filename
        filename = Path(npz_path).stem
        model_name = filename.replace('results_', '').replace('_', ' ')

        if self.verbose:
            print(f"\nAnalyzing: {model_name}")

        # Load NPZ data
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return None

        # Convert to dict and normalize format
        raw_data = {key: data[key] for key in data.files}
        model_data, model_prefix = self._normalize_npz_data(raw_data)

        # Update model name if we have a prefix
        if model_prefix and model_prefix != "model_99":
            # Include model ID in display name
            model_name = f"{model_name} ({model_prefix})"

        # Get number of samples
        n_samples = len(model_data.get('all_position_histories', []))
        if n_samples == 0:
            print(f"  Warning: No samples found in {npz_path}")
            print(f"  Available keys: {list(model_data.keys())[:10]}...")
            return None

        if self.verbose:
            print(f"  Total samples: {n_samples}")

        # Analyze each sample
        sample_results = []
        for i in range(n_samples):
            result = self.analyze_sample(i, model_data)
            sample_results.append(result)

        # Aggregate statistics
        converged_legacy = [r for r in sample_results if r['converged_legacy']]
        not_converged = [r for r in sample_results if not r['converged_legacy']]

        # Extract pre/post tiling FPS from NPZ (stored as one value per sample)
        pre_fps = model_data.get('pre_tiling_fps', [])
        post_fps = model_data.get('post_tiling_fps', [])

        # Check if this is a tiling method
        is_tiling_method = any(r['tiling_activated'] for r in sample_results)

        # Calculate statistics
        stats = {
            'model_name': model_name,
            'npz_path': npz_path,
            'total_samples': n_samples,
            'converged_legacy_count': len(converged_legacy),
            'not_converged_count': len(not_converged),
            'convergence_rate_legacy': len(converged_legacy) / n_samples * 100 if n_samples > 0 else 0,
            'is_tiling_method': is_tiling_method,
            'sample_results': sample_results,
        }

        # Helper to compute mean/std for a threshold
        def compute_threshold_stats(samples, iter_key, time_key):
            reached = [r for r in samples if r[iter_key] is not None]
            if len(reached) > 0:
                iters = [r[iter_key] for r in reached]
                times = [r[time_key] for r in reached if r[time_key] is not None]
                return {
                    f'{iter_key}_mean': np.mean(iters),
                    f'{iter_key}_std': np.std(iters),
                    f'{time_key}_mean': np.mean(times) if times else None,
                    f'{time_key}_std': np.std(times) if times else None,
                    f'{iter_key.replace("iter_", "reached_")}_count': len(reached),
                    f'{iter_key.replace("iter_", "reached_")}_pct': len(reached) / n_samples * 100,
                }
            return {
                f'{iter_key}_mean': None,
                f'{iter_key}_std': None,
                f'{time_key}_mean': None,
                f'{time_key}_std': None,
                f'{iter_key.replace("iter_", "reached_")}_count': 0,
                f'{iter_key.replace("iter_", "reached_")}_pct': 0.0,
            }

        # Compute stats for legacy + all thresholds
        for threshold in ['legacy', '90', '95', '99']:
            thresh_stats = compute_threshold_stats(sample_results, f'iter_{threshold}', f'time_{threshold}')
            stats.update(thresh_stats)

        # Compute stats for all absolute thresholds
        for threshold in ['5cm', '2cm', '1cm']:
            thresh_stats = compute_threshold_stats(sample_results, f'iter_{threshold}', f'time_{threshold}')
            stats.update(thresh_stats)

        # Final error statistics (for converged samples)
        if len(converged_legacy) > 0:
            final_pos = [r['final_pos_error'] for r in converged_legacy]
            final_rot = [r['final_rot_error'] for r in converged_legacy]
            stats['final_pos_mean'] = np.mean(final_pos)
            stats['final_pos_std'] = np.std(final_pos)
            stats['final_rot_mean'] = np.mean(final_rot)
            stats['final_rot_std'] = np.std(final_rot)
        else:
            stats['final_pos_mean'] = None
            stats['final_pos_std'] = None
            stats['final_rot_mean'] = None
            stats['final_rot_std'] = None

        # Best precision statistics (all samples)
        best_pos = [r['best_pos_error'] for r in sample_results if r['best_pos_error'] > 0]
        best_rot = [r['best_rot_error'] for r in sample_results if r['best_rot_error'] > 0]
        if best_pos:
            stats['best_pos_mean'] = np.mean(best_pos)
            stats['best_pos_std'] = np.std(best_pos)
        else:
            stats['best_pos_mean'] = None
            stats['best_pos_std'] = None
        if best_rot:
            stats['best_rot_mean'] = np.mean(best_rot)
            stats['best_rot_std'] = np.std(best_rot)
        else:
            stats['best_rot_mean'] = None
            stats['best_rot_std'] = None

        # FPS statistics (from stored values)
        if len(pre_fps) > 0:
            valid_pre_fps = [f for f in pre_fps if f > 0]
            stats['pre_fps_mean'] = np.mean(valid_pre_fps) if valid_pre_fps else None
        else:
            stats['pre_fps_mean'] = None

        if len(post_fps) > 0 and is_tiling_method:
            valid_post_fps = [f for f in post_fps if f > 0]
            stats['post_fps_mean'] = np.mean(valid_post_fps) if valid_post_fps else None
        else:
            stats['post_fps_mean'] = None

        if self.verbose:
            print(f"  Converged (legacy): {stats['converged_legacy_count']}/{n_samples} ({stats['convergence_rate_legacy']:.1f}%)")
            if stats.get('time_legacy_mean') is not None:
                print(f"  Mean time to convergence: {stats['time_legacy_mean']:.2f}s")

        return stats

    def load_and_analyze(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all NPZ files and analyze.

        Returns:
            Dictionary mapping model names to their statistics
        """
        for npz_path in self.npz_files:
            stats = self.analyze_model(npz_path)
            if stats is not None:
                self.model_results[stats['model_name']] = stats

        return self.model_results

    def generate_table(self) -> str:
        """
        Generate primary table with legacy convergence criterion.

        Returns:
            Formatted table string
        """
        if not self.model_results:
            return "No results to display."

        # Header
        lines = []
        lines.append("=" * 130)
        lines.append("UNIFIED VISUAL SERVOING EVALUATION (Legacy Convergence Criterion)")
        lines.append("=" * 130)
        lines.append("Criterion: 90% relative reduction OR absolute <1cm position AND <1° rotation")
        lines.append("")

        # Column headers
        header = f"{'Method':<22} | {'Conv%':>6} | {'Time(s)':>10} | {'Final Pos':>10} | {'Final Rot':>10} | {'Best Pos':>10} | {'Pre-FPS':>8} | {'Post-FPS':>9}"
        lines.append(header)
        lines.append("-" * 130)

        # Sort by model name
        sorted_models = sorted(self.model_results.keys())

        for model_name in sorted_models:
            stats = self.model_results[model_name]

            # Format values
            conv_pct = f"{stats['convergence_rate_legacy']:.1f}%"

            if stats['time_legacy_mean'] is not None:
                time_str = f"{stats['time_legacy_mean']:.1f}±{stats['time_legacy_std']:.1f}"
            else:
                time_str = "N/A"

            if stats['final_pos_mean'] is not None:
                final_pos_str = f"{stats['final_pos_mean']:.2f}±{stats['final_pos_std']:.2f}"
            else:
                final_pos_str = "N/A"

            if stats['final_rot_mean'] is not None:
                final_rot_str = f"{stats['final_rot_mean']:.2f}±{stats['final_rot_std']:.2f}"
            else:
                final_rot_str = "N/A"

            if stats['best_pos_mean'] is not None:
                best_pos_str = f"{stats['best_pos_mean']:.2f}±{stats['best_pos_std']:.2f}"
            else:
                best_pos_str = "N/A"

            if stats['pre_fps_mean'] is not None:
                pre_fps_str = f"{stats['pre_fps_mean']:.1f}"
            else:
                pre_fps_str = "N/A"

            if stats['post_fps_mean'] is not None:
                post_fps_str = f"{stats['post_fps_mean']:.1f}"
            elif stats['is_tiling_method']:
                post_fps_str = "-"
            else:
                post_fps_str = "N/A"

            # Truncate model name if too long
            display_name = model_name[:22] if len(model_name) > 22 else model_name

            row = f"{display_name:<22} | {conv_pct:>6} | {time_str:>10} | {final_pos_str:>10} | {final_rot_str:>10} | {best_pos_str:>10} | {pre_fps_str:>8} | {post_fps_str:>9}"
            lines.append(row)

        lines.append("-" * 130)
        lines.append("")
        lines.append("Notes:")
        lines.append("  - Conv%: Percentage using legacy criterion (90% relative OR <1cm absolute)")
        lines.append("  - Time(s): Mean time to reach legacy convergence (seconds)")
        lines.append("  - Final Pos/Rot: Final errors at iteration end (cm/°) for converged samples")
        lines.append("  - Best Pos: Best position achieved during trajectory (cm)")
        lines.append("  - Pre/Post FPS: Processing rate before/after tiling activation")
        lines.append("")

        return "\n".join(lines)

    def generate_threshold_table(self) -> str:
        """
        Generate table comparing all convergence thresholds.

        Returns:
            Formatted table string
        """
        if not self.model_results:
            return "No results to display."

        lines = []
        lines.append("=" * 120)
        lines.append("MULTI-THRESHOLD ANALYSIS (Time to reach threshold in seconds)")
        lines.append("=" * 120)
        lines.append("")

        # Column headers
        header = f"{'Method':<22} | {'@90%(s)':>9} | {'@95%(s)':>9} | {'@99%(s)':>9} | {'<5cm(s)':>9} | {'<2cm(s)':>9} | {'<1cm(s)':>9} | {'%<1cm':>7}"
        lines.append(header)
        lines.append("-" * 120)

        sorted_models = sorted(self.model_results.keys())

        for model_name in sorted_models:
            stats = self.model_results[model_name]

            # Format time values
            def fmt_time(mean_key, std_key=None):
                mean = stats.get(mean_key)
                if mean is not None:
                    if std_key and stats.get(std_key) is not None:
                        return f"{mean:.1f}±{stats[std_key]:.1f}"
                    return f"{mean:.1f}"
                return "N/A"

            time_90 = fmt_time('time_90_mean', 'time_90_std')
            time_95 = fmt_time('time_95_mean', 'time_95_std')
            time_99 = fmt_time('time_99_mean', 'time_99_std')
            time_5cm = fmt_time('time_5cm_mean', 'time_5cm_std')
            time_2cm = fmt_time('time_2cm_mean', 'time_2cm_std')
            time_1cm = fmt_time('time_1cm_mean', 'time_1cm_std')

            # % reaching <1cm
            pct_1cm = stats.get('reached_1cm_pct', 0)
            pct_str = f"{pct_1cm:.0f}%"

            display_name = model_name[:22] if len(model_name) > 22 else model_name
            row = f"{display_name:<22} | {time_90:>9} | {time_95:>9} | {time_99:>9} | {time_5cm:>9} | {time_2cm:>9} | {time_1cm:>9} | {pct_str:>7}"
            lines.append(row)

        lines.append("-" * 120)
        lines.append("")
        lines.append("Notes:")
        lines.append("  - @X%: Time to reduce BOTH position AND rotation to X% of initial")
        lines.append("  - <Xcm: Time to reach BOTH position <Xcm AND rotation <X°")
        lines.append("  - %<1cm: Percentage of samples that reached <1cm precision")
        lines.append("")

        return "\n".join(lines)

    def generate_paper_table(self) -> str:
        """
        Generate paper-ready summary combining all key metrics.

        Returns:
            Formatted table string
        """
        if not self.model_results:
            return "No results to display."

        lines = []
        lines.append("=" * 115)
        lines.append("PAPER SUMMARY: COMPLETE EVALUATION METRICS")
        lines.append("=" * 115)
        lines.append("")

        # Create comparison table
        header = f"{'Method':<22} | {'Conv%':>6} | {'Time(s)':>8} | {'@99%(s)':>8} | {'<1cm(s)':>8} | {'%<1cm':>6} | {'BestPos':>8} | {'FPS':>10}"
        lines.append(header)
        lines.append("-" * 115)

        sorted_models = sorted(self.model_results.keys())

        for model_name in sorted_models:
            stats = self.model_results[model_name]

            def fmt_time_short(mean_key):
                mean = stats.get(mean_key)
                if mean is not None:
                    return f"{mean:.1f}"
                return "N/A"

            conv_pct = f"{stats['convergence_rate_legacy']:.0f}%"
            time_conv = fmt_time_short('time_legacy_mean')
            t99 = fmt_time_short('time_99_mean')
            t1cm = fmt_time_short('time_1cm_mean')

            pct_1cm = stats.get('reached_1cm_pct', 0)
            pct_str = f"{pct_1cm:.0f}%"

            best_pos = stats.get('best_pos_mean')
            best_str = f"{best_pos:.2f}cm" if best_pos else "N/A"

            # FPS - show pre→post for tiling methods
            fps_str = ""
            if stats.get('pre_fps_mean') is not None:
                fps_str = f"{stats['pre_fps_mean']:.1f}"
                if stats.get('post_fps_mean') is not None and stats['is_tiling_method']:
                    fps_str += f"→{stats['post_fps_mean']:.1f}"

            display_name = model_name[:22] if len(model_name) > 22 else model_name
            row = f"{display_name:<22} | {conv_pct:>6} | {time_conv:>8} | {t99:>8} | {t1cm:>8} | {pct_str:>6} | {best_str:>8} | {fps_str:>10}"
            lines.append(row)

        lines.append("-" * 115)
        lines.append("")
        lines.append("KEY:")
        lines.append("  - Conv%: Legacy convergence rate (90% relative OR <1cm absolute)")
        lines.append("  - Time(s): Time to legacy convergence")
        lines.append("  - @99%(s): Time to 99% error reduction")
        lines.append("  - <1cm(s): Time to reach <1cm position AND <1° rotation")
        lines.append("  - %<1cm: Percentage reaching sub-centimeter precision")
        lines.append("  - BestPos: Best position achieved during trajectory")
        lines.append("  - FPS: Pre-tiling→Post-tiling frames per second")
        lines.append("")

        return "\n".join(lines)

    def generate_detailed_report(self) -> str:
        """
        Generate detailed per-sample breakdown.

        Returns:
            Formatted detailed report string
        """
        if not self.model_results:
            return "No results to display."

        lines = []
        lines.append("=" * 110)
        lines.append("DETAILED PER-SAMPLE ANALYSIS")
        lines.append("=" * 110)

        for model_name in sorted(self.model_results.keys()):
            stats = self.model_results[model_name]

            lines.append("")
            lines.append(f"Model: {model_name}")
            lines.append(f"File: {stats['npz_path']}")
            lines.append(f"Convergence Rate (Legacy): {stats['converged_legacy_count']}/{stats['total_samples']} ({stats['convergence_rate_legacy']:.1f}%)")
            lines.append("-" * 90)

            # Per-sample table header
            lines.append(f"{'Sample':>6} | {'Conv?':>5} | {'Time(s)':>9} | {'Init Pos':>9} | {'Final Pos':>9} | {'Best Pos':>9} | {'Tile?':>5}")
            lines.append("-" * 90)

            for result in stats['sample_results']:
                idx = result['sample_idx']
                converged = "Yes" if result['converged_legacy'] else "No"
                time_s = f"{result['time_legacy']:.2f}" if result['time_legacy'] is not None else "-"
                init_pos = f"{result['initial_pos_error']:.2f}cm"
                final_pos = f"{result['final_pos_error']:.2f}cm"
                best_pos = f"{result['best_pos_error']:.2f}cm"
                tiled = "Yes" if result['tiling_activated'] else "No"

                row = f"{idx:>6} | {converged:>5} | {time_s:>9} | {init_pos:>9} | {final_pos:>9} | {best_pos:>9} | {tiled:>5}"
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def generate_csv(self, output_path: str) -> None:
        """
        Export results to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        if not self.model_results:
            print("No results to export.")
            return

        import csv

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'model_name', 'total_samples', 'converged_legacy_count', 'convergence_rate_legacy',
                'time_legacy_mean', 'time_legacy_std',
                'time_90_mean', 'time_95_mean', 'time_99_mean',
                'time_5cm_mean', 'time_2cm_mean', 'time_1cm_mean',
                'reached_1cm_pct',
                'final_pos_mean', 'final_pos_std', 'final_rot_mean', 'final_rot_std',
                'best_pos_mean', 'best_pos_std', 'best_rot_mean', 'best_rot_std',
                'pre_fps_mean', 'post_fps_mean', 'is_tiling_method'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for model_name in sorted(self.model_results.keys()):
                stats = self.model_results[model_name]
                row = {field: stats.get(field, '') for field in fieldnames}
                writer.writerow(row)

        print(f"Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified complete evaluation of visual servoing results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Primary table with legacy convergence criterion
    python3 unified_eval_complete.py results_*.npz --table

    # Multi-threshold analysis (90%, 95%, 99%, <5cm, <2cm, <1cm)
    python3 unified_eval_complete.py results_*.npz --thresholds

    # Paper-ready summary table
    python3 unified_eval_complete.py results_*.npz --paper

    # Detailed per-sample view
    python3 unified_eval_complete.py results_*.npz --detailed

    # Export to CSV
    python3 unified_eval_complete.py results_*.npz --csv output.csv

    # All outputs at once
    python3 unified_eval_complete.py results_*.npz --all
        """
    )

    parser.add_argument('npz_files', nargs='+',
                        help='NPZ result files to analyze')
    parser.add_argument('--table', action='store_true',
                        help='Generate primary table with legacy convergence')
    parser.add_argument('--thresholds', action='store_true',
                        help='Generate multi-threshold table (90%%, 95%%, 99%%, <5cm, <2cm, <1cm)')
    parser.add_argument('--paper', action='store_true',
                        help='Generate paper-ready summary table')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed per-sample breakdown')
    parser.add_argument('--csv', type=str, metavar='FILE',
                        help='Export results to CSV file')
    parser.add_argument('--all', action='store_true',
                        help='Generate all output formats')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress information')

    args = parser.parse_args()

    # Validate files exist
    valid_files = []
    for f in args.npz_files:
        path = Path(f)
        if path.exists():
            valid_files.append(str(path))
        else:
            print(f"Warning: File not found: {f}")

    if not valid_files:
        print("Error: No valid NPZ files found.")
        sys.exit(1)

    # Create evaluator and analyze
    evaluator = UnifiedCompleteEvaluator(valid_files, verbose=args.verbose)
    evaluator.load_and_analyze()

    # Track if any output was generated
    output_generated = False

    # Generate requested outputs
    if args.csv:
        evaluator.generate_csv(args.csv)
        output_generated = True

    if args.detailed or args.all:
        print(evaluator.generate_detailed_report())
        output_generated = True

    if args.thresholds or args.all:
        print(evaluator.generate_threshold_table())
        output_generated = True

    if args.paper or args.all:
        print(evaluator.generate_paper_table())
        output_generated = True

    # Default to primary table if no specific output requested
    if args.table or args.all or not output_generated:
        print(evaluator.generate_table())


if __name__ == '__main__':
    main()
