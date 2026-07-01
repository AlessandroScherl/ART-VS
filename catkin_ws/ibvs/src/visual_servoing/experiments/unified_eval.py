#!/usr/bin/env python3
"""
Unified Evaluation Script for Visual Servoing Results
======================================================
Combines all evaluation metrics from individual scripts into one comprehensive analysis tool.

Metrics calculated:
- Convergence statistics (rate, iterations)
- Position and orientation errors (final, lowest, average)
- Absolute Pose Error (APE) - deviation from optimal geodesic path
- Trajectory length ratio - actual path vs optimal straight line
- Velocity profiles and smoothness
- Per-model and overall statistics
"""

import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tabulate import tabulate


class UnifiedEvaluator:
    """Comprehensive evaluator for visual servoing experiments."""
    
    def __init__(self, npz_file: str, verbose: bool = False):
        """
        Initialize evaluator with NPZ file.
        
        Args:
            npz_file: Path to NPZ results file
            verbose: Print detailed information during processing
        """
        self.npz_file = npz_file
        self.verbose = verbose
        
        # Goal pose (standard for all experiments)
        self.desired_position = np.array([0, 0, 0.61])
        self.desired_orientation = np.array([0, 0.7071068, 0, 0.7071068])
        
        # Load data
        self.data = np.load(npz_file, allow_pickle=True)
        self.model_results = self._extract_model_results()
        
    def _extract_model_results(self) -> Dict[int, Dict[str, Any]]:
        """Extract and organize results by model."""
        model_results = {}
        
        # Find all model prefixes in the data
        model_prefixes = set()
        for key in self.data.files:
            if key.startswith('model_'):
                parts = key.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    model_prefixes.add(f"model_{parts[1]}")
        
        # Extract data for each model
        for prefix in model_prefixes:
            model_id = int(prefix.split('_')[1])
            model_data = {}
            
            # Extract all data for this model
            for key in self.data.files:
                if key.startswith(prefix + '_'):
                    field_name = key[len(prefix) + 1:]
                    model_data[field_name] = self.data[key]
            
            if model_data:  # Only add if we have data
                model_results[model_id] = model_data
                
        return model_results
    
    def evaluate_convergence(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate convergence statistics for a model."""
        results = {}
        
        # Handle different key names for backward compatibility
        convergence_key = 'convergence_flags' if 'convergence_flags' in model_data else 'converged'
        if convergence_key not in model_data:
            return results
            
        convergence_flags = model_data[convergence_key]
        
        # Skip if no data
        if len(convergence_flags) == 0:
            return results
        
        total_samples = len(convergence_flags)
        converged_samples = np.sum(convergence_flags)
        
        results['total_samples'] = total_samples
        results['converged_samples'] = converged_samples
        results['convergence_rate'] = (converged_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Iteration statistics for converged samples
        if 'all_iteration_histories' in model_data and converged_samples > 0:
            iterations = model_data['all_iteration_histories']
            converged_iterations = iterations[convergence_flags]
            results['mean_iterations'] = np.mean(converged_iterations)
            results['std_iterations'] = np.std(converged_iterations)
            results['min_iterations'] = np.min(converged_iterations)
            results['max_iterations'] = np.max(converged_iterations)
        
        return results
    
    def evaluate_initial_poses(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate initial pose statistics for a model."""
        results = {}
        
        # Initial positions
        if 'initial_positions' in model_data:
            initial_pos = np.array(model_data['initial_positions'])
            if len(initial_pos) > 0:
                # Calculate distance from goal
                goal_pos = np.array([0, 0, 0.61])
                distances = [np.linalg.norm(pos - goal_pos) * 100 for pos in initial_pos]
                results['mean_initial_distance'] = np.mean(distances)
                results['std_initial_distance'] = np.std(distances)
                results['min_initial_distance'] = np.min(distances)
                results['max_initial_distance'] = np.max(distances)
                
                # Position statistics
                results['mean_initial_x'] = np.mean(initial_pos[:, 0])
                results['mean_initial_y'] = np.mean(initial_pos[:, 1])
                results['mean_initial_z'] = np.mean(initial_pos[:, 2])
        
        # Initial orientations
        if 'initial_orientations' in model_data:
            initial_ori = np.array(model_data['initial_orientations'])
            if len(initial_ori) > 0:
                # Calculate angular distance from goal
                goal_ori = np.array([0, 0.7071068, 0, 0.7071068])
                angles = []
                for ori in initial_ori:
                    angle = self._calculate_orientation_error(ori, goal_ori)
                    angles.append(angle)
                results['mean_initial_angle'] = np.mean(angles)
                results['std_initial_angle'] = np.std(angles)
                results['min_initial_angle'] = np.min(angles)
                results['max_initial_angle'] = np.max(angles)
        
        return results
    
    def evaluate_errors(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate error statistics for a model."""
        results = {}
        
        convergence_key = 'convergence_flags' if 'convergence_flags' in model_data else 'converged'
        if convergence_key not in model_data:
            return results
            
        convergence_flags = model_data[convergence_key]
        
        if len(convergence_flags) == 0:
            return results
            
        converged_mask = convergence_flags == True
        
        # Final errors
        if 'position_errors' in model_data:
            pos_errors = model_data['position_errors'][converged_mask]
            if len(pos_errors) > 0:
                results['mean_position_error'] = np.mean(pos_errors)
                results['std_position_error'] = np.std(pos_errors)
                results['min_position_error'] = np.min(pos_errors)
                results['max_position_error'] = np.max(pos_errors)
        
        if 'orientation_errors' in model_data:
            ori_errors = model_data['orientation_errors'][converged_mask]
            if len(ori_errors) > 0:
                results['mean_orientation_error'] = np.mean(ori_errors)
                results['std_orientation_error'] = np.std(ori_errors)
                results['min_orientation_error'] = np.min(ori_errors)
                results['max_orientation_error'] = np.max(ori_errors)
        
        # Lowest errors achieved
        if 'lowest_position_errors' in model_data:
            lowest_pos = model_data['lowest_position_errors'][converged_mask]
            if len(lowest_pos) > 0:
                results['mean_lowest_position'] = np.mean(lowest_pos)
                results['std_lowest_position'] = np.std(lowest_pos)
        
        if 'lowest_orientation_errors' in model_data:
            lowest_ori = model_data['lowest_orientation_errors'][converged_mask]
            if len(lowest_ori) > 0:
                results['mean_lowest_orientation'] = np.mean(lowest_ori)
                results['std_lowest_orientation'] = np.std(lowest_ori)
        
        return results
    
    def calculate_ape(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Absolute Pose Error (APE) for a model."""
        results = {}
        
        convergence_key = 'convergence_flags' if 'convergence_flags' in model_data else 'converged'
        if convergence_key not in model_data:
            return results
            
        convergence_flags = model_data[convergence_key]
        
        if not any(['all_position_histories' in model_data, 
                    'all_orientation_histories' in model_data]):
            return results
        
        position_apes = []
        orientation_apes = []
        
        # Process each converged sample
        for idx, converged in enumerate(convergence_flags):
            if not converged:
                continue
                
            try:
                # Get trajectories
                if 'all_position_histories' in model_data:
                    pos_history = model_data['all_position_histories'][idx]
                    if 'all_iteration_histories' in model_data:
                        num_iterations = model_data['all_iteration_histories'][idx]
                    else:
                        num_iterations = len(pos_history)
                    
                    # Calculate position APE
                    initial_pos = pos_history[0]
                    geodesic_pos = self._calculate_position_geodesic(initial_pos, num_iterations)
                    pos_errors = [np.linalg.norm(pos_history[i] - geodesic_pos[i]) * 100 
                                  for i in range(min(len(pos_history), len(geodesic_pos)))]
                    position_apes.append(np.mean(pos_errors))
                
                if 'all_orientation_histories' in model_data:
                    ori_history = model_data['all_orientation_histories'][idx]
                    if 'all_iteration_histories' in model_data:
                        num_iterations = model_data['all_iteration_histories'][idx]
                    else:
                        num_iterations = len(ori_history)
                    
                    # Calculate orientation APE
                    initial_ori = ori_history[0]
                    geodesic_ori = self._calculate_orientation_geodesic(initial_ori, num_iterations)
                    ori_errors = [self._calculate_orientation_error(ori_history[i], geodesic_ori[i])
                                  for i in range(min(len(ori_history), len(geodesic_ori)))]
                    orientation_apes.append(np.mean(ori_errors))
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating APE for sample {idx}: {e}")
                continue
        
        # Calculate statistics
        if position_apes:
            results['mean_position_ape'] = np.mean(position_apes)
            results['std_position_ape'] = np.std(position_apes)
        
        if orientation_apes:
            results['mean_orientation_ape'] = np.mean(orientation_apes)
            results['std_orientation_ape'] = np.std(orientation_apes)
        
        return results
    
    def calculate_fps(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate FPS (frames per second) statistics for a model.
        
        Handles both tiling and non-tiling modes:
        - If tiling is active: Reports pre-tiling and post-tiling FPS separately
        - If no tiling: Reports overall FPS throughout the experiment
        """
        results = {}
        
        # Check for tiling-specific FPS data first
        has_tiling = False
        
        # Pre-tiling FPS (before tiling activation)
        if 'pre_tiling_fps' in model_data:
            pre_fps = np.array([fps for fps in model_data['pre_tiling_fps'] if fps > 0])
            if len(pre_fps) > 0:
                results['mean_pre_tiling_fps'] = np.mean(pre_fps)
                results['std_pre_tiling_fps'] = np.std(pre_fps)
                results['min_pre_tiling_fps'] = np.min(pre_fps)
                results['max_pre_tiling_fps'] = np.max(pre_fps)
                has_tiling = True
        
        # Post-tiling FPS (after tiling activation)
        if 'post_tiling_fps' in model_data:
            post_fps = np.array([fps for fps in model_data['post_tiling_fps'] if fps > 0])
            if len(post_fps) > 0:
                results['mean_post_tiling_fps'] = np.mean(post_fps)
                results['std_post_tiling_fps'] = np.std(post_fps)
                results['min_post_tiling_fps'] = np.min(post_fps)
                results['max_post_tiling_fps'] = np.max(post_fps)
                has_tiling = True
        
        # Tiling activation statistics
        if 'tiling_switch_iterations' in model_data:
            switch_iters = [s for s in model_data['tiling_switch_iterations'] if s is not None]
            if switch_iters:
                results['tiling_activated_count'] = len(switch_iters)
                results['tiling_activation_rate'] = (len(switch_iters) / len(model_data.get('convergence_flags', [])) * 100) if 'convergence_flags' in model_data else 0
                results['mean_tiling_switch_iter'] = np.mean(switch_iters)
                results['std_tiling_switch_iter'] = np.std(switch_iters)
        
        # If no tiling-specific data, try to calculate from iteration times
        if not has_tiling and 'all_iteration_times' in model_data:
            all_fps = []
            
            # Process each sample's iteration times
            for iteration_times in model_data['all_iteration_times']:
                if isinstance(iteration_times, (list, np.ndarray)) and len(iteration_times) > 0:
                    # Calculate mean time per iteration for this sample
                    mean_time = np.mean(iteration_times)
                    if mean_time > 0:
                        fps = 1.0 / mean_time
                        all_fps.append(fps)
            
            if all_fps:
                results['mean_fps'] = np.mean(all_fps)
                results['std_fps'] = np.std(all_fps)
                results['min_fps'] = np.min(all_fps)
                results['max_fps'] = np.max(all_fps)
        
        return results
    
    def calculate_convergence_time(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate convergence time in SECONDS for a model.
        
        This is the KEY metric for comparing speed across methods.
        
        For ART-VS (tiling methods):
            Time = (switch_iter / pre_fps) + ((total_iter - switch_iter) / post_fps)
        
        For non-tiling methods:
            Time = total_iterations / fps
        
        Returns:
            Dict with mean, std, min, max convergence times in seconds
        """
        results = {}
        
        convergence_key = 'convergence_flags' if 'convergence_flags' in model_data else 'converged'
        if convergence_key not in model_data:
            return results
        
        convergence_flags = model_data[convergence_key]
        
        # Need iteration counts
        if 'all_iteration_histories' not in model_data:
            return results
        
        iterations = model_data['all_iteration_histories']
        convergence_times = []
        
        # Check if this is a tiling method (ART-VS)
        has_tiling = ('pre_tiling_fps' in model_data and 'post_tiling_fps' in model_data and 
                      'tiling_switch_iterations' in model_data)
        
        if has_tiling:
            # ART-VS method: two-phase timing
            pre_fps_list = model_data['pre_tiling_fps']
            post_fps_list = model_data['post_tiling_fps']
            switch_iters = model_data['tiling_switch_iterations']
            
            for idx, converged in enumerate(convergence_flags):
                if not converged:
                    continue
                
                try:
                    total_iter = iterations[idx]
                    switch_iter = switch_iters[idx] if switch_iters[idx] is not None else total_iter
                    pre_fps = pre_fps_list[idx] if pre_fps_list[idx] > 0 else 1.0
                    post_fps = post_fps_list[idx] if post_fps_list[idx] > 0 else 1.0
                    
                    # Phase 1: iterations before switch
                    phase1_iters = min(switch_iter, total_iter)
                    phase1_time = phase1_iters / pre_fps
                    
                    # Phase 2: iterations after switch  
                    phase2_iters = max(0, total_iter - switch_iter)
                    phase2_time = phase2_iters / post_fps
                    
                    total_time = phase1_time + phase2_time
                    convergence_times.append(total_time)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating convergence time for sample {idx}: {e}")
                    continue
        else:
            # Non-tiling method: simple calculation
            # Try to get FPS from all_iteration_times or use mean_fps
            fps_stats = self.calculate_fps(model_data)
            
            if 'mean_fps' in fps_stats:
                mean_fps = fps_stats['mean_fps']
                
                for idx, converged in enumerate(convergence_flags):
                    if not converged:
                        continue
                    
                    try:
                        total_iter = iterations[idx]
                        total_time = total_iter / mean_fps
                        convergence_times.append(total_time)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error calculating convergence time for sample {idx}: {e}")
                        continue
            
            elif 'all_iteration_times' in model_data:
                # Use actual iteration times if available
                all_iter_times = model_data['all_iteration_times']
                
                for idx, converged in enumerate(convergence_flags):
                    if not converged:
                        continue
                    
                    try:
                        iter_times = all_iter_times[idx]
                        total_iter = iterations[idx]
                        # Sum the actual iteration times
                        total_time = np.sum(iter_times[:total_iter])
                        convergence_times.append(total_time)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error calculating convergence time for sample {idx}: {e}")
                        continue
        
        # Calculate statistics
        if convergence_times:
            results['mean_convergence_time'] = np.mean(convergence_times)
            results['std_convergence_time'] = np.std(convergence_times)
            results['min_convergence_time'] = np.min(convergence_times)
            results['max_convergence_time'] = np.max(convergence_times)
            results['convergence_times'] = convergence_times  # Store raw data for analysis
        
        return results

    def calculate_length_ratio(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate trajectory length ratio for a model."""
        results = {}
        
        convergence_key = 'convergence_flags' if 'convergence_flags' in model_data else 'converged'
        if convergence_key not in model_data:
            return results
            
        convergence_flags = model_data[convergence_key]
        
        if 'all_position_histories' not in model_data:
            return results
        
        length_ratios = []
        
        # Process each converged sample
        for idx, converged in enumerate(convergence_flags):
            if not converged:
                continue
                
            try:
                positions = model_data['all_position_histories'][idx]
                
                # Calculate actual trajectory length
                actual_length = self._calculate_trajectory_length(positions)
                
                # Calculate geodesic length
                initial_pos = positions[0]
                geodesic_length = np.linalg.norm(self.desired_position - initial_pos)
                
                if geodesic_length > 0:
                    ratio = actual_length / geodesic_length
                    length_ratios.append(ratio)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating length ratio for sample {idx}: {e}")
                continue
        
        # Calculate statistics
        if length_ratios:
            results['mean_length_ratio'] = np.mean(length_ratios)
            results['std_length_ratio'] = np.std(length_ratios)
            results['min_length_ratio'] = np.min(length_ratios)
            results['max_length_ratio'] = np.max(length_ratios)
        
        return results
    
    def evaluate_velocities(self, model_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate velocity profiles for a model."""
        results = {}
        
        # Check for velocity data
        velocity_keys = ['all_average_velocities', 'all_velocity_mean_100', 'all_velocity_mean_10']
        
        for key in velocity_keys:
            if key in model_data:
                velocities = model_data[key]
                if len(velocities) > 0:
                    # Calculate smoothness metrics
                    try:
                        # Flatten if needed
                        if isinstance(velocities[0], (list, np.ndarray)):
                            all_vels = []
                            for vel_array in velocities:
                                if len(vel_array) > 0:
                                    all_vels.extend(vel_array)
                            if all_vels:
                                metric_name = key.replace('all_', '').replace('_', ' ')
                                results[f'mean_{metric_name}'] = np.mean(all_vels)
                                results[f'std_{metric_name}'] = np.std(all_vels)
                    except:
                        pass
        
        return results
    
    def _calculate_position_geodesic(self, initial_pos: np.ndarray, num_steps: int) -> np.ndarray:
        """Calculate position geodesic trajectory."""
        t = np.linspace(0, 1, num_steps)
        return np.array([initial_pos * (1-ti) + self.desired_position * ti for ti in t])
    
    def _calculate_orientation_geodesic(self, initial_quat: np.ndarray, num_steps: int) -> np.ndarray:
        """Calculate orientation geodesic trajectory using SLERP."""
        times = np.linspace(0, 1, num_steps)
        interpolated = []
        
        for t in times:
            # SLERP interpolation
            q1 = initial_quat
            q2 = self.desired_orientation
            
            # Ensure shortest path
            if np.dot(q1, q2) < 0:
                q2 = -q2
            
            # Linear interpolation and normalize
            q = (1-t) * q1 + t * q2
            q = q / np.linalg.norm(q)
            interpolated.append(q)
        
        return np.array(interpolated)
    
    def _calculate_orientation_error(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate angular difference between quaternions in degrees."""
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        relative_rot = r1.inv() * r2
        return np.degrees(relative_rot.magnitude())
    
    def _calculate_trajectory_length(self, positions: np.ndarray) -> float:
        """Calculate total length of trajectory."""
        differences = positions[1:] - positions[:-1]
        segment_lengths = np.linalg.norm(differences, axis=1)
        return np.sum(segment_lengths)
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Optional path to save report as text file
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VISUAL SERVOING EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"File: {os.path.basename(self.npz_file)}")
        
        # Calculate overall initial pose statistics across all models
        all_initial_distances = []
        all_initial_angles = []
        for model_data in self.model_results.values():
            initial_stats = self.evaluate_initial_poses(model_data)
            if initial_stats and 'mean_initial_distance' in initial_stats:
                all_initial_distances.append(initial_stats['mean_initial_distance'])
            if initial_stats and 'mean_initial_angle' in initial_stats:
                all_initial_angles.append(initial_stats['mean_initial_angle'])
        
        if all_initial_distances or all_initial_angles:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("OVERALL INITIAL POSE CONFIGURATION")
            report_lines.append("=" * 80)
            if all_initial_distances:
                mean_dist = np.mean(all_initial_distances)
                std_dist = np.std(all_initial_distances)
                report_lines.append(f"Average Initial Position Error: {mean_dist:.2f} ± {std_dist:.2f} cm")
            if all_initial_angles:
                mean_angle = np.mean(all_initial_angles)
                std_angle = np.std(all_initial_angles)
                report_lines.append(f"Average Initial Orientation Error: {mean_angle:.2f} ± {std_angle:.2f}°")
            report_lines.append("=" * 80)
        
        report_lines.append("")
        
        # Process each model
        for model_id in sorted(self.model_results.keys()):
            model_data = self.model_results[model_id]
            model_name = model_data.get('model_name', f'Model {model_id}')
            
            report_lines.append(f"\n{'-'*60}")
            report_lines.append(f"MODEL {model_id}: {model_name}")
            report_lines.append(f"{'-'*60}")
            
            # Initial pose statistics (MOST IMPORTANT)
            initial_stats = self.evaluate_initial_poses(model_data)
            if initial_stats:
                report_lines.append("\n1. INITIAL POSE STATISTICS (Sampling Configuration):")
                report_lines.append("   " + "="*50)
                if 'mean_initial_distance' in initial_stats:
                    report_lines.append(f"   Initial Position Error: {initial_stats['mean_initial_distance']:.2f} ± "
                                      f"{initial_stats['std_initial_distance']:.2f} cm")
                    report_lines.append(f"      Range: [{initial_stats['min_initial_distance']:.2f}, "
                                      f"{initial_stats['max_initial_distance']:.2f}] cm")
                
                if 'mean_initial_angle' in initial_stats:
                    report_lines.append(f"   Initial Orientation Error: {initial_stats['mean_initial_angle']:.2f} ± "
                                      f"{initial_stats['std_initial_angle']:.2f}°")
                    report_lines.append(f"      Range: [{initial_stats['min_initial_angle']:.2f}, "
                                      f"{initial_stats['max_initial_angle']:.2f}]°")
                report_lines.append("   " + "="*50)
                
                if 'mean_initial_x' in initial_stats:
                    report_lines.append(f"   Mean Initial Position: "
                                      f"X={initial_stats['mean_initial_x']:.3f}, "
                                      f"Y={initial_stats['mean_initial_y']:.3f}, "
                                      f"Z={initial_stats['mean_initial_z']:.3f}")
            
            # Convergence statistics
            conv_stats = self.evaluate_convergence(model_data)
            time_stats = self.calculate_convergence_time(model_data)
            if conv_stats:
                report_lines.append("\n2. CONVERGENCE STATISTICS:")
                report_lines.append(f"   Total samples: {conv_stats.get('total_samples', 0)}")
                report_lines.append(f"   Converged: {conv_stats.get('converged_samples', 0)} "
                                  f"({conv_stats.get('convergence_rate', 0):.1f}%)")
                
                if 'mean_iterations' in conv_stats:
                    report_lines.append(f"   Iterations: {conv_stats['mean_iterations']:.1f} ± "
                                      f"{conv_stats['std_iterations']:.1f} "
                                      f"[{conv_stats['min_iterations']:.0f}, {conv_stats['max_iterations']:.0f}]")
                
                # Add convergence time (KEY METRIC)
                if time_stats and 'mean_convergence_time' in time_stats:
                    report_lines.append(f"   Convergence Time: {time_stats['mean_convergence_time']:.1f} ± "
                                      f"{time_stats['std_convergence_time']:.1f} s "
                                      f"[{time_stats['min_convergence_time']:.1f}, "
                                      f"{time_stats['max_convergence_time']:.1f}]")
            
            # Error statistics
            error_stats = self.evaluate_errors(model_data)
            if error_stats:
                report_lines.append("\n3. ERROR STATISTICS (converged samples):")
                
                if 'mean_position_error' in error_stats:
                    report_lines.append(f"   Position Error: {error_stats['mean_position_error']:.2f} ± "
                                      f"{error_stats['std_position_error']:.2f} cm "
                                      f"[{error_stats['min_position_error']:.2f}, "
                                      f"{error_stats['max_position_error']:.2f}]")
                
                if 'mean_orientation_error' in error_stats:
                    report_lines.append(f"   Orientation Error: {error_stats['mean_orientation_error']:.2f} ± "
                                      f"{error_stats['std_orientation_error']:.2f}° "
                                      f"[{error_stats['min_orientation_error']:.2f}, "
                                      f"{error_stats['max_orientation_error']:.2f}]")
                
                if 'mean_lowest_position' in error_stats:
                    report_lines.append(f"   Lowest Position: {error_stats['mean_lowest_position']:.2f} ± "
                                      f"{error_stats['std_lowest_position']:.2f} cm")
                
                if 'mean_lowest_orientation' in error_stats:
                    report_lines.append(f"   Lowest Orientation: {error_stats['mean_lowest_orientation']:.2f} ± "
                                      f"{error_stats['std_lowest_orientation']:.2f}°")
            
            # APE statistics
            ape_stats = self.calculate_ape(model_data)
            if ape_stats:
                report_lines.append("\n4. ABSOLUTE POSE ERROR (APE):")
                
                if 'mean_position_ape' in ape_stats:
                    report_lines.append(f"   Position APE: {ape_stats['mean_position_ape']:.2f} ± "
                                      f"{ape_stats['std_position_ape']:.2f} cm")
                
                if 'mean_orientation_ape' in ape_stats:
                    report_lines.append(f"   Orientation APE: {ape_stats['mean_orientation_ape']:.2f} ± "
                                      f"{ape_stats['std_orientation_ape']:.2f}°")
            
            # Length ratio statistics
            length_stats = self.calculate_length_ratio(model_data)
            if length_stats:
                report_lines.append("\n5. TRAJECTORY LENGTH RATIO:")
                report_lines.append(f"   Mean Ratio: {length_stats['mean_length_ratio']:.3f} ± "
                                  f"{length_stats['std_length_ratio']:.3f} "
                                  f"[{length_stats['min_length_ratio']:.3f}, "
                                  f"{length_stats['max_length_ratio']:.3f}]")
            
            # FPS statistics
            fps_stats = self.calculate_fps(model_data)
            if fps_stats:
                report_lines.append("\n6. PERFORMANCE (FPS) STATISTICS:")
                
                # Check if tiling mode data is present
                if 'mean_pre_tiling_fps' in fps_stats or 'mean_post_tiling_fps' in fps_stats:
                    # Tiling mode active
                    if 'mean_pre_tiling_fps' in fps_stats:
                        report_lines.append(f"   Pre-Tiling FPS: {fps_stats['mean_pre_tiling_fps']:.1f} ± "
                                          f"{fps_stats.get('std_pre_tiling_fps', 0):.1f} "
                                          f"[{fps_stats.get('min_pre_tiling_fps', 0):.1f}, "
                                          f"{fps_stats.get('max_pre_tiling_fps', 0):.1f}]")
                    
                    if 'mean_post_tiling_fps' in fps_stats:
                        report_lines.append(f"   Post-Tiling FPS: {fps_stats['mean_post_tiling_fps']:.1f} ± "
                                          f"{fps_stats.get('std_post_tiling_fps', 0):.1f} "
                                          f"[{fps_stats.get('min_post_tiling_fps', 0):.1f}, "
                                          f"{fps_stats.get('max_post_tiling_fps', 0):.1f}]")
                    
                    if 'tiling_activation_rate' in fps_stats:
                        report_lines.append(f"   Tiling Activation Rate: {fps_stats['tiling_activation_rate']:.1f}%")
                    
                    if 'mean_tiling_switch_iter' in fps_stats:
                        report_lines.append(f"   Tiling Switch Iteration: {fps_stats['mean_tiling_switch_iter']:.0f} ± "
                                          f"{fps_stats.get('std_tiling_switch_iter', 0):.0f}")
                else:
                    # Non-tiling mode or general FPS
                    if 'mean_fps' in fps_stats:
                        report_lines.append(f"   Overall FPS: {fps_stats['mean_fps']:.1f} ± "
                                          f"{fps_stats.get('std_fps', 0):.1f} "
                                          f"[{fps_stats.get('min_fps', 0):.1f}, "
                                          f"{fps_stats.get('max_fps', 0):.1f}]")
            
            # Velocity statistics
            vel_stats = self.evaluate_velocities(model_data)
            if vel_stats:
                report_lines.append("\n7. VELOCITY STATISTICS:")
                for key, value in vel_stats.items():
                    if 'mean' in key:
                        std_key = key.replace('mean', 'std')
                        if std_key in vel_stats:
                            metric_name = key.replace('mean_', '').replace('_', ' ').title()
                            report_lines.append(f"   {metric_name}: {value:.4f} ± {vel_stats[std_key]:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {save_path}")
        
        return report
    
    def generate_summary_table(self) -> str:
        """Generate a summary table of all models with mean ± std."""
        table_data = []
        
        # Check if we have tiling data to determine headers
        has_tiling_data = any('pre_tiling_fps' in self.model_results.get(model_id, {}) or 
                              'post_tiling_fps' in self.model_results.get(model_id, {})
                              for model_id in self.model_results.keys())
        
        if has_tiling_data:
            headers = ['Model', 'Conv.', 'Time [s]', 'Pos Err [cm]', 'Ori Err [°]', 
                       'Pre-FPS', 'Post-FPS', 'Iter.']
        else:
            headers = ['Model', 'Conv.', 'Time [s]', 'Pos Err [cm]', 'Ori Err [°]', 
                       'FPS', 'Iter.']
        
        for model_id in sorted(self.model_results.keys()):
            model_data = self.model_results[model_id]
            
            conv_stats = self.evaluate_convergence(model_data)
            error_stats = self.evaluate_errors(model_data)
            fps_stats = self.calculate_fps(model_data)
            time_stats = self.calculate_convergence_time(model_data)
            
            # Format with mean ± std where available
            def format_stat(stats, mean_key, std_key=None, fmt=".2f"):
                if not stats or mean_key not in stats:
                    return "N/A"
                mean_val = stats[mean_key]
                if std_key and std_key in stats:
                    std_val = stats[std_key]
                    return f"{mean_val:{fmt}} ± {std_val:{fmt}}"
                return f"{mean_val:{fmt}}"
            
            if has_tiling_data:
                # Table with separate pre/post FPS columns
                row = [
                    f"Model {model_id}",
                    f"{conv_stats.get('convergence_rate', 0):.0f}%" if conv_stats else "N/A",
                    format_stat(time_stats, 'mean_convergence_time', 'std_convergence_time', ".1f"),
                    format_stat(error_stats, 'mean_position_error', 'std_position_error', ".2f"),
                    format_stat(error_stats, 'mean_orientation_error', 'std_orientation_error', ".2f"),
                    format_stat(fps_stats, 'mean_pre_tiling_fps', 'std_pre_tiling_fps', ".1f") if 'mean_pre_tiling_fps' in fps_stats else format_stat(fps_stats, 'mean_fps', 'std_fps', ".1f"),
                    format_stat(fps_stats, 'mean_post_tiling_fps', 'std_post_tiling_fps', ".1f") if 'mean_post_tiling_fps' in fps_stats else "N/A",
                    format_stat(conv_stats, 'mean_iterations', 'std_iterations', ".0f"),
                ]
            else:
                # Table with single FPS column
                row = [
                    f"Model {model_id}",
                    f"{conv_stats.get('convergence_rate', 0):.0f}%" if conv_stats else "N/A",
                    format_stat(time_stats, 'mean_convergence_time', 'std_convergence_time', ".1f"),
                    format_stat(error_stats, 'mean_position_error', 'std_position_error', ".2f"),
                    format_stat(error_stats, 'mean_orientation_error', 'std_orientation_error', ".2f"),
                    format_stat(fps_stats, 'mean_fps', 'std_fps', ".1f"),
                    format_stat(conv_stats, 'mean_iterations', 'std_iterations', ".0f"),
                ]
            table_data.append(row)
        
        return tabulate(table_data, headers=headers, tablefmt='grid')


def main():
    parser = argparse.ArgumentParser(
        description='Unified evaluation of visual servoing results',
        epilog='Examples:\n'
               '  # Single NPZ file:\n'
               '  python3 unified_eval.py results.npz --table\n\n'
               '  # Multiple NPZ files:\n'
               '  python3 unified_eval.py results_*.npz --compare\n\n'
               '  # Folder of NPZ files:\n'
               '  python3 unified_eval.py /path/to/folder/*.npz --table',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('npz_files', type=str, nargs='+', 
                       help='Path to NPZ results file(s) or pattern (e.g., results_*.npz)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    parser.add_argument('--save-report', '-s', type=str, help='Save report to file')
    parser.add_argument('--table', '-t', action='store_true', help='Show summary table')
    parser.add_argument('--compare', '-c', action='store_true', 
                       help='Compare multiple files (shows comparison table)')
    
    args = parser.parse_args()
    
    # Expand glob patterns and collect all NPZ files
    import glob
    all_files = []
    for pattern in args.npz_files:
        matched_files = glob.glob(pattern)
        if matched_files:
            all_files.extend(matched_files)
        elif os.path.exists(pattern):  # Single file without glob
            all_files.append(pattern)
        else:
            print(f"Warning: No files matched pattern '{pattern}'")
    
    if not all_files:
        print("Error: No valid NPZ files found.")
        sys.exit(1)
    
    # Remove duplicates while preserving order
    all_files = list(dict.fromkeys(all_files))
    
    print(f"Found {len(all_files)} NPZ file(s) to evaluate")
    
    if args.compare and len(all_files) > 1:
        # Comparison mode for multiple files
        print("\n" + "="*80)
        print("COMPARISON MODE - EVALUATING MULTIPLE FILES")
        print("="*80)
        
        comparison_data = []
        for npz_file in sorted(all_files):
            print(f"\nProcessing: {os.path.basename(npz_file)}")
            try:
                evaluator = UnifiedEvaluator(npz_file, verbose=args.verbose)
                
                # Get overall statistics
                overall_stats = {
                    'File': os.path.basename(npz_file),
                    'Models': len(evaluator.model_results),
                    'Total Samples': 0,
                    'Overall Conv.': 0,
                    'Mean Pos Err': [],
                    'Std Pos Err': [],
                    'Mean Ori Err': [],
                    'Std Ori Err': [],
                    'Mean Conv Time': [],
                    'Std Conv Time': [],
                    'Pre FPS': [],
                    'Post FPS': [],
                    'General FPS': []
                }
                
                for model_id, model_data in evaluator.model_results.items():
                    conv_stats = evaluator.evaluate_convergence(model_data)
                    error_stats = evaluator.evaluate_errors(model_data)
                    fps_stats = evaluator.calculate_fps(model_data)
                    time_stats = evaluator.calculate_convergence_time(model_data)
                    
                    if conv_stats:
                        overall_stats['Total Samples'] += conv_stats.get('total_samples', 0)
                        overall_stats['Overall Conv.'] += conv_stats.get('converged_samples', 0)
                    
                    if error_stats and 'mean_position_error' in error_stats:
                        overall_stats['Mean Pos Err'].append(error_stats['mean_position_error'])
                        overall_stats['Std Pos Err'].append(error_stats.get('std_position_error', 0))
                    
                    if error_stats and 'mean_orientation_error' in error_stats:
                        overall_stats['Mean Ori Err'].append(error_stats['mean_orientation_error'])
                        overall_stats['Std Ori Err'].append(error_stats.get('std_orientation_error', 0))
                    
                    # Collect convergence time
                    if time_stats and 'mean_convergence_time' in time_stats:
                        overall_stats['Mean Conv Time'].append(time_stats['mean_convergence_time'])
                        overall_stats['Std Conv Time'].append(time_stats.get('std_convergence_time', 0))
                    
                    # Collect FPS separately for pre-tiling, post-tiling, and general
                    if fps_stats:
                        if 'mean_pre_tiling_fps' in fps_stats:
                            overall_stats['Pre FPS'].append(fps_stats['mean_pre_tiling_fps'])
                        if 'mean_post_tiling_fps' in fps_stats:
                            overall_stats['Post FPS'].append(fps_stats['mean_post_tiling_fps'])
                        if 'mean_fps' in fps_stats:
                            overall_stats['General FPS'].append(fps_stats['mean_fps'])
                
                # Calculate averages with std
                pos_err_str = "N/A"
                if overall_stats['Mean Pos Err']:
                    mean_pos = np.mean(overall_stats['Mean Pos Err'])
                    std_pos = np.mean(overall_stats['Std Pos Err'])
                    pos_err_str = f"{mean_pos:.2f} ± {std_pos:.2f}"
                
                ori_err_str = "N/A"
                if overall_stats['Mean Ori Err']:
                    mean_ori = np.mean(overall_stats['Mean Ori Err'])
                    std_ori = np.mean(overall_stats['Std Ori Err'])
                    ori_err_str = f"{mean_ori:.2f} ± {std_ori:.2f}"
                
                # Convergence time (THE KEY METRIC)
                conv_time_str = "N/A"
                if overall_stats['Mean Conv Time']:
                    mean_time = np.mean(overall_stats['Mean Conv Time'])
                    std_time = np.mean(overall_stats['Std Conv Time'])
                    conv_time_str = f"{mean_time:.1f} ± {std_time:.1f}"
                
                # FPS info (for reference)
                fps_str = "N/A"
                if overall_stats['Pre FPS'] and overall_stats['Post FPS']:
                    # Tiling mode - show both
                    pre_fps = np.mean(overall_stats['Pre FPS'])
                    post_fps = np.mean(overall_stats['Post FPS'])
                    fps_str = f"{pre_fps:.0f}/{post_fps:.0f}"
                elif overall_stats['General FPS']:
                    fps_str = f"{np.mean(overall_stats['General FPS']):.0f}"
                
                comparison_row = [
                    overall_stats['File'][:40],  # Truncate long filenames
                    f"{overall_stats['Total Samples']}",
                    f"{(overall_stats['Overall Conv.'] / overall_stats['Total Samples'] * 100):.1f}%" if overall_stats['Total Samples'] > 0 else "N/A",
                    conv_time_str,
                    pos_err_str,
                    ori_err_str,
                    fps_str
                ]
                comparison_data.append(comparison_row)
                
            except Exception as e:
                print(f"  Error processing file: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print comparison table
        if comparison_data:
            headers = ['File', 'Samples', 'Conv.%', 'Time [s]', 'Pos Err [cm]', 'Ori Err [°]', 'FPS']
            print("\n" + tabulate(comparison_data, headers=headers, tablefmt='grid'))
            print("\nNote: For tiling methods, FPS shows Pre/Post values")
    
    else:
        # Single file or sequential processing mode
        for npz_file in all_files:
            if len(all_files) > 1:
                print(f"\n{'='*80}")
                print(f"EVALUATING: {os.path.basename(npz_file)}")
                print(f"{'='*80}")
            
            # Check file exists
            if not os.path.exists(npz_file):
                print(f"Error: File '{npz_file}' does not exist.")
                continue
            
            # Create evaluator
            evaluator = UnifiedEvaluator(npz_file, verbose=args.verbose)
            
            # Generate and print report
            save_path = None
            if args.save_report:
                if len(all_files) > 1:
                    # Multiple files: append filename to save path
                    base, ext = os.path.splitext(args.save_report)
                    npz_base = os.path.splitext(os.path.basename(npz_file))[0]
                    save_path = f"{base}_{npz_base}{ext}"
                else:
                    save_path = args.save_report
            
            report = evaluator.generate_report(save_path=save_path)
            print(report)
            
            # Show summary table if requested
            if args.table:
                print("\n" + "="*80)
                print("SUMMARY TABLE")
                print("="*80)
                print(evaluator.generate_summary_table())


if __name__ == "__main__":
    main()
