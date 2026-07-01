#!/usr/bin/env python3
"""
Measure Peak VRAM Usage for TEST1 ViT Methods

This script measures peak GPU VRAM usage for the 6 Vision Transformer methods
in both non-perturbed and perturbed modes (12 runs total, 4 samples each).

NOTE: Classical methods (SIFT/ORB/AKAZE) are skipped - they run on CPU.

Usage:
    # Terminal 1: Start Gazebo simulation (REQUIRED)
    cd catkin_ws/ibvs/src && ./run_ibvs.sh

    # Terminal 2: Run VRAM measurement
    cd catkin_ws/ibvs/src/visual_servoing/experiments/paper_evaluation
    python3 measure_vram_test1.py

    # Results saved to: vram_results_test1.csv
"""

import subprocess
import threading
import time
import csv
import os
import sys
from datetime import datetime

# Try to import pynvml for VRAM monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[WARNING] pynvml not available, will use nvidia-smi fallback")

# Try to import torch for cache clearing
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] torch not available, GPU cache clearing disabled")


class VRAMMonitor:
    """Thread-based VRAM monitoring with peak tracking."""

    def __init__(self, gpu_index=0, sample_interval=0.1):
        """
        Initialize VRAM monitor.

        Args:
            gpu_index: GPU device index to monitor
            sample_interval: Sampling interval in seconds (default 100ms)
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.peak_vram = 0
        self.running = False
        self.thread = None

        # Initialize pynvml if available
        if PYNVML_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def start(self):
        """Start VRAM monitoring in background thread."""
        self.peak_vram = 0
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def _monitor_loop(self):
        """Main monitoring loop - runs in separate thread."""
        while self.running:
            try:
                if PYNVML_AVAILABLE:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    current_vram = info.used
                else:
                    # Fallback to nvidia-smi
                    current_vram = self._get_vram_nvidia_smi()

                self.peak_vram = max(self.peak_vram, current_vram)
            except Exception as e:
                print(f"[VRAM Monitor] Error: {e}")

            time.sleep(self.sample_interval)

    def _get_vram_nvidia_smi(self):
        """Fallback VRAM measurement using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            # Returns value in MB, convert to bytes
            return int(result.stdout.strip()) * 1024 * 1024
        except Exception:
            return 0

    def stop(self):
        """Stop monitoring and return peak VRAM in MB."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return self.peak_vram / (1024 * 1024)  # Convert bytes to MB

    def __del__(self):
        """Cleanup pynvml on destruction."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def clear_gpu_cache():
    """Clear GPU cache between runs for isolated measurements."""
    if TORCH_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)  # Brief pause to ensure memory is freed
        print("[GPU] Cache cleared")
    else:
        print("[GPU] Cache clearing skipped (torch not available)")


def run_experiment(config_path, output_path, samples=4, perturbation=False):
    """
    Run a single experiment via subprocess.

    Args:
        config_path: Path to config YAML file
        output_path: Path for output NPZ file
        samples: Number of samples to run
        perturbation: Whether to use perturbation mode

    Returns:
        True if successful, False otherwise
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.dirname(script_dir)

    # Build command
    cmd = [
        'python3',
        os.path.join(experiments_dir, 'run_multi_model_experiment_with_hollywood.py'),
        '--config', config_path,
        '--models', '99',
        '--samples', str(samples),
        '--output', output_path,
        '--iteration-display-freq', '100'  # Reduce output spam
    ]

    if perturbation:
        cmd.append('--perturbation')

    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per run (tiling + perturbation can be slow)
        )

        if result.returncode != 0:
            print(f"[ERROR] Experiment failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
            return False
        return True

    except subprocess.TimeoutExpired:
        print("[ERROR] Experiment timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] Exception running experiment: {e}")
        return False


# Define the 6 ViT TEST1 methods with their config files
# NOTE: SIFT/ORB/AKAZE skipped - they run on CPU, no VRAM usage
TEST1_METHODS = [
    {
        'name': 'DINOv2-308',
        'short_name': 'dinov2_308',
        'config': 'config_dinov2_308_no_tiling.yaml'
    },
    {
        'name': 'DINOv2-518',
        'short_name': 'dinov2_518',
        'config': 'config_dinov2_518_no_tiling.yaml'
    },
    {
        'name': 'DINOv2-224-7x7',
        'short_name': 'dinov2_224_7x7',
        'config': 'config_dinov2_224_7x7_tiling.yaml'
    },
    {
        'name': 'DINOv3-256-6x6',
        'short_name': 'dinov3_256_6x6',
        'config': 'config_dinov3_256_6x6_tiling.yaml'
    },
    {
        'name': 'DINOv3-1440',
        'short_name': 'dinov3_1440',
        'config': 'config_dinov3_1440_no_tiling.yaml'
    },
    {
        'name': 'AM-RADIO-256-6x6',
        'short_name': 'amradio_256_6x6',
        'config': 'config_amradio_256_6x6_tiling.yaml'
    }
]


def main():
    """Main function to run VRAM measurements for all TEST1 methods."""

    print("=" * 60)
    print("     TEST1 Peak VRAM Usage Measurement (ViT Only)")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ViT Methods: {len(TEST1_METHODS)}")
    print(f"  - DINOv2-308, DINOv2-518, DINOv2-224-7x7")
    print(f"  - DINOv3-256-6x6, DINOv3-1440, AM-RADIO-256-6x6")
    print(f"Modes: Non-Perturbed + Perturbed")
    print(f"Samples per run: 4")
    print(f"Total runs: {len(TEST1_METHODS) * 2}")
    print(f"Skipped: SIFT, ORB, AKAZE (CPU-only)")
    print("=" * 60)
    print()

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'configs', 'hollywood_500_evaluation')

    # Create results directory
    results_dir = os.path.join(script_dir, 'results', 'vram_measurement')
    os.makedirs(results_dir, exist_ok=True)

    # Store results
    results = []

    # Initialize VRAM monitor
    monitor = VRAMMonitor()

    # Run each method
    for method_idx, method in enumerate(TEST1_METHODS, 1):
        method_name = method['name']
        config_file = method['config']
        config_path = os.path.join(configs_dir, config_file)

        # Check config exists
        if not os.path.exists(config_path):
            print(f"[ERROR] Config not found: {config_path}")
            results.append({
                'method': method_name,
                'non_perturbed_mb': -1,
                'perturbed_mb': -1,
                'error': 'Config not found'
            })
            continue

        method_results = {
            'method': method_name,
            'non_perturbed_mb': 0,
            'perturbed_mb': 0,
            'error': None
        }

        # Test both modes
        for perturbation in [False, True]:
            mode_name = "Perturbed" if perturbation else "Non-Perturbed"
            mode_key = "perturbed_mb" if perturbation else "non_perturbed_mb"

            print(f"\n[{method_idx}/{len(TEST1_METHODS)}] {method_name} - {mode_name}")
            print("-" * 50)

            # Clear GPU cache
            clear_gpu_cache()

            # Create output path
            output_path = os.path.join(
                results_dir,
                f"temp_{method['short_name']}_{'perturbed' if perturbation else 'normal'}.npz"
            )

            # Start VRAM monitoring
            monitor.start()

            # Run experiment
            success = run_experiment(
                config_path=config_path,
                output_path=output_path,
                samples=4,
                perturbation=perturbation
            )

            # Stop monitoring and get peak VRAM
            peak_vram = monitor.stop()

            if success:
                method_results[mode_key] = peak_vram
                print(f"✓ Peak VRAM: {peak_vram:.0f} MB")
            else:
                method_results[mode_key] = -1
                method_results['error'] = f"Failed in {mode_name} mode"
                print(f"✗ FAILED")

            # Clean up temp file
            if os.path.exists(output_path):
                os.remove(output_path)

        results.append(method_results)

    # Print results table
    print("\n")
    print("=" * 60)
    print("         TEST1 Peak VRAM Usage Results")
    print("=" * 60)
    print()
    print(f"| {'Method':<18} | {'Non-Perturbed (MB)':>18} | {'Perturbed (MB)':>14} |")
    print(f"|{'-'*20}|{'-'*20}|{'-'*16}|")

    for r in results:
        non_pert = f"{r['non_perturbed_mb']:.0f}" if r['non_perturbed_mb'] > 0 else "ERROR"
        pert = f"{r['perturbed_mb']:.0f}" if r['perturbed_mb'] > 0 else "ERROR"
        print(f"| {r['method']:<18} | {non_pert:>18} | {pert:>14} |")

    print()

    # Save to CSV
    csv_path = os.path.join(script_dir, 'vram_results_test1.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'non_perturbed_mb', 'perturbed_mb'])
        for r in results:
            writer.writerow([
                r['method'],
                r['non_perturbed_mb'] if r['non_perturbed_mb'] > 0 else 'ERROR',
                r['perturbed_mb'] if r['perturbed_mb'] > 0 else 'ERROR'
            ])

    print(f"Results saved to: {csv_path}")
    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
