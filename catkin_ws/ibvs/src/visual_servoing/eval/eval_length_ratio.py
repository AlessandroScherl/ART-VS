import numpy as np
from scipy.spatial.transform import Rotation as R

class LengthRatioCalculator:
    def __init__(self, npz_file: str):
        """Initialize with data from NPZ file."""
        self.desired_position = np.array([0, 0, 0.61])
        
        # Load data
        self.data = np.load(npz_file, allow_pickle=True)
        self.convergence_flags = self.data['convergence_flags']
        self.position_histories = self.data['all_position_histories']
        self.iteration_histories = self.data['all_iteration_histories']

    def calculate_actual_trajectory_length(self, positions: np.ndarray) -> float:
        """
        Calculate the length of the actual trajectory taken.
        Args:
            positions: Array of positions representing the trajectory
        Returns:
            Total length of the trajectory
        """
        # Calculate distances between consecutive points
        differences = positions[1:] - positions[:-1]
        # Calculate length of each segment
        segment_lengths = np.linalg.norm(differences, axis=1)
        # Sum all segments to get total length
        return np.sum(segment_lengths)

    def calculate_geodesic_length(self, start_pos: np.ndarray) -> float:
        """
        Calculate the length of the geodesic (straight line) path.
        Args:
            start_pos: Starting position
        Returns:
            Length of the straight line to the goal
        """
        # For position, geodesic is simply the direct distance
        return np.linalg.norm(self.desired_position - start_pos)

    def calculate_sample_length_ratio(self, sample_idx: int) -> float:
        """
        Calculate length ratio for a single sample.
        Args:
            sample_idx: Index of the sample to process
        Returns:
            Length ratio for this sample
        """
        # Get the position history for this sample
        positions = self.position_histories[sample_idx]
        initial_position = positions[0]
        
        # Calculate both lengths
        actual_length = self.calculate_actual_trajectory_length(positions)
        geodesic_length = self.calculate_geodesic_length(initial_position)
        
        # Calculate ratio
        return actual_length / geodesic_length

    def calculate_overall_length_ratios(self):
        """Calculate length ratios for all converged samples."""
        length_ratios = []
        
        # Process only converged samples
        for idx, converged in enumerate(self.convergence_flags):
            if converged:
                try:
                    ratio = self.calculate_sample_length_ratio(idx)
                    length_ratios.append(ratio)
                    
                    # Print detailed info for first few samples
                    if len(length_ratios) <= 5:
                        positions = self.position_histories[idx]
                        actual_length = self.calculate_actual_trajectory_length(positions)
                        geodesic_length = self.calculate_geodesic_length(positions[0])
                        print(f"\nSample {idx}:")
                        print(f"Actual trajectory length: {actual_length:.3f} m")
                        print(f"Geodesic length: {geodesic_length:.3f} m")
                        print(f"Length ratio: {ratio:.3f}")
                        
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    continue
        
        length_ratios = np.array(length_ratios)
        
        return {
            'mean': np.mean(length_ratios),
            'std': np.std(length_ratios),
            'min': np.min(length_ratios),
            'max': np.max(length_ratios),
            'num_samples': len(length_ratios)
        }

def main():
    calculator = LengthRatioCalculator('results_config_standard06_dino_standard.npz')
    results = calculator.calculate_overall_length_ratios()
    
    print("\nOverall Length Ratio Results:")
    print("============================")
    print(f"Mean ± Std: {results['mean']:.3f} ± {results['std']:.3f}")
    print(f"Range: {results['min']:.3f} to {results['max']:.3f}")
    print(f"Number of converged samples processed: {results['num_samples']}")

if __name__ == "__main__":
    main()
