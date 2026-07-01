import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List

class APECalculator:
    def __init__(self, npz_file: str):
        self.desired_position = np.array([0, 0, 0.61])
        self.desired_orientation = np.array([0, 0.7071068, 0, 0.7071068])
        
        self.data = np.load(npz_file, allow_pickle=True)
        self.convergence_flags = self.data['convergence_flags']
        self.position_histories = self.data['all_position_histories']
        self.orientation_histories = self.data['all_orientation_histories']
        self.iteration_histories = self.data['all_iteration_histories']
        self.initial_orientations = self.data['initial_orientations']  # Added this for comparison

        # Print basic info about the data
        print(f"Total samples: {len(self.convergence_flags)}")
        print(f"Converged samples: {np.sum(self.convergence_flags)}")
        
        # Let's examine the first converged sample
        first_converged_idx = np.where(self.convergence_flags)[0][0]
        print(f"\nExamining first converged sample (index {first_converged_idx}):")
        print(f"Initial orientation from data: {self.initial_orientations[first_converged_idx]}")
        print(f"First orientation from history: {self.orientation_histories[first_converged_idx][0]}")
        print(f"Number of iterations: {self.iteration_histories[first_converged_idx]}")

    def calculate_position_geodesic(self, initial_pos: np.ndarray, num_steps: int) -> np.ndarray:
        """Calculate the position geodesic (straight line) trajectory."""
        t = np.linspace(0, 1, num_steps)
        return np.array([initial_pos * (1-ti) + self.desired_position * ti for ti in t])

    def calculate_orientation_geodesic(self, initial_quat: np.ndarray, num_steps: int) -> np.ndarray:
        """Calculate the orientation geodesic trajectory."""
        times = np.linspace(0, 1, num_steps)
        
        interpolated = []
        for t in times:
            # Calculate weighted sum of quaternions
            w1 = 1 - t
            w2 = t
            q1 = initial_quat
            q2 = self.desired_orientation
            
            # Ensure we're taking the shortest path
            if np.dot(q1, q2) < 0:
                q2 = -q2
                
            # Normalize and store the interpolated quaternion
            q = w1 * q1 + w2 * q2
            q = q / np.linalg.norm(q)
            interpolated.append(q)
            
        return np.array(interpolated)

    def calculate_orientation_error(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate the angular difference between two quaternions in degrees."""
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        # Calculate the relative rotation and convert to angle in degrees
        relative_rot = r1.inv() * r2
        return np.degrees(relative_rot.magnitude())

    def calculate_sample_ape(self, sample_idx: int) -> Tuple[float, float]:
        """Calculate APE for a single sample with debug information."""
        num_iterations = self.iteration_histories[sample_idx]
        actual_positions = self.position_histories[sample_idx]
        actual_orientations = self.orientation_histories[sample_idx]
        
        if sample_idx < 5:  # Print debug info for first 5 samples
            print(f"\nProcessing sample {sample_idx}:")
            print(f"Number of iterations: {num_iterations}")
            print(f"Initial position: {actual_positions[0]}")
            print(f"Initial orientation (history): {actual_orientations[0]}")
            print(f"Initial orientation (stored): {self.initial_orientations[sample_idx]}")
        
        initial_position = actual_positions[0]
        initial_orientation = actual_orientations[0]
        
        geodesic_positions = self.calculate_position_geodesic(initial_position, num_iterations)
        geodesic_orientations = self.calculate_orientation_geodesic(initial_orientation, num_iterations)
        
        position_errors = []
        orientation_errors = []
        
        for i in range(num_iterations):
            pos_error = np.linalg.norm(actual_positions[i] - geodesic_positions[i]) * 100
            orient_error = self.calculate_orientation_error(actual_orientations[i], geodesic_orientations[i])
            
            position_errors.append(pos_error)
            orientation_errors.append(orient_error)
            
            # Print some sample errors for the first sample
            if sample_idx == 0 and i < 5:
                print(f"\nIteration {i}:")
                print(f"Position error: {pos_error:.2f} cm")
                print(f"Orientation error: {orient_error:.2f} degrees")
        
        mean_pos_error = np.mean(position_errors)
        mean_orient_error = np.mean(orientation_errors)
        
        if sample_idx < 5:  # Print final results for first 5 samples
            print(f"\nFinal results for sample {sample_idx}:")
            print(f"Mean position error: {mean_pos_error:.2f} cm")
            print(f"Mean orientation error: {mean_orient_error:.2f} degrees")
        
        return mean_pos_error, mean_orient_error

    def calculate_overall_ape(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate overall APE statistics for all converged samples."""
        position_apes = []
        orientation_apes = []
        
        # Process only converged samples
        for idx, converged in enumerate(self.convergence_flags):
            if converged:
                try:
                    pos_ape, orient_ape = self.calculate_sample_ape(idx)
                    position_apes.append(pos_ape)
                    orientation_apes.append(orient_ape)
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    continue
        
        # Convert to numpy arrays for statistical calculations
        position_apes = np.array(position_apes)
        orientation_apes = np.array(orientation_apes)
        
        # Print some statistics about the APEs
        print("\nAPE Statistics:")
        print(f"Number of successfully processed samples: {len(position_apes)}")
        print(f"Position APE range: {np.min(position_apes):.2f} to {np.max(position_apes):.2f} cm")
        print(f"Orientation APE range: {np.min(orientation_apes):.2f} to {np.max(orientation_apes):.2f} degrees")
        
        # Calculate mean and standard deviation
        pos_mean, pos_std = np.mean(position_apes), np.std(position_apes)
        orient_mean, orient_std = np.mean(orientation_apes), np.std(orientation_apes)
        
        return (pos_mean, pos_std), (orient_mean, orient_std)


def print_sample_data(npz_file: str, num_samples: int = 5):
    """Print detailed data for the first few samples."""
    data = np.load(npz_file, allow_pickle=True)
    convergence_flags = data['convergence_flags']
    orientation_histories = data['all_orientation_histories']
    initial_orientations = data['initial_orientations']
    
    converged_samples = np.where(convergence_flags)[0][:num_samples]
    
    print(f"\nDetailed analysis of first {num_samples} converged samples:")
    print("="*50)
    
    for idx in converged_samples:
        print(f"\nSample {idx}:")
        print("-" * 20)
        print(f"Initial orientation (stored):")
        print(initial_orientations[idx])
        print(f"\nFirst orientation from history:")
        print(orientation_histories[idx][0])
        print(f"\nFirst 5 orientations from history:")
        for i, orientation in enumerate(orientation_histories[idx][:5]):
            print(f"Step {i}: {orientation}")
        print("-" * 50)


def main():
    npz_file = 'results_config_dino_hybrid_standard_1.npz'
    
    # First, print detailed sample data
    print("Analyzing sample data...")
    print_sample_data(npz_file)
    
    # Then calculate APE
    print("\nCalculating APE...")
    calculator = APECalculator(npz_file)
    (pos_mean, pos_std), (orient_mean, orient_std) = calculator.calculate_overall_ape()
    
    print("\nFinal Results:")
    print("=============")
    print(f"Position APE: {pos_mean:.2f} ± {pos_std:.2f} cm")
    print(f"Orientation APE: {orient_mean:.2f} ± {orient_std:.2f}°")
    print(f"Number of converged samples: {np.sum(calculator.convergence_flags)}")


if __name__ == "__main__":
    main()
