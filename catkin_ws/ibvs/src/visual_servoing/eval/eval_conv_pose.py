import numpy as np
import os
import argparse

def print_npz_contents(file_path):
    """
    Read and print the contents of an NPZ file, showing at least two samples for each element.
    Calculate statistics for converged samples including mean and standard deviation of position and orientation errors.
    Analyze average velocities at specific points in the visual servoing process.
    
    Args:
    file_path (str): Path to the NPZ file
    """
    # Print the name of the NPZ file being analyzed
    print(f"Analyzing NPZ file: {os.path.basename(file_path)}")
    print("=" * 40)

    # Load the NPZ file
    data = np.load(file_path, allow_pickle=True)
    
    print("Contents of the NPZ file:")
    print("-------------------------")
    
    # Initialize variables to store relevant data
    convergence_flags = None
    position_errors = None
    orientation_errors = None
    lowest_position_errors = None
    lowest_orientation_errors = None
    all_iteration_histories = None
    all_average_velocities = None
    all_velocity_mean_100 = None
    all_velocity_mean_10 = None

    for key in data.files:
        print(f"\nKey: {key}")
        value = data[key]
        
        # Store relevant data for later analysis
        if key == "convergence_flags":
            convergence_flags = value
        elif key == "position_errors":
            position_errors = value
        elif key == "orientation_errors":
            orientation_errors = value
        elif key == "lowest_position_errors":
            lowest_position_errors = value
        elif key == "lowest_orientation_errors":
            lowest_orientation_errors = value
        elif key == "all_iteration_histories":
            all_iteration_histories = value
        elif key == "all_average_velocities":
            all_average_velocities = value
        elif key == "all_velocity_mean_100":
            all_velocity_mean_100 = value
        elif key == "all_velocity_mean_10":
            all_velocity_mean_10 = value

    # Calculate statistics for converged samples
    if convergence_flags is not None and position_errors is not None and orientation_errors is not None:
        converged_mask = convergence_flags == True
        converged_position_errors = position_errors[converged_mask]
        converged_orientation_errors = orientation_errors[converged_mask]

        total_samples = len(convergence_flags)
        converged_samples = np.sum(converged_mask)
        convergence_percentage = (converged_samples / total_samples) * 100

        print("\nConvergence Statistics:")
        print(f"Converged samples: {converged_samples} out of {total_samples} ({convergence_percentage:.2f}%)")
        

        if lowest_position_errors is not None and lowest_orientation_errors is not None:
            converged_lowest_position_errors = lowest_position_errors[converged_mask]
            converged_lowest_orientation_errors = lowest_orientation_errors[converged_mask]
            
            print("\nLowest Position Errors (for converged samples):")
            print(f"Mean: {np.mean(converged_lowest_position_errors):.6f}")
            print(f"Standard Deviation: {np.std(converged_lowest_position_errors):.6f}")
            
            print("\nLowest Orientation Errors (for converged samples):")
            print(f"Mean: {np.mean(converged_lowest_orientation_errors):.6f}")
            print(f"Standard Deviation: {np.std(converged_lowest_orientation_errors):.6f}")

        if all_iteration_histories is not None:
            converged_iteration_histories = all_iteration_histories[converged_mask]
            print("\nIteration Statistics (for converged samples):")
            print(f"Mean iterations: {np.mean(converged_iteration_histories):.2f}")
            print(f"Standard Deviation of iterations: {np.std(converged_iteration_histories):.2f}")

    if "total_execution_time" in data:
        print(f"\nTotal Execution Time: {data['total_execution_time']:.2f} seconds")

    # Velocity Analysis
    if all_average_velocities is not None and convergence_flags is not None:
        print("\nVelocity Analysis:")
        print("------------------")

def main():
    parser = argparse.ArgumentParser(description='Analyze contents of an NPZ file')
    parser.add_argument('file_path', type=str, help='Path to the NPZ file')
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        return

    if not args.file_path.endswith('.npz'):
        print(f"Warning: File '{args.file_path}' does not have .npz extension.")

    try:
        print_npz_contents(args.file_path)
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
