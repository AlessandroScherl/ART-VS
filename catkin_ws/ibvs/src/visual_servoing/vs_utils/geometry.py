import numpy as np
from scipy.spatial.transform import Rotation as R


def sample_camera_positions(volume_dimensions, num_samples, desired_position):
    """
    Sample random camera positions within a specified volume.

    Args:
        volume_dimensions (np.ndarray): The dimensions of the volume for sampling (width, height, depth).
        num_samples (int): The number of samples to generate.
        desired_position (np.ndarray): The desired central position to offset the samples from.

    Returns:
        np.ndarray: An array of sampled camera positions.
    """
    # Offset the volume to be centered around the desired position
    half_dims = volume_dimensions / 2
    min_bounds = desired_position - half_dims
    max_bounds = desired_position + half_dims

    # Sample positions uniformly within the defined bounds
    positions = np.random.uniform(min_bounds, max_bounds, size=(num_samples, 3))
    return positions


def sample_focal_points_original(num_samples, reference_point, num_circles, circle_radius_aug):
    """
    Sample focal points based on the original implementation in PoseLookingAtSamePointWithNoiseAndRotationZGenerator.

    Args:
        num_samples (int): The total number of samples to generate.
        reference_point (np.ndarray): The reference point (3D vector: [x, y, z]).
        num_circles (int): Number of circles to generate points on.
        circle_radius_aug (float): Radius augmentation factor for circles.

    Returns:
        np.ndarray: An array of sampled focal points (shape: [num_samples, 3]).
    """
    samples_per_circle = num_samples // num_circles
    looked_at_points = np.empty((num_samples, 3))

    for cn in range(num_circles):
        radius = circle_radius_aug * (cn + 1)
        istart = cn * samples_per_circle

        # Sample points on a circle
        rand_theta = np.random.uniform(-np.pi, np.pi, size=samples_per_circle)
        x = np.cos(rand_theta) * radius + reference_point[0]
        y = np.sin(rand_theta) * radius + reference_point[1]
        z = np.repeat(reference_point[2], samples_per_circle)

        points = np.column_stack((x, y, z))
        looked_at_points[istart: istart + samples_per_circle] = points

    return looked_at_points


def calculate_position_error(positions, desired_position):
    """
    Calculate the position error as the Euclidean distance from the desired position.

    Args:
        positions (np.ndarray): The sampled camera positions.
        desired_position (np.ndarray): The desired central position.

    Returns:
        tuple: The average error and standard deviation of the error in centimeters.
    """
    errors = np.linalg.norm(positions - desired_position, axis=1)
    average_error = np.mean(errors) * 100  # Convert to centimeters
    std_deviation = np.std(errors) * 100  # Convert to centimeters
    return average_error, std_deviation


def calculate_orientation_error(quaternion_list, desired_orientation):
    """
    Calculate orientation errors between current and desired orientations.
    
    Args:
        quaternion_list: Array of quaternions [x, y, z, w]
        desired_orientation: Desired orientation quaternion [x, y, z, w]
    
    Returns:
        tuple: (mean_error, std_dev_error) in degrees
    """
    # Ensure quaternion_list is a numpy array
    quaternion_list = np.array(quaternion_list)

    # Convert desired_orientation to a Rotation object
    desired_rotation = R.from_quat(desired_orientation)

    errors = []

    for quaternion in quaternion_list:
        # Convert the current quaternion to a Rotation object
        current_rotation = R.from_quat(quaternion)

        # Calculate the relative rotation from current to desired
        relative_rotation = current_rotation.inv() * desired_rotation

        # Get the angle of rotation (in radians)
        angle = relative_rotation.magnitude()

        # Convert to degrees
        error_degrees = np.degrees(angle)

        errors.append(error_degrees)

    errors = np.array(errors)

    # Calculate mean and standard deviation of the errors
    mean_error = np.mean(errors)
    std_dev_error = np.std(errors)

    return mean_error, std_dev_error


def calculate_look_at_orientation(camera_positions, focal_points):
    """
    Calculate the rotation matrix and quaternion for the camera to look at the target position.

    Args:
        camera_positions (np.ndarray): The positions of the camera.
        focal_points (np.ndarray): The target positions (focal point).

    Returns:
        tuple: A tuple containing two numpy arrays:
               - An array of rotation matrices for each sample
               - An array of quaternions (x, y, z, w) for each sample
    """
    num_samples = len(camera_positions)
    rotation_matrices = np.empty((num_samples, 3, 3))
    quaternions = np.empty((num_samples, 4))

    for i in range(num_samples):
        # Calculate the forward vector (X-axis of camera)
        forward = focal_points[i] - camera_positions[i]
        forward = forward / np.linalg.norm(forward)

        # Calculate the right vector (negative Y-axis of camera)
        world_up = np.array([-1, 0, 0])  # Z is up in world space
        right = -np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        # Calculate the up vector (Z-axis of camera)
        up = np.cross(right, forward)

        # Construct the rotation matrix
        rotation_matrix = np.column_stack((forward, -right, up))
        rotation_matrices[i] = rotation_matrix

        # Convert rotation matrix to quaternion
        r = R.from_matrix(rotation_matrix)
        quaternions[i] = r.as_quat()  # Returns in [x, y, z, w] format

    return rotation_matrices, quaternions


def apply_z_axis_rotation(rotation_matrices, num_circles, samples_per_circle, rz_max=np.radians(120)):
    """
    Apply a random rotation around the optical axis (z-axis) to the given rotation matrices.

    Args:
        rotation_matrices (np.ndarray): The initial rotation matrices.
        num_circles (int): Number of circles used in sampling.
        samples_per_circle (int): Number of samples per circle.
        rz_max (float): Maximum rotation angle around the optical axis in radians.

    Returns:
        np.ndarray: An array of quaternions (x, y, z, w) for each sample after z-axis rotation.
    """
    num_samples = len(rotation_matrices)
    quaternions = []

    for cn in range(num_circles):
        # Generate a sequence of rotation angles for this circle
        rz_values = np.linspace(-rz_max, rz_max, num=samples_per_circle)

        for i in range(samples_per_circle):
            idx = cn * samples_per_circle + i
            if idx >= num_samples:
                break

            # Create a rotation matrix for the optical axis rotation
            rz = rz_values[i]
            cos_rz = np.cos(rz)
            sin_rz = np.sin(rz)
            Rx = np.array([
                [1, 0, 0],
                [0, cos_rz, -sin_rz],
                [0, sin_rz, cos_rz]
            ])

            # Apply the optical axis rotation to the initial rotation matrix
            final_rotation_matrix = np.dot(rotation_matrices[idx], Rx)

            # Convert final rotation matrix to quaternion using scipy
            r = R.from_matrix(final_rotation_matrix)
            quaternion = r.as_quat()  # Returns in [x, y, z, w] format

            quaternions.append(quaternion)

    return np.array(quaternions)


def rotate_camera_x_axis(orientation_quaternion, angle_degrees):
    """
    Rotate the camera around its X-axis by the specified angle.

    Args:
        orientation_quaternion (np.ndarray): The original orientation quaternion.
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The new orientation quaternion after rotation.
    """
    # Convert the original quaternion to a rotation object
    original_rotation = R.from_quat(orientation_quaternion)

    # Create a rotation around the X-axis
    x_rotation = R.from_euler('x', angle_degrees, degrees=True)

    # Combine the rotations
    new_rotation = original_rotation * x_rotation

    # Convert back to quaternion
    new_quaternion = new_rotation.as_quat()

    return new_quaternion
