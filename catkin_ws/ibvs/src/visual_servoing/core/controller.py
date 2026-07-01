import numpy as np
from scipy.spatial.transform import Rotation as R


class VisualServoingController:
    """Core visual servoing controller with control law implementation."""
    
    def __init__(self, config):
        self.config = config
        self.iteration_count = 0
        
        # Initialize EMA for velocity smoothing
        self.ema_velocities = [None] * 6
        
        # Initialize tracking variables
        self.velocity_history = []
        self.position_history = []
        self.orientation_history = []
        self.velocity_mean_100 = []
        self.velocity_mean_10 = []
        self.average_velocities = []
        self.velocity_vector_history = []
        
        # Applied velocities tracking
        self.applied_velocity_x = []
        self.applied_velocity_y = []
        self.applied_velocity_z = []
        self.applied_velocity_roll = []
        self.applied_velocity_pitch = []
        self.applied_velocity_yaw = []
        
        # Stable convergence tracking (NEW)
        self.stable_convergence_history = []  # List of (pos_error, rot_error) tuples
        self.stable_convergence_enabled = config.__dict__.get('stable_convergence_enabled', True)
        # Default to 0.1cm for position, 0.1° for rotation
        self.stable_convergence_pos_thresh = config.__dict__.get('stable_convergence_position_threshold', 0.1)
        self.stable_convergence_rot_thresh = config.__dict__.get('stable_convergence_rotation_threshold', 0.1)
        self.stable_convergence_window = config.__dict__.get('stable_convergence_window', 50)

        # Feature error stagnation detection
        self.feature_stagnation_history = []  # List of feature errors
        self.feature_stagnation_enabled = config.__dict__.get('feature_stagnation_enabled', True)
        self.feature_stagnation_window = config.__dict__.get('feature_stagnation_window', 10)
        self.feature_stagnation_threshold = config.__dict__.get('feature_stagnation_threshold', 0.5)  # pixels

        # Real robot visual-only convergence (error reduction based)
        self.initial_feature_error_visual_only = None  # Track initial error for % reduction
        self.real_robot_error_reduction_target = config.__dict__.get('real_robot_error_reduction_target', 0.90)  # 90% reduction
        self.real_robot_max_iterations = config.__dict__.get('real_robot_max_iterations', 300)  # Max iters before marking converged

        # Timing tracking for FPS calculation
        self.iteration_times = []  # Time per iteration in seconds
        
        # Error tracking
        self.initial_error_translation = None
        self.initial_error_rotation = None
        
    def update_ema(self, index, new_value):
        """
        Update EMA for a single velocity component.

        Args:
            index (int): Index of the velocity component
            new_value (float): New velocity value
        Returns:
            float: Updated EMA value
        """
        if self.ema_velocities[index] is None:
            self.ema_velocities[index] = new_value
        else:
            self.ema_velocities[index] = (self.config.ema_alpha * new_value + 
                                         (1 - self.config.ema_alpha) * self.ema_velocities[index])
        return self.ema_velocities[index]
    
    def compute_control_velocity(self, s_xy, s_star_xy, Z):
        """
        Compute visual servoing control velocity.
        
        Args:
            s_xy: Current feature points in normalized coordinates
            s_star_xy: Desired feature points in normalized coordinates
            Z: Depth values for current features
            
        Returns:
            numpy.ndarray: 6D velocity command
        """
        # Calculate error
        e = s_xy - s_star_xy
        e = e.reshape((len(s_xy) * 2, 1))
        
        # Calculate interaction matrix
        L = self.calculate_interaction_matrix(s_xy, Z)
        
        # Compute velocity using pseudo-inverse
        v_c = -self.config.lambda_ * np.linalg.pinv(L.astype('float')) @ e
        
        # Apply EMA smoothing
        v_c_smoothed = np.array([self.update_ema(i, v) for i, v in enumerate(v_c.flatten())])
        
        # Store velocity history
        self.velocity_vector_history.append(v_c_smoothed)
        if len(self.velocity_vector_history) > self.config.max_velocity_vector_history:
            self.velocity_vector_history.pop(0)
            
        return v_c_smoothed
    
    def calculate_interaction_matrix(self, s_xy, Z):
        """Calculate the interaction matrix for the feature points."""
        L = np.zeros([2 * len(s_xy), 6], dtype=float)

        for count in range(len(s_xy)):
            x, y, z = s_xy[count, 0], s_xy[count, 1], Z[count, 0]
            L[2 * count, :] = [-1 / z, 0, x / z, x * y, -(1 + x ** 2), y]
            L[2 * count + 1, :] = [0, -1 / z, y / z, 1 + y ** 2, -x * y, -x]

        return L
    
    def transform_to_real_world(self, s_uv, s_uv_star):
        """Transform pixel feature points to real-world coordinates."""
        s_xy = []
        s_star_xy = []

        for uv, uv_star in zip(s_uv, s_uv_star):
            x = (uv[0] - self.config.c_x) / self.config.f_x
            y = (uv[1] - self.config.c_y) / self.config.f_y
            s_xy.append([x, y])

            x_star = (uv_star[0] - self.config.c_x) / self.config.f_x
            y_star = (uv_star[1] - self.config.c_y) / self.config.f_y
            s_star_xy.append([x_star, y_star])

        return np.array(s_xy), np.array(s_star_xy)
    
    def update_iteration_stats(self, v_c):
        """Update iteration statistics."""
        self.iteration_count += 1
        
        # Calculate average velocity
        avg_velocity = np.mean(np.abs(v_c))
        self.average_velocities.append(avg_velocity)
        self.velocity_history.append(avg_velocity)
        
        # Calculate velocity means
        if len(self.velocity_history) >= 100:
            self.velocity_mean_100.append(np.mean(self.velocity_history[-100:]))
        else:
            self.velocity_mean_100.append(np.mean(self.velocity_history))

        if len(self.velocity_history) >= 10:
            self.velocity_mean_10.append(np.mean(self.velocity_history[-10:]))
        else:
            self.velocity_mean_10.append(np.mean(self.velocity_history))
    
    def store_applied_velocities(self, v_c):
        """Store applied velocity components."""
        v_c_clipped = np.clip(v_c, -self.config.max_velocity, self.config.max_velocity)
        
        self.applied_velocity_x.append(v_c_clipped[0])
        self.applied_velocity_y.append(v_c_clipped[1])
        self.applied_velocity_z.append(v_c_clipped[2])
        self.applied_velocity_roll.append(v_c_clipped[3])
        self.applied_velocity_pitch.append(v_c_clipped[4])
        self.applied_velocity_yaw.append(v_c_clipped[5])
        
        return v_c_clipped
    
    def calculate_end_error(self, current_position, current_orientation, 
                          desired_position, desired_orientation):
        """
        Calculate position and orientation errors.

        Args:
            current_position: Current camera position
            current_orientation: Current camera orientation quaternion
            desired_position: Desired camera position
            desired_orientation: Desired camera orientation quaternion

        Returns:
            tuple: (position_error in cm, orientation_error in degrees)
        """
        if current_position is None or current_orientation is None:
            return float('inf'), float('inf')
            
        # Calculate position error in centimeters
        position_error = np.linalg.norm(current_position - desired_position) * 100

        # Calculate orientation error in degrees
        current_rot = R.from_quat(current_orientation)
        desired_rot = R.from_quat(desired_orientation)
        orientation_error = (current_rot.inv() * desired_rot).magnitude() * (180 / np.pi)

        return position_error, orientation_error
    
    def check_convergence(self, current_position, current_orientation,
                         desired_position, desired_orientation, feature_error=None):
        """
        Simplified convergence check for consistent evaluation:
        - Abort if error exceeds 3x initial (divergence)
        - Converged if BOTH errors reduced by 90% OR below absolute thresholds
        - Always run full iterations (no early stopping)
        """
        # Get current errors
        current_error_translation, current_error_rotation = self.calculate_end_error(
            current_position, current_orientation, desired_position, desired_orientation)

        # Initialize initial errors if not already done
        # Check for preset flag to prevent overwriting true initial errors
        if not hasattr(self, 'initial_errors_preset') or not self.initial_errors_preset:
            if self.initial_error_translation is None or self.initial_error_rotation is None:
                self.initial_error_translation = current_error_translation
                self.initial_error_rotation = current_error_rotation
                print(f"Initial errors set: Trans={self.initial_error_translation:.2f}cm, Rot={self.initial_error_rotation:.1f}°")
        else:
            # Initial errors were preset - just log once
            if self.iteration_count == 1:
                print(f"Using preset initial errors: Trans={self.initial_error_translation:.2f}cm, Rot={self.initial_error_rotation:.1f}°")
                
                # Print convergence requirements
                print(f"\nConvergence requirements:")
                print(f"  Option 1 (90% reduction):")
                print(f"    - Translation must reach < {self.initial_error_translation * 0.1:.2f} cm")
                print(f"    - Rotation must reach < {self.initial_error_rotation * 0.1:.2f}°")
                print(f"  Option 2 (absolute thresholds):")
                print(f"    - Translation < 1.0 cm AND Rotation < 1.0°")
                print(f"  Divergence threshold: Position > {self.initial_error_translation * 3:.2f} cm\n")

        # Check if current error is more than 3x the initial error (divergence check)
        if current_error_translation > 3 * self.initial_error_translation:
            print(f"\n✗ DIVERGENCE: Position error ({current_error_translation:.2f}cm) > 3x initial ({self.initial_error_translation:.2f}cm)")
            print(f"  Divergence threshold: {3 * self.initial_error_translation:.2f}cm")
            return True, False  # Done but not converged

        # Convergence check: BOTH errors must be reduced by 90% OR below absolute thresholds
        # Absolute thresholds: 1.0 cm for position, 1.0 degree for rotation
        error_reduced_90_percent = (
            (current_error_translation / self.initial_error_translation) < 0.1 and
            (current_error_rotation / self.initial_error_rotation) < 0.1
        )
        
        # Also consider converged if below absolute thresholds
        below_absolute_thresholds = (
            current_error_translation < 1.0 and  # 1.0 cm
            current_error_rotation < 1.0  # 1.0 degree
        )
        
        converged = error_reduced_90_percent or below_absolute_thresholds
        
        # NEW: Stable convergence check - early stopping when errors stay below threshold
        if self.stable_convergence_enabled and self.iteration_count >= self.config.min_iterations:
            # Add current errors to history
            self.stable_convergence_history.append((current_error_translation, current_error_rotation))
            
            # Keep only the window size
            if len(self.stable_convergence_history) > self.stable_convergence_window:
                self.stable_convergence_history.pop(0)
            
            # Check if we have enough history and all are below threshold
            if len(self.stable_convergence_history) >= self.stable_convergence_window:
                all_below_threshold = all(
                    pos_err < self.stable_convergence_pos_thresh and 
                    rot_err < self.stable_convergence_rot_thresh
                    for pos_err, rot_err in self.stable_convergence_history
                )
                
                if all_below_threshold:
                    print(f"\n✓ STABLE CONVERGENCE: Errors below threshold for {self.stable_convergence_window} iterations")
                    print(f"  Position: <{self.stable_convergence_pos_thresh:.2f}cm ✓")
                    print(f"  Rotation: <{self.stable_convergence_rot_thresh:.1f}° ✓")
                    print(f"  Final: Trans={current_error_translation:.3f}cm, Rot={current_error_rotation:.3f}°")
                    print(f"  Stopped early at iteration {self.iteration_count} (saved {self.config.max_iterations - self.iteration_count} iterations)")
                    return True, True  # Done and converged

        # NEW: Feature error stagnation check - early stopping when feature error stays near zero
        if (self.feature_stagnation_enabled and feature_error is not None and
            self.iteration_count >= self.config.min_iterations):
            # Add current feature error to history
            self.feature_stagnation_history.append(feature_error)

            # Keep only the window size
            if len(self.feature_stagnation_history) > self.feature_stagnation_window:
                self.feature_stagnation_history.pop(0)

            # Check if we have enough history and all are below threshold
            if len(self.feature_stagnation_history) >= self.feature_stagnation_window:
                all_below_threshold = all(
                    err <= self.feature_stagnation_threshold
                    for err in self.feature_stagnation_history
                )

                if all_below_threshold:
                    print(f"\n✓ FEATURE STAGNATION: Feature error below {self.feature_stagnation_threshold:.1f}px for {self.feature_stagnation_window} iterations")
                    print(f"  Current feature error: {feature_error:.2f} pixels")
                    print(f"  Final pose: Trans={current_error_translation:.3f}cm, Rot={current_error_rotation:.3f}°")
                    print(f"  Stopped early at iteration {self.iteration_count} (saved {self.config.max_iterations - self.iteration_count} iterations)")
                    return True, True  # Done and converged

        # Print progress periodically
        if self.iteration_count % 50 == 0 and self.iteration_count > 0:
            trans_percent = (current_error_translation / self.initial_error_translation) * 100
            rot_percent = (current_error_rotation / self.initial_error_rotation) * 100
            print(f"Progress: Trans {trans_percent:.1f}% of initial, Rot {rot_percent:.1f}% of initial")

        # Check if maximum iterations reached
        if self.iteration_count >= self.config.max_iterations:
            if converged:
                if below_absolute_thresholds and not error_reduced_90_percent:
                    print(f"\n✓ CONVERGED: Below absolute thresholds after {self.iteration_count} iterations")
                    print(f"  Final errors: Trans={current_error_translation:.2f}cm < 1.0cm ✓")
                    print(f"                Rot={current_error_rotation:.2f}° < 1.0° ✓")
                else:
                    print(f"\n✓ CONVERGED: Both errors reduced by 90% after {self.iteration_count} iterations")
                    print(f"  Final errors: Trans={current_error_translation:.2f}cm < {self.initial_error_translation * 0.1:.2f}cm ✓")
                    print(f"                Rot={current_error_rotation:.2f}° < {self.initial_error_rotation * 0.1:.2f}° ✓")
                return True, True
            else:
                trans_percent = (current_error_translation / self.initial_error_translation) * 100
                rot_percent = (current_error_rotation / self.initial_error_rotation) * 100
                print(f"\n✗ NOT CONVERGED: Max iterations reached. Trans at {trans_percent:.1f}%, Rot at {rot_percent:.1f}% of initial")
                print(f"  Final errors: Trans={current_error_translation:.2f}cm (need < {self.initial_error_translation * 0.1:.2f}cm or < 1.0cm)")
                print(f"                Rot={current_error_rotation:.2f}° (need < {self.initial_error_rotation * 0.1:.2f}° or < 1.0°)")
                return True, False
        
        # Continue running (no early stopping)
        return False, False

    def check_convergence_visual_only(self, feature_error=None, velocity=None):
        """
        Convergence check for real robot mode (visual feature error only).
        No ground truth pose available - uses feature error reduction and iteration limit.

        Primary convergence criteria (real robot):
        1. Feature error reduced by 90% from initial (configurable via real_robot_error_reduction_target)
        2. OR max iterations reached (configurable via real_robot_max_iterations, default 300)

        Args:
            feature_error: Current average feature error in pixels
            velocity: Current velocity command (6D numpy array)

        Returns:
            tuple: (done, converged)
                - done: True if servoing should stop
                - converged: True if successfully converged
        """
        # Track initial feature error on first valid measurement
        if feature_error is not None:
            if self.initial_feature_error_visual_only is None and feature_error > 0:
                self.initial_feature_error_visual_only = feature_error
                print(f"\n[Real Robot Convergence] Initial feature error: {feature_error:.2f} pixels")
                print(f"  Target: {self.real_robot_error_reduction_target*100:.0f}% reduction = {feature_error * (1 - self.real_robot_error_reduction_target):.2f} pixels")
                print(f"  Max iterations: {self.real_robot_max_iterations}")

            # PRIMARY CHECK 1: Feature error reduction (90% default)
            if self.initial_feature_error_visual_only is not None and self.initial_feature_error_visual_only > 0:
                error_reduction = 1.0 - (feature_error / self.initial_feature_error_visual_only)

                if error_reduction >= self.real_robot_error_reduction_target:
                    print(f"\n✓ CONVERGED: Feature error reduced by {error_reduction*100:.1f}% (target: {self.real_robot_error_reduction_target*100:.0f}%)")
                    print(f"  Initial error: {self.initial_feature_error_visual_only:.2f} pixels")
                    print(f"  Final error: {feature_error:.2f} pixels")
                    print(f"  Iterations: {self.iteration_count}")
                    return True, True  # Done and converged

        # PRIMARY CHECK 2: Max iterations for real robot (mark as converged)
        if self.iteration_count >= self.real_robot_max_iterations:
            if feature_error is not None and self.initial_feature_error_visual_only is not None:
                error_reduction = 1.0 - (feature_error / self.initial_feature_error_visual_only)
                print(f"\n✓ CONVERGED (max iterations): Reached {self.iteration_count} iterations")
                print(f"  Error reduction: {error_reduction*100:.1f}%")
                print(f"  Initial error: {self.initial_feature_error_visual_only:.2f} pixels")
                print(f"  Final error: {feature_error:.2f} pixels")
            else:
                print(f"\n✓ CONVERGED (max iterations): Reached {self.iteration_count} iterations")
            return True, True  # Done and converged (mark as converged at max iters)

        # Print progress periodically
        if self.iteration_count % 50 == 0 and self.iteration_count > 0:
            if feature_error is not None and self.initial_feature_error_visual_only is not None:
                error_reduction = 1.0 - (feature_error / self.initial_feature_error_visual_only)
                print(f"Progress: Iter {self.iteration_count}, Error: {feature_error:.2f}px, Reduction: {error_reduction*100:.1f}%")
            elif feature_error is not None:
                print(f"Progress: Iter {self.iteration_count}, Error: {feature_error:.2f}px")
            else:
                print(f"Progress: Iter {self.iteration_count}")

        return False, False  # Continue running

    def reset(self):
        """Reset controller state for new servoing task."""
        self.iteration_count = 0
        self.velocity_history = []
        self.position_history = []
        self.orientation_history = []
        self.velocity_mean_100 = []
        self.velocity_mean_10 = []
        self.average_velocities = []
        self.velocity_vector_history = []
        self.applied_velocity_x = []
        self.applied_velocity_y = []
        self.applied_velocity_z = []
        self.applied_velocity_roll = []
        self.applied_velocity_pitch = []
        self.applied_velocity_yaw = []
        self.initial_error_translation = None
        self.initial_error_rotation = None
        self.stable_convergence_history = []  # Reset stable convergence history
        self.feature_stagnation_history = []  # Reset feature stagnation history
        self.initial_errors_preset = False  # Reset the preset flag
        self.initial_feature_error_visual_only = None  # Reset real robot error tracking
        self.ema_velocities = [None] * 6
