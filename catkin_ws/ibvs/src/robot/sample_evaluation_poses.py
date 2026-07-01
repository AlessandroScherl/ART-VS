#!/usr/bin/env python3
"""
Real Robot Pose Sampling Script

Generates and applies sampled camera poses for visual servoing evaluation on UR5 robot.

Features:
- Samples 25 poses within a 0.2m × 0.2m × 0.1m cuboid centered at zero position
- Applies look-at orientation to ensure camera points at grasping plane center
- Applies random roll rotation (±120°) around optical axis
- Interactive verification with pause between poses
- Deterministic sampling with configurable seed

Usage:
    # Generate and apply poses with verification
    python3 sample_evaluation_poses.py --num-poses 25 --seed 42

    # Save poses to file without applying
    python3 sample_evaluation_poses.py --save-only --output sampled_poses.json

    # Apply saved poses from file
    python3 sample_evaluation_poses.py --load sampled_poses.json

    # Use smaller roll range for safety testing
    python3 sample_evaluation_poses.py --roll-range 60
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import json
import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Camera offset from tool0 to camera_color_optical_frame (from TF)
# This means: camera_position = tcp_position + CAMERA_OFFSET (when frames are aligned)
CAMERA_OFFSET = np.array([0.071, -0.001, 0.011])  # meters


def sample_camera_positions(zero_position, box_size, num_samples, seed=42):
    """
    Sample positions uniformly within a cuboid centered at zero_position.

    Args:
        zero_position: [x, y, z] center of sampling volume
        box_size: [dx, dy, dz] dimensions of sampling cuboid
        num_samples: number of positions to generate
        seed: random seed for reproducibility

    Returns:
        positions: (num_samples, 3) array of [x, y, z] positions
    """
    np.random.seed(seed)
    half_box = np.array(box_size) / 2

    # Uniform sampling within box
    offsets = np.random.uniform(-half_box, half_box, size=(num_samples, 3))
    positions = np.array(zero_position) + offsets

    return positions


def calculate_look_at_orientation(camera_position, target_point, looking_down=True, tcp_frame_correction=90.0):
    """
    Calculate quaternion for camera to look at target point.

    The camera coordinate frame convention:
    - Z axis (forward/optical axis) points from camera to target
    - Y axis points down
    - X axis points right

    Args:
        camera_position: [x, y, z] camera position in world frame
        target_point: [x, y, z] point to look at in world frame
        looking_down: if True, camera is looking downward (use X as reference up)
                      if False, camera is looking horizontally (use Z as reference up)
        tcp_frame_correction: rotation in degrees to align with robot TCP frame (default: 90°)
                              Set to 0 for pure look-at without correction.

    Returns:
        quaternion: [qx, qy, qz, qw] orientation quaternion
    """
    camera_position = np.array(camera_position)
    target_point = np.array(target_point)

    # Forward vector: camera to target (will be camera Z axis)
    forward = target_point - camera_position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        raise ValueError("Camera position and target point are too close")
    forward = forward / forward_norm

    # Choose reference "up" vector based on camera orientation
    # CRITICAL: Must be consistent across all poses to avoid orientation discontinuities
    if looking_down:
        # Camera looking downward: use world X as reference (avoids gimbal lock at Z-aligned forward)
        # This ensures consistent orientation when camera looks nearly straight down
        world_up = np.array([1, 0, 0])
    else:
        # Camera looking horizontally: use world Z as reference
        world_up = np.array([0, 0, 1])

    # Right vector: perpendicular to forward and world up
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        # Fallback if forward is parallel to world_up
        # This shouldn't happen with proper looking_down setting
        fallback_up = np.array([0, 1, 0])
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)

    right = right / right_norm

    # Down vector: perpendicular to forward and right (camera Y axis)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Rotation matrix: columns are camera axes in world frame
    # Camera frame: X=right, Y=down, Z=forward
    rotation_matrix = np.column_stack([right, down, forward])

    # Convert to quaternion
    r = R.from_matrix(rotation_matrix)
    base_quat = r.as_quat()  # [qx, qy, qz, qw]

    # Apply TCP frame correction: rotate around Z (optical axis) to align with robot's TCP frame
    # This accounts for the 90° difference between calculated look-at and actual robot TCP orientation
    if abs(tcp_frame_correction) > 0.01:
        base_rot = R.from_quat(base_quat)
        correction_rot = R.from_euler('z', tcp_frame_correction, degrees=True)
        corrected_rot = base_rot * correction_rot
        return corrected_rot.as_quat()

    return base_quat


def apply_roll_rotation(quaternion, roll_angle_deg):
    """
    Apply roll rotation around camera's optical axis (Z axis).

    Args:
        quaternion: [qx, qy, qz, qw] base orientation
        roll_angle_deg: roll angle in degrees (positive = clockwise when looking forward)

    Returns:
        rotated quaternion: [qx, qy, qz, qw]
    """
    base_rot = R.from_quat(quaternion)
    roll_rot = R.from_euler('z', roll_angle_deg, degrees=True)

    # Apply roll in camera frame (post-multiply)
    combined = base_rot * roll_rot
    return combined.as_quat()


def quaternion_to_rotvec(quaternion):
    """
    Convert quaternion to rotation vector (axis-angle representation).

    Args:
        quaternion: [qx, qy, qz, qw] quaternion

    Returns:
        rotvec: [rx, ry, rz] rotation vector (for ur_rtde)
    """
    r = R.from_quat(quaternion)
    return r.as_rotvec()


def rotvec_to_quaternion(rotvec):
    """
    Convert rotation vector to quaternion.

    Args:
        rotvec: [rx, ry, rz] rotation vector

    Returns:
        quaternion: [qx, qy, qz, qw] quaternion
    """
    r = R.from_rotvec(rotvec)
    return r.as_quat()


def ensure_shortest_path_quaternion(q_current, q_target):
    """
    Ensure quaternion takes the shortest path from current to target.

    Quaternions q and -q represent the same rotation, but interpolating
    to -q might cause the robot to take the "long way around" (>180°).

    Args:
        q_current: [qx, qy, qz, qw] current quaternion
        q_target: [qx, qy, qz, qw] target quaternion

    Returns:
        q_target or -q_target, whichever is closer to q_current
    """
    q_current = np.array(q_current)
    q_target = np.array(q_target)

    # Compute dot product - if negative, the quaternions are on opposite hemispheres
    dot = np.dot(q_current, q_target)

    if dot < 0:
        # Negate to take the shorter path
        return -q_target
    return q_target


def camera_to_tcp_position(camera_position, camera_orientation_quat):
    """
    Convert camera position to TCP position.

    Since the camera is mounted on the tool with a fixed offset,
    we need to transform the offset by the current orientation.

    Args:
        camera_position: [x, y, z] desired camera position in world frame
        camera_orientation_quat: [qx, qy, qz, qw] desired orientation

    Returns:
        tcp_position: [x, y, z] corresponding TCP position
    """
    # The camera offset is defined in the tool frame
    # When the tool rotates, the offset rotates with it
    r = R.from_quat(camera_orientation_quat)

    # Transform the offset from tool frame to world frame
    offset_world = r.apply(CAMERA_OFFSET)

    # TCP position is camera position minus the rotated offset
    tcp_position = np.array(camera_position) - offset_world

    return tcp_position


def sample_poses(zero_pose, box_size, look_at_target, num_samples,
                 roll_range=60.0, seed=42):
    """
    Generate all sampled poses.

    Samples are generated in CAMERA frame coordinates, then converted to TCP
    poses for the robot. This ensures the camera (not the tool) is positioned
    within the specified sampling volume.

    Args:
        zero_pose: [x, y, z, rx, ry, rz] from ur_rtde (zero position TCP pose)
        box_size: [dx, dy, dz] sampling cuboid dimensions in meters
        look_at_target: [x, y, z] point to look at in world frame
        num_samples: number of poses to generate
        roll_range: max roll angle in degrees (samples uniformly from ±roll_range)
        seed: random seed for reproducibility

    Returns:
        poses: list of [x, y, z, rx, ry, rz] TCP poses for moveL
        metadata: dict with sampling parameters and intermediate values
    """
    np.random.seed(seed)

    # Get zero TCP position and orientation
    zero_tcp_position = np.array(zero_pose[:3])
    zero_tcp_rotvec = np.array(zero_pose[3:])
    zero_tcp_quat = rotvec_to_quaternion(zero_tcp_rotvec)

    # Compute camera zero position from TCP zero position
    zero_tcp_rot = R.from_quat(zero_tcp_quat)
    camera_offset_world = zero_tcp_rot.apply(CAMERA_OFFSET)
    zero_camera_position = zero_tcp_position + camera_offset_world

    print(f"  TCP zero position:    [{', '.join(f'{v:.4f}' for v in zero_tcp_position)}]")
    print(f"  Camera zero position: [{', '.join(f'{v:.4f}' for v in zero_camera_position)}]")

    look_at_target = np.array(look_at_target)

    # Sample CAMERA positions within cuboid (centered at camera zero)
    camera_positions = sample_camera_positions(zero_camera_position, box_size, num_samples, seed)

    # Sample roll angles uniformly
    # Use separate seed offset for roll to ensure independence
    np.random.seed(seed + 1000)
    roll_angles = np.random.uniform(-roll_range, roll_range, num_samples)

    poses = []
    metadata = {
        'camera_positions': [],
        'tcp_positions': [],
        'roll_angles': [],
        'quaternions': [],
    }

    # Track previous quaternion for shortest-path calculation
    prev_quat = zero_tcp_quat

    for i in range(num_samples):
        # Calculate look-at orientation for this camera position
        quat = calculate_look_at_orientation(camera_positions[i], look_at_target)

        # Apply roll rotation around optical axis
        quat_rolled = apply_roll_rotation(quat, roll_angles[i])

        # Ensure shortest rotation path from previous pose
        quat_rolled = ensure_shortest_path_quaternion(prev_quat, quat_rolled)
        prev_quat = quat_rolled  # Update for next iteration

        # Convert camera position to TCP position
        tcp_position = camera_to_tcp_position(camera_positions[i], quat_rolled)

        # Convert orientation to rotation vector for ur_rtde
        rotvec = quaternion_to_rotvec(quat_rolled)

        # Combine TCP position and orientation
        pose = list(tcp_position) + list(rotvec)
        poses.append(pose)

        # Store metadata
        metadata['camera_positions'].append(list(camera_positions[i]))
        metadata['tcp_positions'].append(list(tcp_position))
        metadata['roll_angles'].append(float(roll_angles[i]))
        metadata['quaternions'].append(list(quat_rolled))

    return poses, metadata


def save_poses_to_file(filepath, poses, zero_joints, zero_pose, look_at_target,
                       box_size, roll_range, seed, metadata=None):
    """Save generated poses and parameters to JSON file."""
    pose_data = {
        'version': '1.0',
        'description': 'Sampled evaluation poses for real robot visual servoing',
        'parameters': {
            'zero_joints': list(zero_joints),
            'zero_pose': list(zero_pose),
            'look_at_target': list(look_at_target),
            'box_size': list(box_size),
            'roll_range': float(roll_range),
            'seed': int(seed),
            'num_poses': len(poses),
        },
        'poses': [list(p) for p in poses],
    }

    if metadata:
        pose_data['metadata'] = {
            'camera_positions': metadata.get('camera_positions', metadata.get('positions', [])),
            'tcp_positions': metadata.get('tcp_positions', []),
            'roll_angles': metadata['roll_angles'],
            'quaternions': metadata['quaternions'],
        }

    with open(filepath, 'w') as f:
        json.dump(pose_data, f, indent=2)

    print(f"  Saved {len(poses)} poses to: {filepath}")


def load_poses_from_file(filepath):
    """Load poses from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    poses = data['poses']
    params = data['parameters']

    print(f"  Loaded {len(poses)} poses from: {filepath}")
    print(f"  Parameters: box_size={params['box_size']}, seed={params['seed']}")

    return poses, params


def print_pose_info(pose, index, total):
    """Print pose information in readable format."""
    position = pose[:3]
    rotvec = pose[3:]

    # Convert to Euler angles for display
    r = R.from_rotvec(rotvec)
    euler = r.as_euler('xyz', degrees=True)

    print(f"\n--- Pose {index + 1}/{total} ---")
    print(f"  Position (m):    X={position[0]:.4f}, Y={position[1]:.4f}, Z={position[2]:.4f}")
    print(f"  Rotation (deg):  Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}, Yaw={euler[2]:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Sample evaluation poses for real robot visual servoing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and apply 25 poses with visual verification
  python3 sample_evaluation_poses.py --num-poses 25 --seed 42

  # Save poses to file without applying
  python3 sample_evaluation_poses.py --num-poses 25 --save-only

  # Load and apply poses from file
  python3 sample_evaluation_poses.py --load sampled_poses.json

  # Use slower speed for safety
  python3 sample_evaluation_poses.py --speed 0.05
        """
    )

    # Sampling parameters
    parser.add_argument('--num-poses', type=int, default=25,
                        help='Number of poses to sample (default: 25)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--box-size', type=float, nargs=3, default=[0.2, 0.2, 0.1],
                        metavar=('X', 'Y', 'Z'),
                        help='Sampling box size in meters (default: 0.2 0.2 0.1)')
    parser.add_argument('--roll-range', type=float, default=120.0,
                        help='Max roll angle in degrees (default: 120)')

    # Motion parameters
    parser.add_argument('--speed', type=float, default=0.1,
                        help='Motion speed in m/s (default: 0.1)')
    parser.add_argument('--accel', type=float, default=0.1,
                        help='Motion acceleration in m/s² (default: 0.1)')

    # File operations
    parser.add_argument('--save-only', action='store_true',
                        help='Save poses to file without applying')
    parser.add_argument('--output', type=str, default='sampled_poses.json',
                        help='Output file for poses (default: sampled_poses.json)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load poses from file instead of generating')

    # Execution modes
    parser.add_argument('--auto', action='store_true',
                        help='Apply all poses automatically without pauses')
    parser.add_argument('--wait-time', type=float, default=2.0,
                        help='Wait time at each pose in auto mode (default: 2.0s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print poses without connecting to robot')

    args = parser.parse_args()

    # Zero position joint angles (user-provided)
    # Camera is 0.42m above the grasping plane at this configuration
    ZERO_JOINTS = [-0.0003, -1.5635, 1.2421, -1.2495, -1.5702, -1.5695]

    print("=" * 60)
    print("REAL ROBOT POSE SAMPLING SCRIPT")
    print("=" * 60)

    # Dry run mode - just show what would be generated
    if args.dry_run:
        print("\n[DRY RUN MODE - No robot connection]")

        if args.load:
            poses, params = load_poses_from_file(args.load)
            zero_pose = params['zero_pose']
            look_at_target = params['look_at_target']
        else:
            # Simulate zero pose (typical values)
            print("\n[Simulating zero position...]")
            zero_pose = [0.5, -0.1, 0.42, 0.0, 3.14, 0.0]  # Example TCP pose
            look_at_target = [zero_pose[0], zero_pose[1], zero_pose[2] - 0.42]

            print(f"  Simulated zero pose: {zero_pose}")
            print(f"  Look-at target: {look_at_target}")

            poses, metadata = sample_poses(
                zero_pose=zero_pose,
                box_size=args.box_size,
                look_at_target=look_at_target,
                num_samples=args.num_poses,
                roll_range=args.roll_range,
                seed=args.seed
            )

            save_poses_to_file(
                args.output, poses, ZERO_JOINTS, zero_pose, look_at_target,
                args.box_size, args.roll_range, args.seed, metadata
            )

        print(f"\n[Generated {len(poses)} poses]")
        for i, pose in enumerate(poses[:5]):  # Show first 5
            print_pose_info(pose, i, len(poses))
        if len(poses) > 5:
            print(f"\n  ... and {len(poses) - 5} more poses")

        print("\n[DRY RUN COMPLETE]")
        return

    # Connect to robot via MoveIt2
    print(f"\n[Connecting to robot via MoveIt2...]")
    try:
        from moveit_robot_interface import MoveItRobotInterface
        robot = MoveItRobotInterface()
        print("  MoveIt2 robot interface ready!")
    except Exception as e:
        print(f"  [ERROR] Failed to connect: {e}")
        sys.exit(1)

    # Load poses from file if specified
    if args.load:
        poses, params = load_poses_from_file(args.load)
        zero_pose = params['zero_pose']
        look_at_target = params['look_at_target']
        ZERO_JOINTS = params.get('zero_joints', ZERO_JOINTS)
    else:
        # Step 1: Move to zero position and get Cartesian pose
        print("\n[Step 1] Moving to zero position...")
        try:
            robot.move_to_joints(ZERO_JOINTS, speed=0.3, accel=0.3)
        except Exception as e:
            print(f"  [ERROR] Failed to move to zero: {e}")
            sys.exit(1)

        zero_pose = robot.get_tcp_pose()
        if zero_pose is None:
            print("  [ERROR] Could not read TCP pose")
            sys.exit(1)
        print(f"  Zero pose (TCP): [{', '.join(f'{v:.4f}' for v in zero_pose)}]")

        # Calculate camera position from TCP position
        zero_tcp_quat = rotvec_to_quaternion(zero_pose[3:])
        zero_tcp_rot = R.from_quat(zero_tcp_quat)
        camera_offset_world = zero_tcp_rot.apply(CAMERA_OFFSET)
        camera_zero_position = np.array(zero_pose[:3]) + camera_offset_world
        print(f"  Camera position: [{', '.join(f'{v:.4f}' for v in camera_zero_position)}]")

        # Calculate look-at target: point directly below CAMERA at grasping plane
        # The camera is 0.42m above the grasping plane
        look_at_target = [camera_zero_position[0], camera_zero_position[1], camera_zero_position[2] - 0.42]
        print(f"  Look-at target:  [{', '.join(f'{v:.4f}' for v in look_at_target)}]")

        # Confirm with user
        if not args.save_only and not args.auto:
            user_input = input("\n>>> Press ENTER to confirm zero position, 'q' to quit: ").strip().lower()
            if user_input == 'q':
                print("[Aborted by user]")
                return

        # Step 2: Generate sampled poses
        print(f"\n[Step 2] Generating {args.num_poses} sampled poses...")
        print(f"  Box size: {args.box_size} m")
        print(f"  Roll range: ±{args.roll_range}°")
        print(f"  Seed: {args.seed}")

        poses, metadata = sample_poses(
            zero_pose=zero_pose,
            box_size=args.box_size,
            look_at_target=look_at_target,
            num_samples=args.num_poses,
            roll_range=args.roll_range,
            seed=args.seed
        )

        # Calculate initial error statistics (like simulation)
        # The "desired" orientation is the HOME TCP orientation (where goal image is captured)
        # NOT a calculated look-at orientation - use actual robot pose at goal!
        desired_orientation_quat = rotvec_to_quaternion(zero_pose[3:])

        # DEBUG: Show desired orientation (HOME TCP)
        r_desired_debug = R.from_quat(desired_orientation_quat)
        desired_euler = r_desired_debug.as_euler('xyz', degrees=True)
        print(f"\n  [DEBUG] Desired orientation (HOME TCP - where goal image is captured):")
        print(f"          Euler (xyz): Roll={desired_euler[0]:.1f}°, Pitch={desired_euler[1]:.1f}°, Yaw={desired_euler[2]:.1f}°")
        print(f"          Quaternion: [{', '.join(f'{v:.4f}' for v in desired_orientation_quat)}]")

        # Also show the calculated look-at orientation (with 90° TCP frame correction)
        look_at_quat = calculate_look_at_orientation(camera_zero_position, look_at_target)
        r_look_at = R.from_quat(look_at_quat)
        look_at_euler = r_look_at.as_euler('xyz', degrees=True)
        print(f"\n  [DEBUG] Calculated look-at orientation (with 90° TCP correction):")
        print(f"          Euler (xyz): Roll={look_at_euler[0]:.1f}°, Pitch={look_at_euler[1]:.1f}°, Yaw={look_at_euler[2]:.1f}°")

        # Show angle difference between HOME TCP and calculated look-at
        r_home_to_lookat = r_desired_debug.inv() * r_look_at
        home_to_lookat_angle = np.degrees(r_home_to_lookat.magnitude())
        print(f"          Angle from HOME TCP to corrected look-at: {home_to_lookat_angle:.1f}°")

        # Also show uncorrected look-at for comparison
        look_at_quat_uncorrected = calculate_look_at_orientation(camera_zero_position, look_at_target, tcp_frame_correction=0.0)
        r_look_at_uncorrected = R.from_quat(look_at_quat_uncorrected)
        look_at_euler_uncorrected = r_look_at_uncorrected.as_euler('xyz', degrees=True)
        print(f"\n  [DEBUG] Uncorrected look-at (without TCP correction):")
        print(f"          Euler (xyz): Roll={look_at_euler_uncorrected[0]:.1f}°, Pitch={look_at_euler_uncorrected[1]:.1f}°, Yaw={look_at_euler_uncorrected[2]:.1f}°")

        position_errors = []  # in cm
        orientation_errors = []  # in degrees

        print(f"\n  [DEBUG] Per-pose orientation errors:")
        for i in range(len(poses)):
            # Position error: distance from sampled camera position to zero camera position
            cam_pos = np.array(metadata['camera_positions'][i])
            pos_error_m = np.linalg.norm(cam_pos - camera_zero_position)
            position_errors.append(pos_error_m * 100)  # Convert to cm

            # Orientation error: angular difference from sampled orientation to desired orientation
            # Same formula as simulation: (current_rot.inv() * desired_rot).magnitude()
            sampled_quat = np.array(metadata['quaternions'][i])
            r_sampled = R.from_quat(sampled_quat)
            r_desired = R.from_quat(desired_orientation_quat)
            r_rel = r_sampled.inv() * r_desired
            # Get angle in degrees
            angle_rad = r_rel.magnitude()
            orientation_errors.append(np.degrees(angle_rad))

            # DEBUG: Show first 5 poses in detail
            if i < 5:
                sampled_euler = r_sampled.as_euler('xyz', degrees=True)
                print(f"          Pose {i+1}: Roll={sampled_euler[0]:.1f}°, Pitch={sampled_euler[1]:.1f}°, Yaw={sampled_euler[2]:.1f}° → error={np.degrees(angle_rad):.2f}° (roll_applied={metadata['roll_angles'][i]:.1f}°)")

        position_errors = np.array(position_errors)
        orientation_errors = np.array(orientation_errors)

        print(f"\n  [DEBUG] Orientation error distribution:")
        print(f"          Min: {orientation_errors.min():.2f}°, Max: {orientation_errors.max():.2f}°")
        print(f"          Median: {np.median(orientation_errors):.2f}°")

        pos_mean, pos_std = position_errors.mean(), position_errors.std()
        rot_mean, rot_std = orientation_errors.mean(), orientation_errors.std()

        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │ SAMPLING STATISTICS                                             │")
        print(f"  │ This configuration yields average initial position errors of    │")
        print(f"  │ {pos_mean:.2f} ± {pos_std:.2f} cm and orientation errors of {rot_mean:.2f} ± {rot_std:.2f}°          │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

        # Save poses to file
        save_poses_to_file(
            args.output, poses, ZERO_JOINTS, zero_pose, look_at_target,
            args.box_size, args.roll_range, args.seed, metadata
        )

    # Exit if save-only mode
    if args.save_only:
        print("\n[--save-only] Poses saved. Exiting without applying.")
        return

    # Step 3: Apply poses
    # SAFETY: Each pose is reached from the home position to avoid 360° rotations
    print(f"\n[Step 3] Applying {len(poses)} poses...")
    print("  SAFETY: Robot returns to HOME between each pose")
    if args.auto:
        print(f"  [AUTO MODE] Wait time: {args.wait_time}s per pose")
    else:
        print("  Controls: ENTER=apply, s=skip, q=quit")

    successful_poses = 0
    failed_poses = 0

    for i, pose in enumerate(poses):
        print_pose_info(pose, i, len(poses))

        if not args.auto:
            user_input = input("  [ENTER=apply, s=skip, q=quit]: ").strip().lower()

            if user_input == 'q':
                print("\n[Quit requested]")
                break
            elif user_input == 's':
                print("  [Skipped]")
                continue

        # Apply pose with MoveIt2 Cartesian planning
        print(f"  Moving to pose {i + 1}...")
        try:
            success = robot.move_to_tcp_pose(pose, speed=args.speed, accel=args.accel)

            if not success:
                print(f"  [ERROR] move_to_tcp_pose failed - pose may be unreachable")
                failed_poses += 1
                # Still return to home before next pose
                print("  Returning to HOME...")
                robot.move_to_joints(ZERO_JOINTS, speed=0.3, accel=0.3)
                continue

            # Verify actual pose
            actual_pose = robot.get_tcp_pose()
            if actual_pose is not None:
                pos_error = np.linalg.norm(np.array(actual_pose[:3]) - np.array(pose[:3])) * 1000  # mm
                print(f"  [OK] Pose reached. Position error: {pos_error:.2f} mm")
            successful_poses += 1

            # Wait in auto mode
            if args.auto:
                time.sleep(args.wait_time)

            # Return to HOME position before next pose (avoids 360° rotations)
            if i < len(poses) - 1:  # Don't return home after the last pose (we do it at the end)
                print("  Returning to HOME...")
                robot.move_to_joints(ZERO_JOINTS, speed=0.3, accel=0.3)

        except Exception as e:
            print(f"  [ERROR] Motion failed: {e}")
            failed_poses += 1
            # Still try to return to home before next pose
            try:
                print("  Returning to HOME...")
                robot.move_to_joints(ZERO_JOINTS, speed=0.3, accel=0.3)
            except:
                pass
            continue

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total poses: {len(poses)}")
    print(f"  Successful:  {successful_poses}")
    print(f"  Failed:      {failed_poses}")
    print(f"  Skipped:     {len(poses) - successful_poses - failed_poses}")

    # Return to zero position
    print("\n[Returning to zero position...]")
    try:
        robot.move_to_joints(ZERO_JOINTS, speed=0.3, accel=0.3)
        print("[Complete - Robot at zero position]")
    except Exception as e:
        print(f"[WARNING] Failed to return to zero: {e}")
    finally:
        robot.destroy()


if __name__ == '__main__':
    main()
