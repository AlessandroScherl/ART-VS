#!/usr/bin/env python3
"""Apply saved joint positions to the UR5 robot via MoveIt2.

Usage:
    python3 apply_saved_joints.py                       # uses current_joints.json
    python3 apply_saved_joints.py place_joints.json     # uses given file
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from moveit_robot_interface import MoveItRobotInterface


def _ensure_trajectory_controller():
    """Switch to scaled_joint_trajectory_controller if forward_position_controller is active.
    Safe to call when already on the trajectory controller (the switch silently no-ops)."""
    try:
        subprocess.run(
            ['ros2', 'control', 'switch_controllers',
             '--deactivate', 'forward_position_controller',
             '--activate', 'scaled_joint_trajectory_controller'],
            capture_output=True, text=True, timeout=10
        )
    except Exception:
        pass  # ignore — if it's already correct, switch_controllers errors are harmless


def move_to_saved_joints(filename="current_joints.json"):
    print("Ensuring scaled_joint_trajectory_controller is active...")
    _ensure_trajectory_controller()

    print("Connecting to robot via MoveIt2...")
    robot = MoveItRobotInterface()

    try:
        # Wait for first joint_state message
        print("Waiting for /joint_states...")
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if robot.get_joint_positions() is not None:
                break
            time.sleep(0.1)
        else:
            print("ERROR: No /joint_states received within 8s. Is the UR5 driver running?")
            return

        # Load saved joints
        with open(filename, 'r') as f:
            joints_data = json.load(f)

        target_joints = joints_data['joint_positions']

        print("\nLoaded target joint positions:")
        print(f"Radians: {[round(x, 4) for x in target_joints]}")
        print(f"Degrees: {[round(np.degrees(x), 4) for x in target_joints]}")

        # Get current joint positions
        current_joints = robot.get_joint_positions()
        if current_joints is not None:
            print(f"\nCurrent joint positions:")
            print(f"Radians: {[round(x, 4) for x in current_joints]}")
            print(f"Degrees: {[round(np.degrees(x), 4) for x in current_joints]}")

        # Move to joint positions
        print("\nMoving to target joint positions...")
        success = robot.move_to_joints(target_joints, speed=0.5, accel=0.5)
        if success:
            print("Movement completed!")

            # Print final joint positions
            final_joints = robot.get_joint_positions()
            if final_joints is not None:
                print(f"\nFinal joint positions reached:")
                print(f"Radians: {[round(x, 4) for x in final_joints]}")
                print(f"Degrees: {[round(np.degrees(x), 4) for x in final_joints]}")

                # Check joint errors
                joint_errors = np.array(final_joints) - np.array(target_joints)
                max_error_deg = np.max(np.abs(np.degrees(joint_errors)))
                print(f"\nMaximum joint error: {round(max_error_deg, 4)} degrees")
        else:
            print("Failed to start movement!")

    except FileNotFoundError:
        print(f"Could not find saved joints file: {filename}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move robot to saved joint positions")
    parser.add_argument("filename", nargs="?", default="current_joints.json",
                        help="JSON file with joint_positions (default: current_joints.json)")
    args = parser.parse_args()
    move_to_saved_joints(args.filename)
