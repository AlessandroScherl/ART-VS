#!/usr/bin/env python3
"""Save the current robot joint positions to a JSON file via MoveIt2.

Usage:
    python3 save_current_joint_pose.py                       # → current_joints.json
    python3 save_current_joint_pose.py --output place_joints.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from moveit_robot_interface import MoveItRobotInterface


def save_current_joints(filename="current_joints.json"):
    print("Connecting to robot via MoveIt2...")
    robot = MoveItRobotInterface()

    try:
        # Wait for first joint_state message (pymoveit2 subscriber needs a moment)
        print("Waiting for /joint_states...")
        deadline = time.monotonic() + 8.0
        current_joints = None
        while time.monotonic() < deadline:
            current_joints = robot.get_joint_positions()
            if current_joints is not None:
                break
            time.sleep(0.1)

        if current_joints is None:
            print("ERROR: Could not read joint positions (no /joint_states received within 8s)")
            print("       Is the UR5 driver running?")
            return

        # Create joints dictionary
        joints_data = {
            'joint_positions': list(current_joints)
        }

        print(f"\nCurrent joint positions:")
        print(f"Radians: {[round(x, 4) for x in current_joints]}")
        print(f"Degrees: {[round(np.degrees(x), 4) for x in current_joints]}")

        # Save to file
        with open(filename, 'w') as f:
            json.dump(joints_data, f, indent=2)

        print(f"\nJoint positions saved to {filename}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save current robot joints to JSON")
    parser.add_argument("--output", "-o", default="current_joints.json",
                        help="Output JSON file (default: current_joints.json)")
    args = parser.parse_args()
    save_current_joints(args.output)
