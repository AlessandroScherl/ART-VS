#!/usr/bin/env python3
"""Save the current robot TCP pose to a JSON file via MoveIt2/TF2."""

import json
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from moveit_robot_interface import MoveItRobotInterface


def save_current_pose(filename="current_pose.json"):
    print("Connecting to robot via MoveIt2...")
    robot = MoveItRobotInterface()

    try:
        # Get current pose
        current_pose = robot.get_tcp_pose()
        if current_pose is None:
            print("ERROR: Could not read TCP pose from TF2")
            return

        position = current_pose[:3]
        rotation_vector = current_pose[3:]

        # Convert rotation vector to quaternion
        r = R.from_rotvec(rotation_vector)
        quaternion = r.as_quat()  # Returns [x, y, z, w]

        # Create pose dictionary
        pose_data = {
            'position': list(position),
            'orientation': list(quaternion)
        }

        print(f"\nCurrent pose:")
        print(f"Position (x,y,z): {[round(x, 3) for x in position]}")
        print(f"Quaternion (x,y,z,w): {[round(x, 3) for x in quaternion]}")

        # Also print Euler angles for reference
        euler = r.as_euler('xyz', degrees=True)
        print(f"Euler angles (xyz, degrees): {[round(x, 3) for x in euler]}")

        # Save to file
        with open(filename, 'w') as f:
            json.dump(pose_data, f, indent=2)

        print(f"\nPose saved to {filename}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.destroy()


if __name__ == "__main__":
    save_current_pose()
