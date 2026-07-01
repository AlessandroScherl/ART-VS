#!/usr/bin/env python3
"""Apply a saved TCP pose to the UR5 robot via MoveIt2."""

import json
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from moveit_robot_interface import MoveItRobotInterface


def move_to_saved_pose(filename="current_pose.json"):
    print("Connecting to robot via MoveIt2...")
    robot = MoveItRobotInterface()

    try:
        # Load saved pose
        with open(filename, 'r') as f:
            pose_data = json.load(f)

        position = pose_data['position']
        quaternion = pose_data['orientation']

        # Convert quaternion to rotation vector (which UR robots use)
        r = R.from_quat(quaternion)
        rotation_vector = r.as_rotvec()

        # Combine into target pose [x, y, z, rx, ry, rz]
        target_pose = position + list(rotation_vector)

        print("\nLoaded target pose:")
        print(f"Position (x,y,z): {[round(x, 3) for x in position]}")
        print(f"Quaternion (x,y,z,w): {[round(x, 3) for x in quaternion]}")
        print(f"Rotation vector: {[round(x, 3) for x in rotation_vector]}")

        # Get current pose
        current_pose = robot.get_tcp_pose()
        if current_pose is not None:
            current_position = current_pose[:3]
            current_rotvec = current_pose[3:]
            current_rot = R.from_rotvec(current_rotvec)
            current_quat = current_rot.as_quat()

            print(f"\nCurrent pose:")
            print(f"Position (x,y,z): {[round(x, 3) for x in current_position]}")
            print(f"Quaternion (x,y,z,w): {[round(x, 3) for x in current_quat]}")
            print(f"Rotation vector: {[round(x, 3) for x in current_rotvec]}")

        # Move to pose
        print("\nMoving to target pose...")
        success = robot.move_to_tcp_pose(target_pose, speed=0.1, accel=0.1)
        if success:
            print("Movement completed!")

            # Print final pose
            final_pose = robot.get_tcp_pose()
            if final_pose is not None:
                final_position = final_pose[:3]
                final_rotvec = final_pose[3:]
                final_rot = R.from_rotvec(final_rotvec)
                final_quat = final_rot.as_quat()

                print(f"\nFinal pose reached:")
                print(f"Position (x,y,z): {[round(x, 3) for x in final_position]}")
                print(f"Quaternion (x,y,z,w): {[round(x, 3) for x in final_quat]}")
                print(f"Rotation vector: {[round(x, 3) for x in final_rotvec]}")
        else:
            print("Failed to reach target pose!")

    except FileNotFoundError:
        print(f"Could not find saved pose file: {filename}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.destroy()


if __name__ == "__main__":
    move_to_saved_pose()
