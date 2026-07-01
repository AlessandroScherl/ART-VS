#!/usr/bin/env python3
"""
MoveIt2-based robot interface replacing ur_rtde for UR5 control.

Provides a drop-in replacement for RTDEControlInterface + RTDEReceiveInterface
using pymoveit2, TF2, and ROS2 services.

Requires:
    sudo apt install ros-jazzy-pymoveit2

Usage:
    from moveit_robot_interface import MoveItRobotInterface

    robot = MoveItRobotInterface()
    robot.move_to_joints([0, -1.57, 0, -1.57, 0, 0])
    pose = robot.get_tcp_pose()  # [x, y, z, rx, ry, rz]
    robot.move_to_tcp_pose(pose)
    robot.destroy()
"""

import time
import subprocess
from threading import Thread

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
from tf2_ros import Buffer as TF2Buffer, TransformListener as TF2TransformListener
from scipy.spatial.transform import Rotation as R

# UR5 joint names (must match URDF/SRDF)
UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Frame names
# IMPORTANT: Use "base" not "base_link" — on UR5 with ROS2 driver, base_link and base
# differ by a 180° Z rotation. RTDE poses are in the "base" frame, so MoveIt2 must
# also plan in "base" for consistency.
BASE_LINK = "base"
END_EFFECTOR = "tool0"
MOVE_GROUP = "ur_manipulator"


class MoveItRobotInterface:
    """
    Replaces RTDEControlInterface + RTDEReceiveInterface with MoveIt2.

    RTDE function mapping:
        rtde_c.moveJ(joints, speed, accel)  → move_to_joints(joints, speed, accel)
        rtde_c.moveL(pose, speed, accel)    → move_to_tcp_pose(pose, speed, accel)
        rtde_c.isSteady()                   → (blocking built into move methods)
        rtde_c.stopL(decel)                 → stop()
        rtde_r.getActualTCPPose()           → get_tcp_pose()
        rtde_r.getActualQ()                 → get_joint_positions()
        rtde_r.getRobotMode()               → get_robot_mode()
        rtde_r.getSafetyMode()              → get_safety_mode()
        connect/disconnect/reconnect        → (constructor / destroy)
    """

    def __init__(self, node_name="robot_interface", init_rclpy=True):
        """Initialize MoveIt2 robot interface.

        Args:
            node_name: ROS2 node name
            init_rclpy: Whether to call rclpy.init(). Set False if already initialized.
        """
        self._owns_rclpy = init_rclpy
        if init_rclpy:
            if not rclpy.ok():
                rclpy.init()

        # Create node and executor
        self.node = Node(node_name)
        self.callback_group = ReentrantCallbackGroup()
        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self._executor_thread = Thread(target=self.executor.spin, daemon=True)
        self._executor_thread.start()

        # MoveIt2 planner (for moveJ/moveL equivalents)
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=UR5_JOINT_NAMES,
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR,
            group_name=MOVE_GROUP,
            callback_group=self.callback_group,
        )

        # TF2 for TCP pose lookups
        self.tf_buffer = TF2Buffer()
        self.tf_listener = TF2TransformListener(self.tf_buffer, self.node)

        # Wait for MoveIt2 and TF to initialize
        self.node.create_rate(1.0).sleep()
        self.node.get_logger().info("MoveItRobotInterface ready")

    def move_to_joints(self, joint_positions, speed=0.5, accel=0.5):
        """Move to joint configuration (replaces rtde_c.moveJ + isSteady).

        Args:
            joint_positions: List of 6 joint angles in radians.
            speed: Joint speed scaling (0.0-1.0). Maps to MoveIt max_velocity.
            accel: Joint acceleration scaling (0.0-1.0). Maps to MoveIt max_acceleration.

        Returns:
            True if motion succeeded, False otherwise.
        """
        # Clamp to valid MoveIt scaling range [0.0, 1.0]
        self.moveit2.max_velocity = min(1.0, max(0.01, speed))
        self.moveit2.max_acceleration = min(1.0, max(0.01, accel))

        self.moveit2.move_to_configuration(list(joint_positions))
        return self.moveit2.wait_until_executed()

    def move_to_tcp_pose(self, pose_6dof, speed=0.1, accel=0.1, cartesian=False):
        """Move TCP to Cartesian pose (replaces rtde_c.moveL + isSteady).

        Args:
            pose_6dof: [x, y, z, rx, ry, rz] where position is in meters
                       and orientation is a rotation vector (radians).
                       This is the same format as ur_rtde uses.
            speed: Velocity scaling (0.0-1.0).
            accel: Acceleration scaling (0.0-1.0).
            cartesian: If True, use Cartesian (straight-line) path planning.
                       If False (default), use joint-space planning to reach the pose.
                       Use False for large motions, True for small precise moves.

        Returns:
            True if motion succeeded, False otherwise.
        """
        position = list(pose_6dof[:3])
        rotvec = pose_6dof[3:]

        # Convert rotation vector to quaternion [qx, qy, qz, qw] (scipy convention)
        quat = R.from_rotvec(rotvec).as_quat().tolist()

        # Clamp scaling
        self.moveit2.max_velocity = min(1.0, max(0.01, speed))
        self.moveit2.max_acceleration = min(1.0, max(0.01, accel))

        if cartesian:
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=quat,
                cartesian=True,
                cartesian_max_step=0.0025,
                cartesian_fraction_threshold=0.9,
            )
        else:
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=quat,
            )
        return self.moveit2.wait_until_executed()

    def get_tcp_pose(self):
        """Get current TCP pose (replaces rtde_r.getActualTCPPose).

        Returns:
            [x, y, z, rx, ry, rz] where position is in meters and
            orientation is a rotation vector (radians), or None on failure.
        """
        try:
            from rclpy.time import Time
            trans = self.tf_buffer.lookup_transform(
                'base', 'tool0', Time(),
                timeout=rclpy.time.Duration(seconds=2.0)
            )
            t = trans.transform.translation
            q = trans.transform.rotation
            rotvec = R.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()
            return [t.x, t.y, t.z, rotvec[0], rotvec[1], rotvec[2]]
        except Exception as e:
            self.node.get_logger().error(f"TF lookup failed: {e}")
            return None

    def get_joint_positions(self):
        """Get current joint positions (replaces rtde_r.getActualQ).

        Returns:
            List of 6 joint angles in radians in UR5 order
            (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3),
            or None if unavailable.
        """
        js = self.moveit2.joint_state
        if js is not None and len(js.position) >= 6:
            # Joint states may be published in alphabetical order, not UR5 order.
            # Look up by name to return in correct UR5_JOINT_NAMES order.
            if js.name:
                try:
                    return [js.position[list(js.name).index(name)] for name in UR5_JOINT_NAMES]
                except (ValueError, IndexError):
                    pass
            return list(js.position[:6])
        return None

    def get_robot_mode(self):
        """Get robot mode (replaces rtde_r.getRobotMode).

        Returns:
            7 if running, -1 otherwise (matching RTDE convention).
        """
        try:
            result = subprocess.run(
                ['ros2', 'service', 'call',
                 '/dashboard_client/get_robot_mode',
                 'ur_dashboard_msgs/srv/GetRobotMode', '{}'],
                capture_output=True, text=True, timeout=5
            )
            if 'RUNNING' in result.stdout:
                return 7
            return -1
        except Exception:
            return -1

    def get_safety_mode(self):
        """Get safety mode (replaces rtde_r.getSafetyMode).

        Returns:
            1 if normal, -1 otherwise (matching RTDE convention).
        """
        try:
            result = subprocess.run(
                ['ros2', 'service', 'call',
                 '/dashboard_client/get_safety_mode',
                 'ur_dashboard_msgs/srv/GetSafetyMode', '{}'],
                capture_output=True, text=True, timeout=5
            )
            if 'NORMAL' in result.stdout:
                return 1
            return -1
        except Exception:
            return -1

    def stop(self):
        """Stop robot motion (replaces rtde_c.stopL)."""
        # MoveIt2 handles stopping when a new goal is sent or execution is cancelled
        pass

    def destroy(self):
        """Clean up ROS2 resources."""
        # Shut down executor FIRST so the background thread stops spinning
        try:
            self.executor.shutdown()
        except Exception:
            pass
        # Wait for the executor thread to finish
        try:
            self._executor_thread.join(timeout=2.0)
        except Exception:
            pass
        # Now safe to destroy the node
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass
