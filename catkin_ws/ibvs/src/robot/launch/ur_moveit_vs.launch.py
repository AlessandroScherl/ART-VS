"""
Wrapper launch file for MoveIt + Servo with custom VS servo config.

Usage:
  ros2 launch /path/to/ur_moveit_vs.launch.py ur_type:=ur5

This includes the standard ur_moveit.launch.py with launch_servo:=false,
then launches its own servo_node using ur_servo_vs.yaml (no Butterworth
filter, relaxed collision thresholds).
"""
import os
import yaml

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Resolve paths
    ur_moveit_launch = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py",
    )

    # Custom servo YAML — co-located with this launch file
    servo_yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "ur_servo_vs.yaml",
    )
    with open(servo_yaml_path) as f:
        servo_yaml = yaml.safe_load(f)
    servo_params = {"moveit_servo": servo_yaml}

    ur_type = LaunchConfiguration("ur_type")

    ld = LaunchDescription()

    # Declare arguments forwarded to ur_moveit.launch.py
    ld.add_action(
        DeclareLaunchArgument("ur_type", description="UR robot type (e.g. ur5)")
    )
    ld.add_action(
        DeclareLaunchArgument("launch_rviz", default_value="true")
    )

    # Include standard MoveIt launch — but suppress its built-in servo node
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ur_moveit_launch),
            launch_arguments={
                "ur_type": ur_type,
                "launch_servo": "false",  # We launch our own servo below
                "launch_rviz": LaunchConfiguration("launch_rviz"),
            }.items(),
        )
    )

    # Build MoveIt config for servo_node parameters
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(Path("srdf") / "ur.srdf.xacro", {"name": ur_type})
        .to_moveit_configs()
    )

    # Launch servo_node with our custom VS config
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            servo_params,
        ],
    )
    ld.add_action(servo_node)

    return ld
