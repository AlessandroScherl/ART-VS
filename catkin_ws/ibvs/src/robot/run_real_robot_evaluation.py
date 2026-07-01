#!/usr/bin/env python3
"""
Real Robot Visual Servoing Evaluation Script

Automates the evaluation process for visual servoing on a UR5 robot with RealSense D435i camera.
Iterates through pre-sampled poses, runs visual servoing, performs grasping, and logs results.

Prerequisites (start manually in separate terminals):
    Terminal 1: UR5 Driver
        ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=<ROBOT_IP> launch_rviz:=false

    Terminal 2: MoveIt + Servo
        ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_servo:=true

    Terminal 3: Camera
        ros2 launch realsense2_camera rs_launch.py enable_depth:=true enable_color:=true \
            rgb_camera.color_profile:=1920x1080x6 depth_module.depth_profile:=1280x720x15 align_depth.enable:=true

    Terminal 4: Cropper
        cd catkin_ws/ibvs/src/robot && python3 crop_image_node.py

    Terminal 5: Evaluation
        cd catkin_ws/ibvs/src/robot && python3 run_real_robot_evaluation.py

Usage:
    python3 run_real_robot_evaluation.py
    python3 run_real_robot_evaluation.py --start-index 5
    python3 run_real_robot_evaluation.py --dry-run
    python3 run_real_robot_evaluation.py --skip-rotation
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np

# Add parent directory to path for imports
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# MoveIt2 robot interface (replaces ur_rtde)
try:
    from moveit_robot_interface import MoveItRobotInterface
except ImportError:
    print("ERROR: moveit_robot_interface not found. Ensure moveit_robot_interface.py is in the same directory.")
    print("       Also install: sudo apt install ros-jazzy-pymoveit2")
    sys.exit(1)

# Gripper import
from gripper import Gripper


# ============================================================================
# Configuration
# ============================================================================

GRIPPER_PORT = "/dev/ttyUSB0"

# Motion parameters
MOVE_L_SPEED = 0.1    # m/s (conservative for Cartesian motion)
MOVE_L_ACCEL = 0.1    # m/s^2
MOVE_J_SPEED = 0.5    # rad/s (faster for joint motion home)
MOVE_J_ACCEL = 0.5    # rad/s^2

# Visual servoing timeout
VS_TIMEOUT = 300  # seconds

# Default paths (relative to this script's directory)
DEFAULT_POSES_FILE = "sampled_poses.json"
DEFAULT_JOINTS_FILE = "current_joints.json"
DEFAULT_VS_CONFIG = "../visual_servoing/configs/real_robot/config_real_robot_dinov3.yaml"


# ============================================================================
# Evaluation Runner Class
# ============================================================================

class RealRobotEvaluationRunner:
    """Manages the automated evaluation process."""

    def __init__(self, args):
        self.args = args
        self.interrupted = False
        self.results = {
            "config": str(args.config),
            "poses_file": str(args.poses),
            "timestamp": datetime.now().isoformat(),
            "total_trials": 0,
            "completed_trials": 0,
            "trials": [],
            "summary": {}
        }

        # Robot interface (MoveIt2-based, replaces RTDE)
        self.robot = None
        self.gripper = None

        # Poses data
        self.poses = []
        self.home_joints = []

        # Rosbag recording
        self.rosbag_folder = None
        self.rosbag_process = None

        # Register signal handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n[INTERRUPT] Received Ctrl+C. Stopping robot...")
        self.interrupted = True
        self._stop_rosbag_recording()
        self._stop_robot()
        # Note: Results are only saved at the end of a complete run, not on interrupt
        sys.exit(0)

    def _stop_robot(self):
        """Stop robot motion."""
        if self.robot is not None:
            try:
                self.robot.stop()
                print("  Robot stopped.")
            except Exception as e:
                print(f"  Warning: Could not stop robot: {e}")

    def _create_rosbag_folder(self):
        """Create folder for rosbag recordings."""
        if self.args.no_rosbag:
            print("  Rosbag recording disabled (--no-rosbag flag)")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rosbag_folder = current_dir / f"rosbags_{timestamp}"
        self.rosbag_folder.mkdir(exist_ok=True)
        print(f"  Rosbag folder: {self.rosbag_folder}")

    def _start_rosbag_recording(self, trial_index):
        """Start rosbag recording for a trial.

        Args:
            trial_index: The pose/trial index for naming the bag file

        Returns:
            Path to the bag file, or None if recording is disabled
        """
        if self.args.no_rosbag or self.rosbag_folder is None:
            return None

        topics = [
            "/camera/color/image_cropped",
            "/correspondence_visualization",
            "/correspondence_visualization_crop",
            "/vs/vit_space_correspondences",
            "/vs/vit_space_correspondences_tiled"
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_name = f"trial_{trial_index:02d}_{timestamp}.bag"
        bag_path = self.rosbag_folder / bag_name

        cmd = ["ros2", "bag", "record", "-o", str(bag_path)] + topics

        try:
            self.rosbag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"    Rosbag recording started: {bag_name}")
            # Give rosbag a moment to initialize
            time.sleep(0.5)
            return bag_path
        except Exception as e:
            print(f"    WARNING: Failed to start rosbag recording: {e}")
            self.rosbag_process = None
            return None

    def _stop_rosbag_recording(self):
        """Stop rosbag recording."""
        if self.rosbag_process is not None:
            try:
                self.rosbag_process.terminate()
                self.rosbag_process.wait(timeout=5)
                print("    Rosbag recording stopped.")
            except subprocess.TimeoutExpired:
                self.rosbag_process.kill()
                print("    Rosbag recording killed (timeout).")
            except Exception as e:
                print(f"    WARNING: Error stopping rosbag: {e}")
            finally:
                self.rosbag_process = None

    def connect(self):
        """Connect to robot via MoveIt2 and gripper."""
        print(f"\n[1/4] Connecting to robot via MoveIt2...")
        try:
            self.robot = MoveItRobotInterface()
            print("  MoveIt2 robot interface ready.")
        except Exception as e:
            print(f"  ERROR: Failed to initialize MoveIt2: {e}")
            return False

        if not self.args.dry_run:
            print(f"\n[2/4] Connecting to gripper at {GRIPPER_PORT}...")
            try:
                self.gripper = Gripper(port=GRIPPER_PORT)
                print("  Gripper connected.")
            except Exception as e:
                print(f"  WARNING: Failed to connect to gripper: {e}")
                print("  Continuing without gripper (dry-run mode for gripper)")
                self.gripper = None

        return True

    def load_poses(self):
        """Load poses from JSON file."""
        poses_path = current_dir / self.args.poses
        print(f"\n[3/4] Loading poses from {poses_path}...")

        try:
            with open(poses_path, 'r') as f:
                data = json.load(f)

            self.poses = data['poses']
            params = data['parameters']

            # Load home joint positions
            if 'zero_joints' in params:
                self.home_joints = params['zero_joints']
            else:
                # Fallback to current_joints.json
                joints_path = current_dir / self.args.joints
                with open(joints_path, 'r') as f:
                    joints_data = json.load(f)
                self.home_joints = joints_data['joint_positions']

            print(f"  Loaded {len(self.poses)} poses")
            print(f"  Home joints: {[round(j, 3) for j in self.home_joints]}")

            # Apply start/end index filters
            start = self.args.start_index
            end = self.args.end_index if self.args.end_index is not None else len(self.poses)

            if start >= len(self.poses):
                print(f"  ERROR: start-index {start} >= total poses {len(self.poses)}")
                return False

            self.poses = self.poses[start:end]
            self.results["total_trials"] = len(self.poses)

            print(f"  Running poses {start} to {end-1} ({len(self.poses)} trials)")
            return True

        except FileNotFoundError:
            print(f"  ERROR: Poses file not found: {poses_path}")
            return False
        except Exception as e:
            print(f"  ERROR: Failed to load poses: {e}")
            return False

    def verify_ros_topics(self):
        """Verify that camera topics are available using ros2 CLI."""
        print("\n[4/4] Verifying ROS2 topics...")
        print("  NOTE: Camera, cropper, UR driver, and MoveIt Servo must be started manually!")
        print("  Expected topics:")
        print("    - /camera/color/image_cropped")
        print("    - /camera/aligned_depth_to_color/image_cropped")
        print("    - /servo_node/delta_twist_cmds")

        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True, text=True, timeout=5
            )
            topic_names = result.stdout.strip().split('\n')

            rgb_ok = '/camera/color/image_cropped' in topic_names
            depth_ok = '/camera/aligned_depth_to_color/image_cropped' in topic_names
            servo_ok = '/servo_node/delta_twist_cmds' in topic_names

            if rgb_ok and depth_ok and servo_ok:
                print("  All topics verified OK.")
                return True
            else:
                if not rgb_ok:
                    print("  WARNING: /camera/color/image_cropped not found")
                if not depth_ok:
                    print("  WARNING: /camera/aligned_depth_to_color/image_cropped not found")
                if not servo_ok:
                    print("  WARNING: /servo_node/delta_twist_cmds not found")
                print("\n  Did you start the camera, cropper, and MoveIt Servo?")
                print("  Continue anyway? [y/N]: ", end="")
                response = input().strip().lower()
                return response == 'y'
        except Exception as e:
            print(f"  WARNING: Could not verify topics: {e}")
            print("  Assuming they are running. Continue? [y/N]: ", end="")
            response = input().strip().lower()
            return response == 'y'

    def _switch_to_servo_controller(self):
        """Switch to forward_position_controller for MoveIt Servo.

        Deactivates scaled_joint_trajectory_controller and activates
        forward_position_controller, then enables twist commands on Servo.
        """
        print("\n  Switching to Servo controller...")
        try:
            subprocess.run([
                'ros2', 'control', 'switch_controllers',
                '--deactivate', 'scaled_joint_trajectory_controller',
                '--activate', 'forward_position_controller'
            ], check=True, capture_output=True, text=True, timeout=10)
            print("    Controllers switched (forward_position_controller active)")

            # Enable twist commands on Servo
            subprocess.run([
                'ros2', 'service', 'call', '/servo_node/switch_command_type',
                'moveit_msgs/srv/ServoCommandType', '{command_type: 1}'
            ], check=True, capture_output=True, text=True, timeout=10)
            print("    Servo twist commands enabled")
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Controller switch failed: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("    ERROR: Controller switch timed out")
            return False

    def _switch_to_trajectory_controller(self):
        """Switch back to scaled_joint_trajectory_controller for MoveIt2 planning.

        Deactivates forward_position_controller and activates
        scaled_joint_trajectory_controller for moveJ/moveL via MoveIt2.
        """
        print("\n  Switching to trajectory controller...")
        try:
            subprocess.run([
                'ros2', 'control', 'switch_controllers',
                '--deactivate', 'forward_position_controller',
                '--activate', 'scaled_joint_trajectory_controller'
            ], check=True, capture_output=True, text=True, timeout=10)
            print("    Controllers switched (scaled_joint_trajectory_controller active)")
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Controller switch failed: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("    ERROR: Controller switch timed out")
            return False

    def move_to_pose(self, pose, pose_index):
        """Move robot to a TCP pose using MoveIt2 Cartesian planning."""
        print(f"\n  Moving to pose {pose_index}...")
        print(f"    Position: X={pose[0]:.3f}, Y={pose[1]:.3f}, Z={pose[2]:.3f}")

        if self.args.dry_run:
            print("    [DRY RUN] Skipping actual motion")
            time.sleep(1)
            return True

        try:
            success = self.robot.move_to_tcp_pose(pose, speed=MOVE_L_SPEED, accel=MOVE_L_ACCEL, cartesian=True)
            if not success:
                print("    ERROR: move_to_tcp_pose failed (pose may be unreachable)")
                return False

            # Verify pose reached
            actual = self.robot.get_tcp_pose()
            if actual is not None:
                pos_error = np.linalg.norm(np.array(actual[:3]) - np.array(pose[:3])) * 1000
                print(f"    Pose reached (position error: {pos_error:.1f} mm)")
            return True

        except Exception as e:
            print(f"    ERROR: Motion failed: {e}")
            return False

    def _refresh_robot_interface(self):
        """Rebuild self.robot with a fresh MoveItRobotInterface.

        After VS + Servo + controller switches, pymoveit2's internal state
        (joint_state, planning scene, action clients) becomes stale, causing
        STATUS_ABORTED on subsequent planning requests.  Destroying and
        recreating the interface gives us a clean slate.
        """
        print("    Refreshing MoveIt interface (clean state for planning)...")
        try:
            if self.robot is not None:
                # Prevent old instance from calling rclpy.shutdown()
                self.robot._owns_rclpy = False
                self.robot.destroy()
        except Exception:
            pass

        self.robot = MoveItRobotInterface(node_name=f'robot_interface_{int(time.monotonic()*1000)}', init_rclpy=False)

        # Wait for joint states to populate
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            js = self.robot.moveit2.joint_state
            if js is not None and len(js.position) >= 6:
                break
            time.sleep(0.1)
        time.sleep(0.3)
        print("    MoveIt interface refreshed.")

    def return_to_home(self):
        """Return robot to home position using MoveIt2 joint planning."""
        print("\n  Returning to home position...")

        if self.args.dry_run:
            print("    [DRY RUN] Skipping actual motion")
            time.sleep(1)
            return True

        try:
            # Check robot state before moving
            try:
                robot_mode = self.robot.get_robot_mode()
                safety_mode = self.robot.get_safety_mode()
                print(f"    Robot mode: {robot_mode}, Safety mode: {safety_mode}")

                # Robot mode 7 = Running, Safety mode 1 = Normal
                if robot_mode != 7:
                    print(f"    WARNING: Robot not in running mode (mode={robot_mode})")
                if safety_mode != 1:
                    print(f"    WARNING: Robot not in normal safety mode (safety={safety_mode})")
            except Exception as e:
                print(f"    WARNING: Could not check robot state: {e}")

            success = self.robot.move_to_joints(self.home_joints, speed=MOVE_J_SPEED, accel=MOVE_J_ACCEL)
            if not success:
                print("    ERROR: move_to_joints returned False")
                print("    Possible causes:")
                print("      - Robot in protective stop (check teach pendant)")
                print("      - Emergency stop active")
                print("      - Joint limits exceeded")
                print("      - MoveIt2 planning failure")
                return False

            print("    Home position reached.")
            return True

        except Exception as e:
            print(f"    ERROR: Motion failed: {e}")
            return False

    def run_visual_servoing(self):
        """Run visual servoing as a subprocess with real-time output streaming."""
        print("\n  Starting visual servoing...")

        if self.args.dry_run:
            print("    [DRY RUN] Simulating VS (3 seconds)")
            time.sleep(3)
            return {
                "converged": True,
                "iterations": 100,
                "error": None
            }

        # Build command
        vs_script = current_dir.parent / "visual_servoing" / "run_visual_servoing.py"
        config_path = current_dir / self.args.config
        if not config_path.is_absolute():
            config_path = current_dir / self.args.config

        cmd = [
            sys.executable, "-u",  # Unbuffered output for real-time streaming
            str(vs_script),
            "--config", str(config_path)
        ]

        if self.args.skip_rotation:
            cmd.append("--skip-rotation")

        print(f"    Command: {' '.join(cmd)}")
        print("\n" + "=" * 60)
        print("    VISUAL SERVOING OUTPUT:")
        print("=" * 60 + "\n")

        try:
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                cwd=str(current_dir.parent / "visual_servoing")
            )

            # Stream output in real-time while capturing it for parsing
            output_lines = []
            converged = False
            iterations = 0
            start_time = time.time()

            # Read output line by line
            for line in process.stdout:
                # Print with indent for better readability
                print(f"    {line}", end='')
                output_lines.append(line)

                # Parse convergence from explicit EVAL_ print statements
                if "EVAL_CONVERGED:" in line:
                    converged = "True" in line
                elif "EVAL_ITERATIONS:" in line:
                    try:
                        iterations = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

                # Also check for traditional convergence messages as fallback
                if "Converged: True" in line:
                    converged = True
                elif "Converged: False" in line:
                    converged = False
                elif "Iterations:" in line and "EVAL_" not in line:
                    try:
                        iterations = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

                # Check timeout
                if time.time() - start_time > VS_TIMEOUT:
                    process.kill()
                    print(f"\n    ERROR: VS timeout after {VS_TIMEOUT}s")
                    return {
                        "converged": False,
                        "iterations": iterations,
                        "error": "Timeout"
                    }

            # Wait for process to complete
            process.wait()

            print("\n" + "=" * 60)
            print("    END VISUAL SERVOING OUTPUT")
            print("=" * 60 + "\n")

            return {
                "converged": converged,
                "iterations": iterations,
                "error": None if process.returncode == 0 else f"Exit code: {process.returncode}"
            }

        except Exception as e:
            print(f"\n    ERROR: VS failed: {e}")
            return {
                "converged": False,
                "iterations": 0,
                "error": str(e)
            }

    def close_gripper(self):
        """Close the gripper."""
        print("\n  Closing gripper...")

        if self.args.dry_run or self.gripper is None:
            print("    [DRY RUN/NO GRIPPER] Skipping gripper close")
            return True

        try:
            self.gripper.close_gripper()
            return True
        except Exception as e:
            print(f"    ERROR: Gripper close failed: {e}")
            return False

    def open_gripper(self):
        """Open the gripper."""
        print("\n  Opening gripper...")

        if self.args.dry_run or self.gripper is None:
            print("    [DRY RUN/NO GRIPPER] Skipping gripper open")
            return True

        try:
            self.gripper.open_gripper()
            return True
        except Exception as e:
            print(f"    ERROR: Gripper open failed: {e}")
            return False

    def run_gripping_motion(self):
        """Move TCP 8cm along tool0 Z-axis using MoveIt Cartesian planning.

        Switches to scaled_joint_trajectory_controller, plans a straight-line
        move along tool0 Z, executes it.  Controller stays on trajectory
        mode afterward (ready for return-to-home).
        """
        print("\n  Running gripping motion (8cm along tool0 Z via MoveIt planning)...")

        if self.args.dry_run:
            print("    [DRY RUN] Skipping gripping motion")
            time.sleep(1)
            return True

        try:
            self._switch_to_trajectory_controller()

            # Create a FRESH MoveItRobotInterface for gripping.
            # Reusing self.robot (which survived VS + Servo + controller switches)
            # causes STATUS_ABORTED due to stale pymoveit2 internal state.
            print("    Creating fresh MoveIt interface for gripping...")
            grip_robot = MoveItRobotInterface(node_name=f'gripping_planner_{int(time.monotonic()*1000)}', init_rclpy=False)

            # Wait for joint states (background executor populates them)
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                js = grip_robot.moveit2.joint_state
                if js is not None and len(js.position) >= 6:
                    print(f"    Joint states OK")
                    break
                time.sleep(0.1)
            time.sleep(0.3)

            start_pose = grip_robot.get_tcp_pose()
            if start_pose is None:
                print("    ERROR: Could not get current TCP pose")
                grip_robot.destroy()
                return False

            # Compute target: move 8cm along tool0 Z-axis
            from scipy.spatial.transform import Rotation as Rot
            tool0_rot = Rot.from_rotvec(start_pose[3:])
            tool0_z = tool0_rot.as_matrix()[:, 2]

            target_pose = list(start_pose)
            target_pose[0] += 0.08 * tool0_z[0]
            target_pose[1] += 0.08 * tool0_z[1]
            target_pose[2] += 0.08 * tool0_z[2]

            print(f"    Start pos:  [{start_pose[0]:.4f}, {start_pose[1]:.4f}, {start_pose[2]:.4f}]")
            print(f"    Tool0 Z:    [{tool0_z[0]:.3f}, {tool0_z[1]:.3f}, {tool0_z[2]:.3f}]")
            print(f"    Target pos: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")

            success = grip_robot.move_to_tcp_pose(target_pose, speed=0.3, accel=0.3, cartesian=True)

            if not success:
                print("    Cartesian failed, retrying...")
                time.sleep(2.0)
                success = grip_robot.move_to_tcp_pose(target_pose, speed=0.3, accel=0.3, cartesian=True)

            if not success:
                print("    WARNING: Cartesian planning failed — skipping gripping motion")

            final_pose = grip_robot.get_tcp_pose()
            if final_pose is not None and start_pose is not None:
                dist = np.linalg.norm(np.array(final_pose[:3]) - np.array(start_pose[:3])) * 1000
                print(f"    Moved {dist:.1f}mm. Gripping motion complete.")

            time.sleep(0.1)
            grip_robot.destroy()
            return success

        except Exception as e:
            print(f"    ERROR: Gripping motion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_single_trial(self, pose, pose_index, trial_num, total_trials):
        """Run a single evaluation trial."""
        print("\n" + "=" * 70)
        print(f"TRIAL {trial_num}/{total_trials} (Pose index: {pose_index})")
        print("=" * 70)

        trial_result = {
            "index": pose_index,
            "pose": pose,
            "converged": False,
            "iterations": 0,
            "status": "failed",
            "error_message": None
        }

        # Step 1: Move to initial pose (uses RTDE with trajectory controller)
        if not self.move_to_pose(pose, pose_index):
            trial_result["error_message"] = "Failed to reach initial pose"
            return trial_result

        # Wait for stabilization
        time.sleep(0.5)

        # Step 1.5: Switch to Servo controller for VS
        print("\n" + "-" * 50)
        print("  Robot at initial pose. Switching to Servo controller...")
        if not self._switch_to_servo_controller():
            trial_result["error_message"] = "Failed to switch to Servo controller"
            return trial_result
        time.sleep(0.2)

        # Step 1.6: Start rosbag recording
        bag_path = self._start_rosbag_recording(pose_index)
        if bag_path:
            trial_result["rosbag_file"] = str(bag_path)

        # Step 2: Run visual servoing
        vs_result = self.run_visual_servoing()
        trial_result["converged"] = vs_result["converged"]
        trial_result["iterations"] = vs_result["iterations"]

        if vs_result["error"]:
            trial_result["error_message"] = vs_result["error"]

        if vs_result["converged"]:
            print(f"\n  VS CONVERGED in {vs_result['iterations']} iterations")
        else:
            print(f"\n  VS DID NOT CONVERGE ({vs_result['iterations']} iterations)")

        # Step 2.5: Run gripping motion (switches to trajectory controller internally)
        print("\n" + "-" * 50)
        print("  VS completed. Running gripping motion...")
        self.run_gripping_motion()

        # Step 3: Close gripper (grasp bottle)
        self.close_gripper()

        # Step 3.1: Stop rosbag recording
        self._stop_rosbag_recording()

        # Step 3.5: Trajectory controller already active from gripping motion.
        # Refresh self.robot — pymoveit2 state is stale after VS + Servo.
        self._refresh_robot_interface()

        # Step 4: Return to home
        if not self.return_to_home():
            trial_result["error_message"] = "Failed to return home"
            return trial_result

        # Step 5: Open gripper (release bottle)
        self.open_gripper()

        trial_result["status"] = "completed"
        return trial_result

    def _save_results(self):
        """Save results to JSON file."""
        # Calculate summary
        completed = [t for t in self.results["trials"] if t["status"] == "completed"]
        skipped = [t for t in self.results["trials"] if t["status"] == "skipped"]
        aborted = [t for t in self.results["trials"] if t["status"] == "aborted"]
        converged = [t for t in completed if t["converged"]]

        self.results["completed_trials"] = len(completed)
        self.results["summary"] = {
            "convergence_rate": len(converged) / len(completed) * 100 if completed else 0,
            "mean_iterations": np.mean([t["iterations"] for t in converged]) if converged else 0,
            "total_converged": len(converged),
            "total_failed": len(completed) - len(converged),
            "total_skipped": len(skipped),
            "total_aborted": len(aborted)
        }

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = current_dir / f"evaluation_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def run(self):
        """Run the full evaluation."""
        print("\n" + "=" * 70)
        print("REAL ROBOT VISUAL SERVOING EVALUATION")
        print("=" * 70)

        if self.args.dry_run:
            print("\n*** DRY RUN MODE - No actual robot motion ***\n")

        # Connect to robot and gripper
        if not self.connect():
            return

        # Load poses
        if not self.load_poses():
            return

        # Create rosbag folder for recordings
        self._create_rosbag_folder()

        # Verify ROS topics (unless dry run)
        if not self.args.dry_run and not self.verify_ros_topics():
            print("Aborting due to missing ROS topics.")
            return

        # Start from home position
        print("\n" + "-" * 70)
        print("STARTING EVALUATION")
        print("-" * 70)
        print(f"Total trials: {len(self.poses)}")
        print("Press Enter between trials to replace the bottle.")

        if not self.return_to_home():
            print("ERROR: Failed to reach home position. Aborting.")
            return

        # Open gripper initially
        self.open_gripper()

        # Run trials
        for i, pose in enumerate(self.poses):
            if self.interrupted:
                break

            pose_index = self.args.start_index + i
            trial_num = i + 1
            total_trials = len(self.poses)

            trial_result = self.run_single_trial(pose, pose_index, trial_num, total_trials)
            self.results["trials"].append(trial_result)

            # Print trial summary
            print("\n" + "-" * 40)
            print(f"TRIAL {trial_num} RESULT: {trial_result['status'].upper()}")
            if trial_result["converged"]:
                print(f"  Converged: YES ({trial_result['iterations']} iterations)")
            else:
                print(f"  Converged: NO")
            if trial_result["error_message"]:
                print(f"  Error: {trial_result['error_message']}")
            print("-" * 40)

            # Wait for user before next trial (except after last trial)
            if i < len(self.poses) - 1:
                print("\n>>> Replace the bottle and press ENTER to continue (q to quit)... ", end="")
                user_input = input().strip().lower()
                if user_input == 'q':
                    print("User requested quit.")
                    break

        # Save results
        self._save_results()

        # Final summary
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        summary = self.results["summary"]
        print(f"  Total trials:   {self.results['total_trials']}")
        print(f"  Completed:      {self.results['completed_trials']}")
        print(f"  Skipped:        {summary.get('total_skipped', 0)}")
        print(f"  Aborted:        {summary.get('total_aborted', 0)}")
        print(f"  Converged:      {summary.get('total_converged', 0)}")
        print(f"  Failed:         {summary.get('total_failed', 0)}")
        print(f"  Convergence rate: {summary.get('convergence_rate', 0):.1f}%")
        print(f"  Mean iterations (converged): {summary.get('mean_iterations', 0):.1f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Real Robot Visual Servoing Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites (start manually before running this script):
  Terminal 1 - UR5 Driver:
    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=<ROBOT_IP> launch_rviz:=false

  Terminal 2 - MoveIt + Servo:
    ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_servo:=true

  Terminal 3 - Camera:
    ros2 launch realsense2_camera rs_launch.py enable_depth:=true enable_color:=true \\
      rgb_camera.color_profile:=1920x1080x6 depth_module.depth_profile:=1280x720x15 align_depth.enable:=true

  Terminal 4 - Cropper:
    cd catkin_ws/ibvs/src/robot && python3 crop_image_node.py

  Terminal 5 - Evaluation:
    cd catkin_ws/ibvs/src/robot && python3 run_real_robot_evaluation.py

Examples:
  # Full evaluation (all 25 poses)
  python3 run_real_robot_evaluation.py

  # Resume from pose 10
  python3 run_real_robot_evaluation.py --start-index 10

  # Run only poses 5-10
  python3 run_real_robot_evaluation.py --start-index 5 --end-index 11

  # Test robot motion only (no VS)
  python3 run_real_robot_evaluation.py --dry-run

  # Use different VS config
  python3 run_real_robot_evaluation.py --config ../visual_servoing/configs/real_robot/config_real_robot_sift.yaml
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=DEFAULT_VS_CONFIG,
        help=f'Path to VS config file (default: {DEFAULT_VS_CONFIG})'
    )

    parser.add_argument(
        '--poses', '-p',
        type=str,
        default=DEFAULT_POSES_FILE,
        help=f'Path to poses JSON file (default: {DEFAULT_POSES_FILE})'
    )

    parser.add_argument(
        '--joints', '-j',
        type=str,
        default=DEFAULT_JOINTS_FILE,
        help=f'Path to home joints JSON file (default: {DEFAULT_JOINTS_FILE})'
    )

    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start from this pose index (default: 0)'
    )

    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='End at this pose index (exclusive, default: all poses)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test without actual robot motion or VS'
    )

    parser.add_argument(
        '--skip-rotation',
        action='store_true',
        help='Skip rotation alignment phase in VS'
    )

    parser.add_argument(
        '--no-rosbag',
        action='store_true',
        help='Disable rosbag recording'
    )

    args = parser.parse_args()

    # Run evaluation
    runner = RealRobotEvaluationRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
