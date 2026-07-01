# Running ART-VS on a Real Robot (ROS2 + MoveIt Servo)

This is the setup we used for the grasping experiments in the paper: a UR5 with a
wrist-mounted RealSense D435i, driven through **ROS2 Jazzy + MoveIt2 + MoveIt Servo** on a
single laptop. The vision pipeline computes a Cartesian camera velocity and publishes it as
`geometry_msgs/TwistStamped` to `/servo_node/delta_twist_cmds`; MoveIt Servo handles the
Jacobian, singularity avoidance and joint limits from there. Nothing about ART-VS itself is
UR-specific — any arm with a MoveIt2 config and MoveIt Servo support can consume the same
commands — but the launch files and helper scripts in [`catkin_ws/ibvs/src/robot/`](../catkin_ws/ibvs/src/robot/)
are written for the UR5 and are meant as a working reference, not an abstraction layer.

Note the split: the **simulation benchmark runs on ROS1 Noetic** (in Docker, see the main
README), while the **real robot runs on ROS2 Jazzy** (natively). The vision package supports
both — `ros_interface/ros_controller.py` is the ROS1/sim controller,
`ros_interface/ros2_controller.py` is the ROS2 port used here.

## Hardware we used

- Universal Robots **UR5 (CB3)**, PolyScope 3.15+
- **Intel RealSense D435i** mounted on the end-effector (eye-in-hand, `tool0`)
- **Robotiq 2F-85** gripper, driven over Modbus RTU on `/dev/ttyUSB0` (`robot/gripper.py`)
- Laptop with an NVIDIA GPU (we used an RTX 4070 Mobile), Ubuntu 24.04, same LAN as the robot

## Install (once)

```bash
# ROS2 Jazzy: https://docs.ros.org/en/jazzy/Installation.html
sudo apt install \
  ros-jazzy-ur-robot-driver \
  ros-jazzy-ur-moveit-config \
  ros-jazzy-moveit \
  ros-jazzy-moveit-servo \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-pymoveit2 \
  ros-jazzy-realsense2-camera

# Python side (any recent Python; we used a conda env with Python 3.12)
pip install torch torchvision transformers timm einops opencv-python scipy pyserial
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git   # LangSAM (ROI mode)
git clone https://github.com/researchmm/LightTrack.git ~/LightTrack          # tracker (ROI mode)
# + LightTrack snapshot weights -> ~/LightTrack/snapshot/LightTrackM/LightTrackM.pth
#   (or point LIGHTTRACK_WEIGHTS / the config's lighttrack_weights somewhere else)
# DINOv3 weights are gated on Hugging Face: request access once, then either run
# `huggingface-cli login` or export HF_TOKEN=<your token> before start_camera.sh
```

The UR ROS2 driver needs the **External Control URCap** on the robot once: copy
`/opt/ros/jazzy/share/ur_robot_driver/resources/externalcontrol-*.urcap` to a USB stick,
install it on the teach pendant (Settings → System → URCaps), create a program containing a
single External Control node with your **laptop's IP** and port 50002, and save it (e.g.
`ros_control.urp`). On CB3 the host IP is set inside the program node, not in the URCap
settings.

## Session startup

Each in its own terminal, in this order (source ROS2 in each):

```bash
# 1. UR driver — wait for "System successfully started!", THEN press Play on the pendant
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=<ROBOT_IP> launch_rviz:=false

# 2. MoveIt + Servo
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_servo:=true

# 3. Switch controllers + set Servo to twist mode (must be re-done every session)
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller --activate forward_position_controller
ros2 service call /servo_node/switch_command_type moveit_msgs/srv/ServoCommandType "{command_type: 1}"

# 4. Camera (RealSense color 1920x1080 + aligned depth)
catkin_ws/ibvs/src/robot/start_camera.sh

# 5. Image cropper (1920x1080 -> 1440x1080, fixes the principal point to match the goal images)
cd catkin_ws/ibvs/src/robot && python3 crop_image_node.py
```

Then run visual servoing to a captured goal image:

```bash
cd catkin_ws/ibvs/src/visual_servoing
python3 run_visual_servoing.py --config configs/real_robot/config_real_robot_dinov3.yaml
```

or reproduce the paper's grasping evaluation (servo → grasp → lift → log, over pre-sampled
initial poses):

```bash
cd catkin_ws/ibvs/src/robot
python3 sample_evaluation_poses.py                 # generate the initial-pose set
python3 run_real_robot_evaluation.py               # servo + grasp per pose (config_real_robot_dinov3.yaml)
```

`configs/real_robot/` has the two configs we used (DINOv3 and the SIFT baseline), the goal
image and the object mask. (The paper's shoe trials used this same pipeline, only with a shoe
goal image and prompt.) To servo to your own object: capture a goal image at the desired
pose, set `image_path` and the `detection_keyword` text prompt (ROI mode uses LangSAM to
segment the target once, then LightTrack to track it), and go.

## The velocity interface, in case you bring a different arm

- The controller publishes `geometry_msgs/TwistStamped` on `velocity_topic`
  (`/servo_node/delta_twist_cmds`), ~100 Hz, `header.frame_id` = `tool0` by default
  (`velocity_frame_id` in the config).
- **Timestamps must be current** — MoveIt Servo rejects zero/stale stamps, and its watchdog
  stops the arm if commands pause for >0.1 s. Both are handled by the controller; keep them
  in mind if you write your own consumer.
- Camera input: color + aligned depth `sensor_msgs/Image` on `camera_rgb_topic` /
  `camera_depth_topic`; TF must connect `base_frame → tool_frame → camera_frame`
  (robot state publisher + your hand-eye calibration).

## Debugging

```bash
ros2 topic echo /servo_node/status        # 0 = OK, 1 = halted (singularity/limit), 2 = invalid command
ros2 control list_controllers             # forward_position_controller must be active
ros2 topic echo /joint_states             # is the driver alive?
```

The three most common failure modes: forgot to press Play on the pendant after starting the
driver, forgot the `switch_command_type` service call (it does not persist across sessions),
or the trajectory controller is still active instead of `forward_position_controller`.

Safety: ART-VS outputs raw velocity commands. MoveIt Servo enforces joint limits and
singularity thresholds, but keep the e-stop in reach and start with a clear workspace and
conservative velocity limits.
