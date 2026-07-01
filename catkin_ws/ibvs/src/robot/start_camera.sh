#!/bin/bash
# Start RealSense camera, static TF, and image cropping node for ROS2
#
# Launches three components:
#   1. Static TF: tool0 -> camera_link (connects robot and camera TF trees)
#   2. RealSense camera node
#   3. Image cropper (1920x1080 -> 1440x1080)
#
# Usage:
#   conda activate ros2_vs
#   ./start_camera.sh          # Default 6 FPS
#   ./start_camera.sh --fast   # Fast mode 15 FPS
#   ./start_camera.sh -f       # Fast mode 15 FPS

# Kill stale camera processes from previous runs
echo "Cleaning up stale camera processes..."
pkill -f "realsense2_camera" 2>/dev/null
pkill -f "crop_image_node" 2>/dev/null
sleep 1

# Parse arguments
FPS=6
if [[ "$1" == "--fast" ]] || [[ "$1" == "-f" ]]; then
    FPS=15
fi

echo "=========================================="
echo "Logging into Hugging Face"
echo "=========================================="

# Login to Hugging Face (non-interactive, needed for gated models like DINOv3).
# Provide your own token via the HF_TOKEN environment variable, or log in once with
# `huggingface-cli login` (then this step is skipped).
if [ -n "$HF_TOKEN" ]; then
    python3 -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)" 2>/dev/null \
        && echo "  HuggingFace login OK" || echo "  HuggingFace login failed (models may still work from cache)"
else
    echo "  HF_TOKEN not set — skipping login (gated models like DINOv3 need it unless already cached)"
fi

echo ""
echo "=========================================="
echo "Starting RealSense Camera (${FPS} FPS)"
echo "  Color: 1920x1080"
echo "  Depth: 1280x720"
echo "=========================================="

# Start camera in background (ROS2 launch)
ros2 launch realsense2_camera rs_launch.py \
    enable_depth:=true \
    enable_color:=true \
    align_depth.enable:=true \
    rgb_camera.color_profile:=1920x1080x${FPS} \
    depth_module.depth_profile:=1280x720x${FPS} &
CAMERA_PID=$!

# Wait for camera to initialize
echo "Waiting for camera to initialize..."
sleep 5

# Check if camera is publishing
if ros2 topic list | grep -q "/camera/color/image_raw"; then
    echo "Camera started successfully!"
else
    echo "WARNING: Camera topics not detected yet, continuing anyway..."
fi

echo ""
echo "=========================================="
echo "Starting Image Cropping Node"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start the cropping node (runs in foreground)
python3 "${SCRIPT_DIR}/crop_image_node.py"

# When cropping node exits (Ctrl+C), also kill camera
echo "Shutting down..."
kill $CAMERA_PID 2>/dev/null
