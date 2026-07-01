#!/bin/bash
####################################################################
# Vision Container Startup Script - Desktop/RTX 5090 Version     #
# Uses CUDA 12.8 compatible PyTorch for RTX 5090                 #
####################################################################

# Setup for GPU access
xhost +local:docker

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker rm -f viso_vision_desktop 2>/dev/null || true

# Build Docker image with RTX 5090 compatibility
echo "Building Vision Container for RTX 5090 (CUDA 12.8)..."
docker build -f Dockerfile.vision_50XX .. -t viso_vision_desktop

if [ $? -ne 0 ]; then
    echo "Build failed! Please check the error messages above."
    exit 1
fi

# Run container with GPU support
echo "Starting Vision Container (Desktop/RTX 5090)..."
docker run -it --rm -d \
    --name viso_vision_desktop \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    --privileged \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --mount src="$(pwd)/../catkin_ws",target=/root/vision_ws/src/,type=bind \
    viso_vision_desktop

if [ $? -ne 0 ]; then
    echo "Failed to start container! Please check Docker and NVIDIA runtime."
    exit 1
fi

echo ""
echo "✅ Vision Container (RTX 5090 Compatible) is running!"
echo ""
echo "Container Details:"
echo "  - Name: viso_vision_desktop"
echo "  - PyTorch: CUDA 12.8 compatible"
echo "  - GPU: RTX 5090 optimized"
echo "  - Workspace: /root/vision_ws/src/"
echo ""
echo "To verify GPU compatibility:"
echo "  python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "Connecting to container in 3 seconds..."
sleep 3

# Connect to container
docker exec -it viso_vision_desktop bash
