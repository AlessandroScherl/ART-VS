#!/bin/bash
####################################################################
# Vision Container Startup Script - Laptop/RTX 4070 Version      #
# Uses CUDA 11.8 compatible PyTorch for RTX 4070 Mobile          #
####################################################################

# Setup for GPU access
xhost +local:docker

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker rm -f viso_vision_laptop 2>/dev/null || true

# Build Docker image with RTX 4070 compatibility
echo "Building Vision Container for RTX 4070 Mobile (CUDA 11.8)..."
docker build -f Dockerfile.vision_40XX .. -t viso_vision_laptop

if [ $? -ne 0 ]; then
    echo "Build failed! Please check the error messages above."
    exit 1
fi

# Run container with GPU support
echo "Starting Vision Container (Laptop/RTX 4070)..."
docker run -it --rm -d \
    --name viso_vision_laptop \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    --privileged \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --mount src="$(pwd)/../catkin_ws",target=/root/vision_ws/src/,type=bind \
    viso_vision_laptop

if [ $? -ne 0 ]; then
    echo "Failed to start container! Please check Docker and NVIDIA runtime."
    exit 1
fi

echo ""
echo "✅ Vision Container (RTX 4070 Compatible) is running!"
echo ""
echo "Container Details:"
echo "  - Name: viso_vision_laptop"
echo "  - PyTorch: CUDA 11.8 compatible"
echo "  - GPU: RTX 4070 Mobile optimized"
echo "  - Workspace: /root/vision_ws/src/"
echo ""
echo "To verify GPU compatibility:"
echo "  python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "Connecting to container in 3 seconds..."
sleep 3

# Connect to container
docker exec -it viso_vision_laptop bash