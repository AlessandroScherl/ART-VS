#!/bin/bash
####################################################################
# Visual Servoing Docker Container Startup Script                     #
#                                                                    #
# This code is available under the MIT license and comes without     #
# any explicit or implicit warranty.                                 #
#                                                                    #
# (C) Alessandro Scherl 2024 <alessandro.scherl@technikum-wien.at>  #
####################################################################

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.simulation .. -t viso_sim

# Detect if we're running over SSH
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
    echo "SSH session detected. Using virtual display mode..."
    USE_VIRTUAL_DISPLAY=1
else
    echo "Local session detected. Using X11 forwarding..."
    USE_VIRTUAL_DISPLAY=0
    # Setup for local X11
    xhost +local:docker 2>/dev/null || true
fi

# Setup for RealSense camera (if you have permissions)
udevadm control --reload-rules && udevadm trigger 2>/dev/null || true

# Function to run with virtual display (for SSH/headless)
run_virtual_display() {
    echo "Starting container with virtual display (Xvfb)..."
    docker run -it --rm -t -d \
        --name viso_sim \
        --network="host" \
        -e VIRTUAL_DISPLAY=1 \
        -e DISPLAY=:99 \
        -e QT_X11_NO_MITSHM=1 \
        --privileged \
        --gpus all \
        -p 8888:8888 \
        --mount src="$(pwd)/../catkin_ws",target=/root/catkin_ws/src/,type=bind \
        viso_sim
}

# Function to run with X11 forwarding (for local or SSH with X forwarding)
run_x11_forward() {
    echo "Starting container with X11 forwarding..."
    
    # If over SSH, ensure X11 forwarding is enabled
    if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
        if [ -z "$DISPLAY" ]; then
            echo "Warning: DISPLAY not set. Make sure SSH X11 forwarding is enabled (ssh -X)"
            echo "Falling back to virtual display mode..."
            run_virtual_display
            return
        fi
        # For SSH X11 forwarding
        XSOCK=/tmp/.X11-unix
        XAUTH=/tmp/.docker.xauth
        touch $XAUTH
        xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
        
        docker run -it --rm -t -d \
            --name viso_sim \
            --network="host" \
            -e DISPLAY=$DISPLAY \
            -e XAUTHORITY=$XAUTH \
            -e QT_X11_NO_MITSHM=1 \
            --gpus all \
            --volume=$XSOCK:$XSOCK:rw \
            --volume=$XAUTH:$XAUTH:rw \
            --privileged \
            -p 8888:8888 \
            --mount src="$(pwd)/../catkin_ws",target=/root/catkin_ws/src/,type=bind \
            viso_sim
    else
        # For local display with GPU
        docker run -it --rm -t -d \
            --name viso_sim \
            --network="host" \
            -e DISPLAY=$DISPLAY \
            -e QT_X11_NO_MITSHM=1 \
            --privileged \
            -p 8888:8888 \
            --gpus all \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --mount src="$(pwd)/../catkin_ws",target=/root/catkin_ws/src/,type=bind \
            viso_sim
    fi
}

# Function to run with GPU support
run_with_gpu() {
    if [ "$USE_VIRTUAL_DISPLAY" -eq 1 ]; then
        echo "Starting container with GPU and virtual display..."
        docker run -it --rm -t -d \
            --name viso_sim \
            --network="host" \
            -e VIRTUAL_DISPLAY=1 \
            -e DISPLAY=:99 \
            -e QT_X11_NO_MITSHM=1 \
            --privileged \
            --runtime=nvidia \
            --gpus all \
            -p 8888:8888 \
            --mount src="$(pwd)/../catkin_ws",target=/root/catkin_ws/src/,type=bind \
            viso_sim
    else
        echo "Starting container with GPU and X11 forwarding..."
        docker run -it --rm -t -d \
            --name viso_sim \
            --network="host" \
            -e DISPLAY=$DISPLAY \
            -e QT_X11_NO_MITSHM=1 \
            --privileged \
            --runtime=nvidia \
            --gpus all \
            -p 8888:8888 \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --mount src="$(pwd)/../catkin_ws",target=/root/catkin_ws/src/,type=bind \
            viso_sim
    fi
}

# Check command line arguments
if [ "$1" == "--virtual" ]; then
    run_virtual_display
elif [ "$1" == "--gpu" ]; then
    run_with_gpu
elif [ "$1" == "--x11" ]; then
    run_x11_forward
else
    # Auto-detect best mode
    if [ "$USE_VIRTUAL_DISPLAY" -eq 1 ]; then
        run_virtual_display
    else
        run_x11_forward
    fi
fi

# Wait for container to start
sleep 2

# Connect to container
echo "Connecting to container..."
docker exec -it viso_sim bash
