#!/bin/bash
# Helper script to start the simulation inside the Docker container

# Source ROS environment
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

# Check if virtual display is being used
if [ "$VIRTUAL_DISPLAY" == "1" ] || [ "$DISPLAY" == ":99" ]; then
    echo "Starting Xvfb virtual display..."
    # Kill any existing Xvfb process
    pkill Xvfb 2>/dev/null || true
    
    # Start Xvfb with proper settings for Gazebo
    Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
    sleep 2
    
    export DISPLAY=:99
    export LIBGL_ALWAYS_SOFTWARE=1
    export GAZEBO_IP=127.0.0.1
    export GAZEBO_GUI=false
    
    echo "Virtual display started on :99"
fi

# Set Gazebo environment variables for better performance
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_RESOURCE_PATH=/usr/share/gazebo-11
export GAZEBO_PLUGIN_PATH=/opt/ros/noetic/lib
export GAZEBO_MODEL_PATH=/opt/ros/noetic/share

# For headless mode, use software rendering
if [ "$VIRTUAL_DISPLAY" == "1" ]; then
    export SVGA_VGPU10=0
    export GAZEBO_GUI_INI_FILE=/dev/null
fi

echo "Environment setup complete. Display: $DISPLAY"
echo "Starting simulation..."

# Launch the simulation
roslaunch ibvs ibvs.launch
