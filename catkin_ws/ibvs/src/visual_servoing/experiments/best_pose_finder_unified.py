#!/usr/bin/env python3
"""
Unified best pose finder that uses virtual rotation for both discrete and continuous modes.
"""

import logging
import time
import torch
import os
import sys
from PIL import Image

# Fix Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ros_interface.gazebo_utils import set_camera_pose
from vs_utils.geometry import rotate_camera_x_axis
from features.rotation_finder_fixed import UnifiedRotationFinder

logger = logging.getLogger(__name__)


def find_and_set_best_pose_unified(ros_controller, camera_position, initial_quaternion):
    """
    Find the best camera pose using unified rotation finder.
    Works by capturing one image and testing virtual rotations.
    
    Args:
        ros_controller: ROSVisualServoingController instance
        camera_position: Camera position
        initial_quaternion: Initial orientation quaternion
    
    Returns:
        tuple: (best_position, best_quaternion)
    """
    logger.info("=== Unified Rotation Compensation ===")
    
    # First, set camera to initial pose to capture image
    set_camera_pose(camera_position, initial_quaternion)
    time.sleep(1)  # Wait for camera to settle
    
    # Update controller's pose
    ros_controller.camera_position = camera_position
    ros_controller.orientation_quaternion = initial_quaternion
    
    # Get current camera image
    current_image = ros_controller.get_current_image()
    if current_image is None:
        logger.error("Failed to capture current camera image")
        return camera_position, initial_quaternion
    
    # Get goal image
    goal_image = ros_controller.get_goal_image()
    if goal_image is None:
        logger.error("Failed to load goal image")
        return camera_position, initial_quaternion
    
    # Create rotation finder with ros_controller for visualization
    rotation_finder = UnifiedRotationFinder(
        feature_extractor=ros_controller.feature_extractor,
        config=ros_controller.config,
        ros_controller=ros_controller
    )
    
    # Check if rotation compensation is enabled
    enable_rotation_compensation = getattr(ros_controller.config, 'enable_rotation_compensation', True)
    logger.info(f"Rotation compensation enabled: {enable_rotation_compensation}")
    
    # Also check for classical methods
    is_classical = hasattr(ros_controller.config, 'backbone_model') and 'classical-' in ros_controller.config.backbone_model
    if is_classical:
        logger.info(f"Classical method detected ({ros_controller.config.backbone_model}) - forcing rotation compensation OFF")
        enable_rotation_compensation = False
    
    if not enable_rotation_compensation:
        logger.info("Rotation compensation disabled - skipping rotation search")
        best_quaternion = initial_quaternion
        best_angle = 0
    else:
        # Find best rotation
        rotation_mode = getattr(ros_controller.config, 'rotation_mode', 'discrete')
        logger.info(f"Using rotation mode: {rotation_mode}")
        
        result = rotation_finder.find_best_rotation(goal_image, current_image)
        best_angle = result['best_angle']
        
        # Apply the best rotation to the camera quaternion
        # Note: We negate the angle because image.rotate() rotates counterclockwise
        # but we need the camera to rotate in the opposite direction
        if abs(best_angle) > 0.1:  # Only rotate if angle is significant
            best_quaternion = rotate_camera_x_axis(initial_quaternion, -best_angle)
            logger.info(f"Applying rotation of {-best_angle:.1f}° to camera (opposite of image rotation)")
        else:
            best_quaternion = initial_quaternion
            logger.info("No significant rotation needed")
    
    # Set camera to best pose
    set_camera_pose(camera_position, best_quaternion)
    time.sleep(1)  # Wait for camera to settle
    
    # Update controller's pose
    ros_controller.camera_position = camera_position
    ros_controller.orientation_quaternion = best_quaternion
    
    logger.info(f"Rotation compensation complete: {best_angle:.1f}° applied")
    logger.info("=====================================")
    
    return camera_position, best_quaternion


def find_and_set_best_pose(ros_controller, camera_position, initial_quaternion):
    """
    Wrapper function that maintains backward compatibility.
    Redirects to unified implementation.
    """
    return find_and_set_best_pose_unified(ros_controller, camera_position, initial_quaternion)