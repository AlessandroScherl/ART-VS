#!/usr/bin/env python3

import os
import sys
import subprocess
import time

# Fix imports by adding parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import rospy
import numpy as np
import torch
import yaml
import argparse
from pathlib import Path

# Import from our modules
from core.config import Config
from ros_interface.ros_controller import ROSVisualServoingController
from ros_interface.gazebo_utils import manage_gazebo_models
from gazebo_msgs.srv import DeleteModel, GetWorldProperties
from vs_utils.geometry import (
    sample_camera_positions, sample_focal_points_original,
    calculate_position_error, calculate_orientation_error,
    calculate_look_at_orientation, apply_z_axis_rotation
)
try:
    from experiments.best_pose_finder_unified import find_and_set_best_pose
except ImportError:
    # Fallback to original if unified version not available
    from experiments.best_pose_finder import find_and_set_best_pose

# Import feature extractors from features module
from features.dinov2_extractor import ViTExtractor
from features.multi_backbone_extractor import MultiBackboneViTExtractor

# Import centralized model configuration
import sys
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from models_config import MODELS, YOLO_KEYWORDS, GOAL_IMAGES, VALID_MODELS

# Note: Old model mappings replaced by centralized config
# Models 4, 5, 7, 11, 12 have been removed due to issues

# Object-specific mask mapping (optional) - removed problematic models
MASK_IMAGES = {
    1: "mask_mug.jpg",
    2: "mask_coffeemaker.jpg",
    3: "mask_pan.jpg",
    # 4: "mask_knife.jpg",  # REMOVED
    # 5: "mask_flashlight.jpg",  # REMOVED
    6: "mask_hammer.jpg",
    7: "mask_scissors.jpg",
    8: "mask_screwdriver.jpg",
    9: "mask_keyboard.jpg",
    10: "mask_laptop.jpg",
    # 11: "mask_nitendo_DS.jpg",  # REMOVED
    # 12: "mask_tablet.jpg",  # REMOVED
    13: "mask_shoe_boat.jpg",
    14: "mask_shoe_boot.jpg",
    15: "mask_shoe_sandal.jpg",
    16: "mask_shoe_sport.jpg",
    17: "mask_toy_schoolbus.jpg",
    18: "mask_toy_squirrel.jpg",
    19: "mask_toy_transformer.jpg",
    20: "mask_toy_turtle.jpg",
    # No mask for hollywood - it fills the whole image
    99: None
}


def spawn_hollywood_model():
    """Spawn the hollywood model using gazebo spawn model service"""
    try:
        rospy.loginfo("Spawning Hollywood model...")
        
        # Use gazebo_ros spawn_model service
        from gazebo_msgs.srv import SpawnModel
        from geometry_msgs.msg import Pose, Point, Quaternion
        import tf.transformations as tf_trans
        
        # Wait for the service to be available
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=10.0)
        spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        # Read the model SDF file
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),
            "models", "viso", "model.sdf"
        )
        
        with open(model_path, 'r') as f:
            model_xml = f.read()
        
        # Convert RPY to quaternion
        # Roll = 1.5708 (90 degrees), Pitch = 0, Yaw = 1.5708 (90 degrees)
        quaternion = tf_trans.quaternion_from_euler(1.5708, 0, 1.5708)
        
        # Set pose
        pose = Pose()
        pose.position = Point(x=0.0, y=0.0, z=0.005)
        pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], 
                                    z=quaternion[2], w=quaternion[3])
        
        # Spawn the model
        resp = spawn_model_srv(
            model_name="resized",
            model_xml=model_xml,
            robot_namespace="",
            initial_pose=pose,
            reference_frame="world"
        )
        
        if resp.success:
            rospy.loginfo("Hollywood model spawned successfully")
            time.sleep(2.0)  # Give Gazebo time to settle
            return True
        else:
            rospy.logerr(f"Failed to spawn Hollywood model: {resp.status_message}")
            return False
            
    except Exception as e:
        rospy.logerr(f"Error spawning Hollywood model: {e}")
        return False


def spawn_model_via_script(model_index):
    """Spawn a model using direct service calls"""
    if model_index == 99:
        # Special case for hollywood
        return spawn_hollywood_model()
    
    # Regular models 1-20
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, script_dir)
    
    try:
        from spawn_model_service import spawn_model as spawn_model_func
        
        rospy.loginfo(f"Spawning model {model_index}: {MODELS[model_index]}")
        
        # Call the spawn function directly
        success = spawn_model_func(model_index)
        
        if success:
            rospy.loginfo(f"Successfully spawned model {model_index}")
            # Give Gazebo time to settle
            time.sleep(2.0)
            return True
        else:
            rospy.logerr(f"Failed to spawn model {model_index}")
            return False
            
    except Exception as e:
        rospy.logerr(f"Error spawning model {model_index}: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        return False


def spawn_model_at_pose(model_name, model_folder, position, orientation):
    """Spawn a model at a specific position and orientation"""
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose
    import re

    try:
        # Wait for spawn service
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Get correct path to models directory
        script_file = os.path.abspath(__file__)
        experiments_dir = os.path.dirname(script_file)
        visual_servoing_dir = os.path.dirname(experiments_dir)
        src_dir = os.path.dirname(visual_servoing_dir)
        ibvs_dir = os.path.dirname(src_dir)
        models_dir = os.path.join(ibvs_dir, 'models')
        model_path = os.path.join(models_dir, model_folder, 'model.sdf')

        if not os.path.exists(model_path):
            rospy.logerr(f"Model SDF not found at: {model_path}")
            return False

        # Read and modify SDF
        with open(model_path, 'r') as f:
            model_sdf = f.read()

        # Fix model:// URIs to ensure consistency
        model_sdf = re.sub(
            r'model://[^/]+/',
            f'model://{model_folder}/',
            model_sdf
        )

        # Make sure model is static (for distractors to stay in place)
        if '<static>' not in model_sdf:
            model_sdf = re.sub(
                r'(<model[^>]*>)',
                r'\1\n    <static>true</static>',
                model_sdf
            )
        else:
            model_sdf = re.sub(
                r'<static>[^<]*</static>',
                '<static>true</static>',
                model_sdf
            )

        # Create pose with provided position and orientation
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]

        # Spawn model directly at desired pose
        resp = spawn_sdf(
            model_name=model_name,
            model_xml=model_sdf,
            robot_namespace="",
            initial_pose=pose,
            reference_frame="world"
        )

        if resp.success:
            rospy.loginfo(f"✓ Spawned {model_name} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            return True
        else:
            rospy.logerr(f"✗ Failed to spawn {model_name}: {resp.status_message}")
            return False

    except Exception as e:
        rospy.logerr(f"Error spawning {model_name} at pose: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        return False


def spawn_cluttered_scene(target_model_id):
    """Spawn target model + 4 distractors for cluttered test2 evaluation"""
    # Import cluttered scene configurations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, 'paper_evaluation'))
    try:
        from test2_cluttered_scenes import CLUTTERED_SCENES
    except ImportError:
        rospy.logerr("Cluttered/TEST2 simulation mode is not included in this release "
                     "(test2_cluttered_scenes.py and its 3D object models were not published).")
        return False

    if target_model_id not in CLUTTERED_SCENES:
        rospy.logerr(f"No cluttered scene configuration for model {target_model_id}")
        return False

    # First spawn the target model at its default position (standard test2 behavior)
    rospy.loginfo(f"[CLUTTERED] Spawning target model {target_model_id}")
    if not spawn_model_via_script(target_model_id):
        rospy.logerr(f"Failed to spawn target model {target_model_id}")
        return False

    time.sleep(1.0)  # Let target model settle

    # Now spawn the 4 distractors at their recorded positions
    scene = CLUTTERED_SCENES[target_model_id]
    rospy.loginfo(f"[CLUTTERED] Spawning 4 distractors for {MODELS[target_model_id]}")

    for distractor_name, distractor_data in scene['distractors'].items():
        model_folder = distractor_data['model_name']
        position = distractor_data['position']
        orientation = distractor_data['orientation']

        rospy.loginfo(f"  - Spawning {distractor_name} at {position[:2]}")
        success = spawn_model_at_pose(
            distractor_name,  # Gazebo model name
            model_folder,     # Folder name in models/
            position,
            orientation
        )

        if not success:
            rospy.logwarn(f"Failed to spawn distractor {distractor_name}")

        time.sleep(0.5)  # Small delay between spawns

    rospy.loginfo(f"[CLUTTERED] Scene ready: 1 target + 4 distractors")
    time.sleep(1.0)  # Let all models settle
    return True


def update_goal_image(config, model_index):
    """Update the goal image path in the config for the current model"""
    if model_index in GOAL_IMAGES:
        # Get the goal image name
        goal_image_name = GOAL_IMAGES[model_index]

        # Images moved to consolidated location in ibvs/images/goal/goalrgb_1440x1080/
        # Go up from visual_servoing -> src -> ibvs
        ibvs_dir = os.path.dirname(os.path.dirname(parent_dir))
        images_dir = os.path.join(ibvs_dir, "images", "goal", "goalrgb_1440x1080")

        # Special handling for Hollywood (model 99)
        if model_index == 99:
            # Hollywood can use either goalrgb.jpg or goalrgb_hollywood.jpg
            hollywood_path = os.path.join(images_dir, "goalrgb_hollywood.jpg")
            default_path = os.path.join(images_dir, "goalrgb.jpg")

            if os.path.exists(hollywood_path):
                goal_image_path = hollywood_path
            elif os.path.exists(default_path):
                goal_image_path = default_path
            else:
                rospy.logerr("Hollywood goal image not found")
                return False
        else:
            # Regular models use their specific goal images
            goal_image_path = os.path.join(images_dir, goal_image_name)

        if os.path.exists(goal_image_path):
            config.image_path = goal_image_path
            rospy.loginfo(f"Updated goal image to: {goal_image_name}")
            rospy.loginfo(f"Goal image path: {goal_image_path}")

            # Also update mask path if object-specific mask exists
            update_mask_image(config, model_index)

            return True
        else:
            # Try default goal image as fallback
            default_goal = os.path.join(images_dir, "goalrgb.jpg")
            if os.path.exists(default_goal):
                config.image_path = default_goal
                rospy.logwarn(f"Goal image {goal_image_name} not found, using default: goalrgb.jpg")
                return True
            else:
                rospy.logerr(f"Goal image not found: {goal_image_path}")
                return False
    else:
        rospy.logwarn(f"No goal image mapping for model {model_index}")
        return False


def update_mask_image(config, model_index):
    """Update the mask image path in the config for the current model"""
    if model_index == 99:
        # Hollywood doesn't use a mask
        config.mask_path = None
        rospy.loginfo("[MASK] Hollywood experiment - no mask used")
        return

    if model_index in MASK_IMAGES and MASK_IMAGES[model_index]:
        mask_image_name = MASK_IMAGES[model_index]

        # Masks moved to consolidated location in ibvs/images/masks/
        ibvs_dir = os.path.dirname(os.path.dirname(parent_dir))
        masks_dir = os.path.join(ibvs_dir, "images", "masks")
        mask_path = os.path.join(masks_dir, mask_image_name)

        if os.path.exists(mask_path):
            config.mask_path = mask_path
            rospy.loginfo(f"Updated mask path to: {mask_path}")
            return True
        else:
            # If object-specific mask not found, disable masking
            config.mask_path = None
            rospy.logwarn(f"Object-specific mask not found for model {model_index}: {mask_image_name}")


def restart_simulation_with_world(world_file):
    """Restart the simulation with a different world file"""
    rospy.loginfo(f"Restarting simulation with world: {world_file}")
    
    # Kill existing Gazebo instance
    subprocess.run(["pkill", "-9", "gzserver"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "gzclient"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    # Launch with new world
    launch_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),
        "launch", "ibvs.launch"
    )
    
    # Start new launch process with world argument
    launch_cmd = [
        "roslaunch", "ibvs", "ibvs.launch",
        f"world_name:={world_file}",
        "gui:=false"
    ]
    
    launch_process = subprocess.Popen(launch_cmd)
    
    # Wait for Gazebo to be ready
    rospy.loginfo("Waiting for Gazebo to start...")
    time.sleep(10)
    
    return launch_process


def clear_all_gazebo_models():
    """Clear all models from Gazebo except the camera."""
    try:
        # Get all models in the world
        rospy.wait_for_service('/gazebo/get_world_properties', timeout=5.0)
        get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        
        world_props = get_world_props()
        model_names = world_props.model_names
        
        # Delete each model except the camera
        rospy.wait_for_service('/gazebo/delete_model', timeout=5.0)
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        for model_name in model_names:
            # Keep camera, realsense, table, and ground plane
            if ('camera' not in model_name.lower() and 
                'realsense' not in model_name.lower() and
                'table' not in model_name.lower() and
                'ground_plane' not in model_name.lower()):
                try:
                    delete_model(model_name)
                    rospy.loginfo(f"Deleted model: {model_name}")
                    time.sleep(0.5)
                except Exception as e:
                    rospy.logwarn(f"Failed to delete model {model_name}: {e}")
    except Exception as e:
        rospy.logerr(f"Error clearing models: {e}")


def calculate_separate_fps(iteration_times, tiling_switch_iteration):
    """Calculate separate FPS for pre-tiling and post-tiling phases.

    Args:
        iteration_times: Array of iteration times in seconds
        tiling_switch_iteration: Iteration when tiling was activated (None or -1 if no tiling)

    Returns:
        (pre_tiling_fps, post_tiling_fps) - Returns (overall_fps, None) if no tiling
    """
    if len(iteration_times) == 0:
        return 0.0, None

    # If no tiling switch, return overall FPS
    if tiling_switch_iteration is None or tiling_switch_iteration < 0:
        mean_time = np.mean(iteration_times)
        overall_fps = 1.0 / mean_time if mean_time > 0 else 0.0
        return overall_fps, None

    # Calculate pre-tiling FPS (up to but not including switch iteration)
    if tiling_switch_iteration > 0 and tiling_switch_iteration <= len(iteration_times):
        pre_tiling_times = iteration_times[:tiling_switch_iteration]
        if len(pre_tiling_times) > 0:
            pre_mean_time = np.mean(pre_tiling_times)
            pre_tiling_fps = 1.0 / pre_mean_time if pre_mean_time > 0 else 0.0
        else:
            pre_tiling_fps = 0.0

        # Calculate post-tiling FPS (from switch iteration onwards)
        post_tiling_times = iteration_times[tiling_switch_iteration:]
        if len(post_tiling_times) > 0:
            post_mean_time = np.mean(post_tiling_times)
            post_tiling_fps = 1.0 / post_mean_time if post_mean_time > 0 else 0.0
        else:
            post_tiling_fps = 0.0
    else:
        # If switch iteration is out of bounds, just calculate overall FPS
        mean_time = np.mean(iteration_times)
        overall_fps = 1.0 / mean_time if mean_time > 0 else 0.0
        return overall_fps, None

    return pre_tiling_fps, post_tiling_fps


def run_experiments_for_model(config, feature_extractor, model_index, samples_per_model=1, perturbation=False, use_cluttered=False):
    """Run experiments for a single model with multiple samples.

    Args:
        config: Configuration object
        feature_extractor: Feature extractor to use
        model_index: Model index to test
        samples_per_model: Number of samples to run
        perturbation: If True, swap models for each sample (Hollywood only)
        use_cluttered: If True, spawn target + 4 distractors in cluttered scene (TEST2 models only)
    """
    
    # Check if we need to switch worlds for hollywood
    current_world = "simulation_hollywood.world"  # Default (poster benchmark)
    if model_index == 99:
        # Hollywood needs simulation_hollywood.world (no table, enhanced lighting)
        current_world = "simulation_hollywood.world"
        # Note: World switching would need to be handled externally or via launch file
        rospy.logwarn("Hollywood experiment requires simulation_hollywood.world - make sure to launch with correct world file")
    
    # Clear any existing models
    clear_all_gazebo_models()
    
    # Update goal image for this model
    if not update_goal_image(config, model_index):
        rospy.logerr(f"Failed to update goal image for model {model_index}")
        return None
    
    # Spawn the model (or cluttered scene)
    if use_cluttered:
        rospy.loginfo(f"[CLUTTERED MODE] Spawning cluttered scene for model {model_index}")
        if not spawn_cluttered_scene(model_index):
            rospy.logerr(f"Failed to spawn cluttered scene for model {model_index}")
            return None
    else:
        if not spawn_model_via_script(model_index):
            rospy.logerr(f"Failed to spawn model {model_index}")
            return None
    
    rospy.loginfo(f"Starting experiments for Model {model_index}: {MODELS[model_index]}")
    rospy.loginfo(f"Goal image: {os.path.basename(config.image_path)}")
    
    # Get parameters from config
    num_circles = config.num_circles
    circle_radius_aug = config.circle_radius_aug
    # Use samples_per_model instead of config.num_samples to generate enough samples
    samples_per_circle = samples_per_model // num_circles if samples_per_model >= num_circles else 1
    num_samples = samples_per_model  # Generate as many samples as requested
    
    # UNIFORM SAMPLING BOX: Same dimensions for all models (1.2×1.2×0.3m)
    # Only camera heights and focal point radii differ between Hollywood and 3D models
    box_sample_size = np.array([1.2, 1.2, 0.3])  # Uniform box for all models
    
    if model_index == 99:
        # Hollywood poster - large 2D image at ground level
        circle_radius_aug = 0.08  # Larger radius for focal points
        desired_position = np.array([0, 0, 0.61])  # Camera 0.61m above ground
        reference_point = np.array([0.0, 0.0, 0.01])  # Object at ground level
        # Camera looking straight ahead at vertical poster
        # The poster was rotated with Roll=90° and Yaw=90°
        # For camera to look at it properly, we need to look forward horizontally
        # Converting RPY to quaternion: Roll=0, Pitch=90° (looking horizontal), Yaw=0
        import tf.transformations as tf_trans
        quat = tf_trans.quaternion_from_euler(0, 1.5708, 0)  # Pitch 90° to look horizontal
        desired_orientation = np.array([quat[0], quat[1], quat[2], quat[3]])
    else:
        # Regular 3D models on table - same box size but different heights
        circle_radius_aug = 0.04  # Smaller radius for tighter focal points
        desired_position = np.array([0, 0, 1.385])  # Camera 1.385m above ground
        reference_point = np.array([0.0, 0.0, 0.775])  # Object on table (table height 0.775m)
        # Use same orientation as vitvs_v3.py for consistent sampling statistics
        # This is pitch 90° (looking down at table)
        desired_orientation = np.array([0, 0.7071068, 0, 0.7071068])

        # Special case for keyboard: account for 94° spawn rotation
        # Keyboard object is spawned at Yaw=94.35°, which creates optical axis rotation
        # Use 90° target orientation to match discrete rotation mode (0°, 90°, 180°, 270°)
        if model_index == 9:  # Keyboard
            rospy.loginfo("[KEYBOARD] Using keyboard-specific target orientation (90° yaw compensation)")
            desired_orientation = np.array([-0.5, 0.5, 0.5, 0.5])  # Roll=0°, Pitch=90°, Yaw=90°

    # Set random seed for reproducibility
    np.random.seed(41)
    
    # Sample camera positions and orientations
    camera_positions = sample_camera_positions(box_sample_size, num_samples, desired_position)
    focal_points = sample_focal_points_original(num_samples, reference_point, num_circles, circle_radius_aug)
    look_at_matrices, look_at_quaternions = calculate_look_at_orientation(camera_positions, focal_points)
    orientations = apply_z_axis_rotation(look_at_matrices, num_circles, samples_per_circle)
    
    rospy.loginfo(f"Generated {len(camera_positions)} camera positions for model {model_index}")
    rospy.loginfo(f"  Unified sampling: box_size={box_sample_size}, circle_radius={circle_radius_aug}")
    rospy.loginfo(f"  Camera height: {desired_position[2]:.3f}m, Reference height: {reference_point[2]:.3f}m")
    
    # Print detailed camera sampling information
    rospy.loginfo("\nCAMERA SAMPLING DETAILS:")
    rospy.loginfo(f"  Sampling box: {box_sample_size[0]:.1f}m × {box_sample_size[1]:.1f}m × {box_sample_size[2]:.1f}m")
    rospy.loginfo(f"  Box centered at: [{desired_position[0]:.3f}, {desired_position[1]:.3f}, {desired_position[2]:.3f}]")
    rospy.loginfo(f"  Number of circles: {num_circles}")
    rospy.loginfo(f"  Samples per circle: {samples_per_circle}")
    rospy.loginfo(f"  Circle radius increment: {circle_radius_aug:.3f}m")
    
    # Calculate statistics for all sampled positions
    rospy.loginfo("\nCALCULATING INITIAL POSE STATISTICS...")
    all_distances = []
    all_angles = []
    
    # Import Rotation here once
    from scipy.spatial.transform import Rotation
    
    for i, pos in enumerate(camera_positions):
        dist_from_goal = np.linalg.norm(pos - desired_position)
        all_distances.append(dist_from_goal * 100)  # Convert to cm
        
        # Calculate orientation error for this sample (matching vitvs_v3.py)
        if i < len(orientations):
            R_current = Rotation.from_quat(orientations[i])
            R_desired = Rotation.from_quat(desired_orientation)
            # Calculate the relative rotation from current to desired (corrected order)
            R_error = R_current.inv() * R_desired
            angle_error = np.degrees(R_error.magnitude())
            all_angles.append(angle_error)
    
    # Only print detailed positions if we have 10 or fewer samples
    if len(camera_positions) <= 10:
        rospy.loginfo("\nSAMPLED CAMERA POSITIONS:")
        for i, pos in enumerate(camera_positions):
            circle_idx = i // samples_per_circle
            dist_from_goal = np.linalg.norm(pos - desired_position)
            rospy.loginfo(f"  Sample {i+1} (Circle {circle_idx+1}): "
                         f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                         f"- Distance from goal: {dist_from_goal*100:.1f}cm")
    
    # Display initial pose configuration summary
    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("INITIAL POSE CONFIGURATION SUMMARY")
    rospy.loginfo("="*60)
    if all_distances:
        mean_dist = np.mean(all_distances)
        std_dist = np.std(all_distances)
        rospy.loginfo(f"Initial Position Error: {mean_dist:.2f} ± {std_dist:.2f} cm")
        rospy.loginfo(f"  Range: [{np.min(all_distances):.2f}, {np.max(all_distances):.2f}] cm")
    if all_angles:
        mean_angle = np.mean(all_angles)
        std_angle = np.std(all_angles)
        rospy.loginfo(f"Initial Orientation Error: {mean_angle:.2f} ± {std_angle:.2f}°")
        rospy.loginfo(f"  Range: [{np.min(all_angles):.2f}, {np.max(all_angles):.2f}]°")
    rospy.loginfo("="*60)
    
    # Initialize controller object with updated config
    ros_controller = ROSVisualServoingController(config, feature_extractor, desired_position, desired_orientation)
    
    # Set YOLO keyword if ROI detection is enabled
    if config.use_roi_detection and model_index in YOLO_KEYWORDS:
        keyword = YOLO_KEYWORDS[model_index]
        ros_controller.set_yolo_keyword(keyword)
        rospy.loginfo(f"Set YOLO detection keyword: '{keyword}' for model {model_index} ({MODELS[model_index]})")
        
        # Check if goal detection succeeded
        if ros_controller.goal_roi_bbox is None:
            rospy.logwarn(f"Goal ROI detection failed for '{keyword}', MODE 4 will fall back to MODE 3")
            if hasattr(config, 'use_roi_tiling') and config.use_roi_tiling:
                rospy.logwarn("ROI tiling is enabled but no goal ROI detected - tiling will not work properly")
    
    # Initialize result storage for this model
    results = {
        'model_index': model_index,
        'model_name': MODELS[model_index],
        'goal_image': os.path.basename(config.image_path),
        'initial_positions': [],  # Store initial camera positions
        'initial_orientations': [],  # Store initial camera orientations
        'final_positions': [],
        'final_quaternions': [],
        'convergence_flags': [],
        'position_errors': [],
        'orientation_errors': [],
        'best_poses': [],
        'all_position_histories': [],
        'all_orientation_histories': [],
        'all_iteration_histories': [],
        'lowest_position_errors': [],
        'lowest_orientation_errors': [],
        'all_average_velocities': [],
        'all_velocity_mean_100': [],
        'all_velocity_mean_10': [],
        'all_applied_velocity_x': [],
        'all_applied_velocity_y': [],
        'all_applied_velocity_z': [],
        'all_applied_velocity_roll': [],
        'all_applied_velocity_pitch': [],
        'all_applied_velocity_yaw': [],
        'tiling_switch_iterations': [],
        'all_iteration_times': [],  # Store processing time per iteration for FPS calculation
        'pre_tiling_fps': [],  # FPS before tiling activation
        'post_tiling_fps': []  # FPS after tiling activation (None if no tiling)
    }
    
    # Run experiments for each sample
    for i in range(min(samples_per_model, len(camera_positions))):
        print(f"\n{'='*80}")
        print(f"STARTING: Model {model_index} ({MODELS[model_index]}), Sample {i + 1}/{samples_per_model}")
        print(f"{'='*80}")
        rospy.loginfo(f"\nRunning model {model_index}, sample {i + 1}/{samples_per_model}")

        # Reset LightTrack tracking for new sample
        if hasattr(ros_controller.yolo_detector, 'reset_tracking'):
            ros_controller.yolo_detector.reset_tracking()
            ros_controller.tracking_started = False
        
        # Handle perturbation mode for Hollywood
        if perturbation and model_index == 99:
            rospy.loginfo(f"Perturbation mode: Switching to perturbed model {i + 1}")
            manage_gazebo_models(i + 1)
            rospy.sleep(1)  # Give time for model to spawn
        
        try:
            # CRITICAL: Calculate initial errors from ORIGINAL sampled pose before rotation compensation
            # This ensures fair evaluation - convergence should be measured from the true starting point
            from ros_interface.gazebo_utils import set_camera_pose
            
            # Set camera to original sampled pose temporarily
            set_camera_pose(camera_positions[i], orientations[i])
            rospy.sleep(0.5)  # Brief wait for camera to settle
            
            # Calculate true initial errors from sampled pose
            from scipy.spatial.transform import Rotation as R
            true_initial_pos_error = np.linalg.norm(camera_positions[i] - desired_position) * 100  # cm
            current_rot = R.from_quat(orientations[i])
            desired_rot = R.from_quat(desired_orientation)
            true_initial_rot_error = (current_rot.inv() * desired_rot).magnitude() * (180 / np.pi)  # degrees
            
            rospy.loginfo(f"TRUE initial errors (from sampled pose): Trans={true_initial_pos_error:.2f}cm, Rot={true_initial_rot_error:.1f}°")
            
            # Now find best pose with rotation compensation
            best_pose = find_and_set_best_pose(
                ros_controller,
                camera_positions[i],
                orientations[i]
            )
            
            if best_pose is None:
                rospy.logerr(f"Failed to find best pose for model {model_index}, sample {i + 1}")
                continue
            
            rospy.loginfo(f"Found best pose for model {model_index}, sample {i + 1}")

            # Start LightTrack tracking after rotation alignment is found
            if hasattr(ros_controller.yolo_detector, 'start_tracking'):
                # Get current image for initial detection
                current_image = ros_controller.latest_pil_image
                if current_image is None:
                    rospy.logwarn("No current image available for tracking initialization")
                else:
                    # Run ONE detection on current image to get initial bbox
                    keyword = ros_controller.yolo_keyword
                    rospy.loginfo(f"Running initial detection on current image with keyword: '{keyword}'")

                    # The detector is still in DETECT state, so this will use LangSAM
                    detection = ros_controller.yolo_detector.detect(
                        current_image,
                        keyword=keyword,
                        padding_ratio=ros_controller.config.roi_padding_ratio
                    )

                    if detection['bbox'] is not None:
                        ros_controller.current_roi_bbox = detection['padded_bbox']
                        rospy.loginfo(f"Initial detection bbox: {detection['bbox']}")
                        rospy.loginfo(f"Initializing LightTrack tracking...")

                        ros_controller.yolo_detector.start_tracking(
                            np.array(current_image),
                            detection['bbox']  # Use raw bbox, not padded
                        )
                        ros_controller.tracking_started = True
                        rospy.loginfo("LightTrack tracking initialized successfully!")
                    else:
                        rospy.logwarn(f"No detection in current image for '{keyword}', staying in detection mode")

            # Store the TRUE initial errors in the controller (not the post-rotation errors)
            ros_controller.vs_controller.initial_error_translation = true_initial_pos_error
            ros_controller.vs_controller.initial_error_rotation = true_initial_rot_error
            ros_controller.vs_controller.initial_errors_preset = True  # Flag to prevent overwriting
            
            # Store initial pose for this sample
            results['initial_positions'].append(camera_positions[i])
            results['initial_orientations'].append(orientations[i])
            
            # Verify controller state before running
            rospy.loginfo("Checking controller state before visual servoing...")
            if ros_controller.latest_image is None:
                rospy.logerr("Controller has no latest_image!")
                continue
            if ros_controller.goal_image is None:
                rospy.logerr("Controller has no goal_image!")
                continue
            
            # Run visual servoing
            rospy.loginfo(f"Starting visual servoing for model {model_index}, sample {i + 1}...")
            rospy.loginfo(f"About to call ros_controller.run()...")
            
            result_tuple = ros_controller.run()
            
            rospy.loginfo(f"ros_controller.run() returned: {result_tuple is not None}")
            
            if result_tuple is None:
                rospy.logerr("ros_controller.run() returned None!")
                continue
            
            rospy.loginfo(f"Controller returned {len(result_tuple)} values")
            
            # Handle exact number of values based on what controller returns
            try:
                if len(result_tuple) == 20:
                    # New format with iteration times for FPS calculation
                    (final_position, final_quaternion, converged,
                     current_position_error, current_orientation_error,
                     position_history, orientation_history, iteration_count,
                     lowest_position_error, lowest_orientation_error,
                     average_velocities, velocity_mean_100, velocity_mean_10,
                     applied_velocity_x, applied_velocity_y, applied_velocity_z,
                     applied_velocity_roll, applied_velocity_pitch, applied_velocity_yaw,
                     iteration_times) = result_tuple
                    
                    # These are already numpy arrays from the controller
                    rospy.logdebug(f"Velocity arrays have {len(applied_velocity_x)} elements")
                    
                elif len(result_tuple) == 19:
                    # Legacy format without iteration times
                    (final_position, final_quaternion, converged,
                     current_position_error, current_orientation_error,
                     position_history, orientation_history, iteration_count,
                     lowest_position_error, lowest_orientation_error,
                     average_velocities, velocity_mean_100, velocity_mean_10,
                     applied_velocity_x, applied_velocity_y, applied_velocity_z,
                     applied_velocity_roll, applied_velocity_pitch, applied_velocity_yaw) = result_tuple
                    
                    iteration_times = np.array([])  # Empty array for backward compatibility
                    
                elif len(result_tuple) == 16:
                    # Legacy format without rotation velocities
                    (final_position, final_quaternion, converged,
                     current_position_error, current_orientation_error,
                     position_history, orientation_history, iteration_count,
                     lowest_position_error, lowest_orientation_error,
                     average_velocities, velocity_mean_100, velocity_mean_10,
                     applied_velocity_x, applied_velocity_y, applied_velocity_z) = result_tuple
                    
                    # Create empty arrays for rotation velocities and iteration times
                    applied_velocity_roll = np.array([])
                    applied_velocity_pitch = np.array([])
                    applied_velocity_yaw = np.array([])
                    iteration_times = np.array([])
                    
                else:
                    rospy.logerr(f"Unexpected number of values from run(): {len(result_tuple)}")
                    rospy.logerr(f"Expected 16 or 19 values")
                    continue
            except ValueError as e:
                rospy.logerr(f"Error unpacking result tuple: {e}")
                rospy.logerr(f"Tuple length: {len(result_tuple)}")
                continue
            
            rospy.loginfo(f"Visual servoing completed after {iteration_count} iterations")
            
            # Debug: Check data types
            rospy.logdebug(f"Type of applied_velocity_x: {type(applied_velocity_x)}, length if list: {len(applied_velocity_x) if isinstance(applied_velocity_x, list) else 'N/A'}")
            rospy.logdebug(f"Type of position_history: {type(position_history)}, length: {len(position_history) if hasattr(position_history, '__len__') else 'N/A'}")
            
            # Use the errors already calculated by the controller
            position_error = current_position_error
            orientation_error = current_orientation_error
            
            # Store results
            results['final_positions'].append(final_position)
            results['final_quaternions'].append(final_quaternion)
            results['convergence_flags'].append(converged)
            results['position_errors'].append(position_error)
            results['orientation_errors'].append(orientation_error)
            results['best_poses'].append(best_pose)
            results['all_position_histories'].append(position_history)
            results['all_orientation_histories'].append(orientation_history)
            results['all_iteration_histories'].append(iteration_count)
            results['lowest_position_errors'].append(lowest_position_error)
            results['lowest_orientation_errors'].append(lowest_orientation_error)
            results['all_average_velocities'].append(average_velocities)
            results['all_velocity_mean_100'].append(velocity_mean_100)
            results['all_velocity_mean_10'].append(velocity_mean_10)
            results['all_applied_velocity_x'].append(applied_velocity_x)
            results['all_applied_velocity_y'].append(applied_velocity_y)
            results['all_applied_velocity_z'].append(applied_velocity_z)
            results['all_applied_velocity_roll'].append(applied_velocity_roll)
            results['all_applied_velocity_pitch'].append(applied_velocity_pitch)
            results['all_applied_velocity_yaw'].append(applied_velocity_yaw)
            results['all_iteration_times'].append(iteration_times)

            # Determine tiling switch iteration for FPS calculation
            tiling_switch = None

            # Check for regular tiling (MODE 2)
            if config.use_hybrid_mode and hasattr(ros_controller, 'tiling_switch_iteration') and ros_controller.tiling_switch_iteration is not None:
                tiling_switch = ros_controller.tiling_switch_iteration
                results['tiling_switch_iterations'].append(tiling_switch)
            # Check for ROI tiling (MODE 4)
            elif hasattr(config, 'use_roi_tiling') and config.use_roi_tiling and hasattr(ros_controller, 'roi_tiling_switch_iteration') and ros_controller.roi_tiling_switch_iteration is not None:
                tiling_switch = ros_controller.roi_tiling_switch_iteration
                results['tiling_switch_iterations'].append(tiling_switch)
            elif config.use_hybrid_mode or (hasattr(config, 'use_roi_tiling') and config.use_roi_tiling):
                results['tiling_switch_iterations'].append(-1)  # Tiling configured but not activated
            else:
                results['tiling_switch_iterations'].append(-1)  # No tiling configured

            # Calculate separate FPS for pre-tiling and post-tiling
            if len(iteration_times) > 0:
                pre_fps, post_fps = calculate_separate_fps(iteration_times, tiling_switch)
                results['pre_tiling_fps'].append(pre_fps)
                results['post_tiling_fps'].append(post_fps if post_fps is not None else 0.0)

                # Display FPS information
                if post_fps is not None and tiling_switch is not None and tiling_switch > 0:
                    rospy.loginfo(f"FPS Analysis: Pre-tiling={pre_fps:.1f} Hz, Post-tiling={post_fps:.1f} Hz (switched at iter {tiling_switch})")
                    if pre_fps > 0 and post_fps > 0:
                        slowdown = pre_fps / post_fps
                        rospy.loginfo(f"  Tiling slowdown factor: {slowdown:.2f}x")
                else:
                    rospy.loginfo(f"Average FPS: {pre_fps:.1f} Hz (no tiling activation)")
            else:
                results['pre_tiling_fps'].append(0.0)
                results['post_tiling_fps'].append(0.0)
            
            print(f"\n{'='*80}")
            print(f"COMPLETED: Model {model_index} ({MODELS[model_index]}), Sample {i + 1}/{samples_per_model}")
            print(f"Status: {'CONVERGED' if converged else 'NOT CONVERGED'}")
            print(f"Iterations: {iteration_count}")
            print(f"Final Error: Position={position_error:.2f}cm, Rotation={orientation_error:.1f}°")
            if len(iteration_times) > 0:
                if post_fps is not None and tiling_switch is not None and tiling_switch > 0:
                    print(f"Performance: Pre-tiling={pre_fps:.1f} FPS, Post-tiling={post_fps:.1f} FPS (switch@{tiling_switch})")
                else:
                    print(f"Performance: {pre_fps:.1f} FPS (no tiling)")
            print(f"{'='*80}")
            rospy.loginfo(f"Completed model {model_index}, sample {i + 1} (Converged: {converged})")
            
        except Exception as e:
            rospy.logerr(f"Error processing model {model_index}, sample {i + 1}: {str(e)}")
            import traceback
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Calculate statistics for this model
    rospy.loginfo(f"\nModel {model_index} experiment completed")
    rospy.loginfo(f"  Total samples attempted: {samples_per_model}")
    rospy.loginfo(f"  Successful runs: {len(results['convergence_flags'])}")

    if len(results['convergence_flags']) > 0:
        converged_count = sum(results['convergence_flags'])
        convergence_rate = converged_count / len(results['convergence_flags']) * 100

        rospy.loginfo(f"\nModel {model_index} Results:")
        rospy.loginfo(f"  Convergence rate: {convergence_rate:.1f}% ({converged_count}/{len(results['convergence_flags'])})")

        if converged_count > 0:
            converged_pos_errors = [e for e, c in zip(results['position_errors'], results['convergence_flags']) if c]
            converged_ori_errors = [e for e, c in zip(results['orientation_errors'], results['convergence_flags']) if c]

            avg_pos_error = np.mean(converged_pos_errors)
            avg_ori_error = np.mean(converged_ori_errors)

            rospy.loginfo(f"  Average converged position error: {avg_pos_error:.2f} cm")
            rospy.loginfo(f"  Average converged orientation error: {avg_ori_error:.2f} degrees")

        # Display FPS statistics
        if results['pre_tiling_fps']:
            valid_pre_fps = [fps for fps in results['pre_tiling_fps'] if fps > 0]
            if valid_pre_fps:
                mean_pre_fps = np.mean(valid_pre_fps)
                rospy.loginfo(f"\nFPS Statistics:")
                rospy.loginfo(f"  Pre-tiling/Overall: {mean_pre_fps:.1f} ± {np.std(valid_pre_fps):.1f} Hz")

                valid_post_fps = [fps for fps in results['post_tiling_fps'] if fps > 0]
                if valid_post_fps:
                    mean_post_fps = np.mean(valid_post_fps)
                    rospy.loginfo(f"  Post-tiling: {mean_post_fps:.1f} ± {np.std(valid_post_fps):.1f} Hz")
                    if mean_post_fps > 0:
                        slowdown = mean_pre_fps / mean_post_fps
                        rospy.loginfo(f"  Average slowdown factor: {slowdown:.2f}x")

                    # Count how many samples activated tiling
                    tiling_activated = sum(1 for s in results['tiling_switch_iterations'] if s > 0)
                    rospy.loginfo(f"  Tiling activated: {tiling_activated}/{len(results['tiling_switch_iterations'])} samples")
    else:
        rospy.logwarn(f"No successful runs for model {model_index}!")
    
    return results


def despawn_table():
    """Remove table model for Hollywood experiments."""
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=5.0)
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # Try to delete common table model names
        table_names = ['Cafe_table', 'cafe_table', 'table', 'Table']
        
        for table_name in table_names:
            try:
                result = delete_model(table_name)
                if result.success:
                    rospy.loginfo(f"Successfully despawned table: {table_name}")
                    return True
                else:
                    rospy.logdebug(f"Model {table_name} not found or already deleted")
            except rospy.ServiceException as e:
                rospy.logdebug(f"Failed to delete {table_name}: {e}")
        
        rospy.loginfo("No table model found to delete (this is normal for Hollywood setup)")
        return True
        
    except rospy.ROSException:
        rospy.logwarn("Timeout waiting for /gazebo/delete_model service")
        return False
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run visual servoing experiments on multiple models')
    parser.add_argument('--config', type=str, default='config/config_amradio.yaml',
                        help='Path to configuration file')
    parser.add_argument('--models', type=str, default='1-20',
                        help='Models to test: "1-20" for regular models, "99" for hollywood, "all" for everything')
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of samples per model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (optional)')
    parser.add_argument('--perturbation', action='store_true',
                        help='Enable perturbation mode for Hollywood poster')
    parser.add_argument('--cluttered', action='store_true',
                        help='Enable cluttered scene mode (spawn target + 4 distractors)')
    parser.add_argument('--iteration-display-freq', type=int, default=1,
                        help='Display iteration output every N iterations (default: 1)')
    
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('multi_model_experiment', anonymous=True)
    
    # Parse models to determine if we need to despawn table
    model_str = args.models if args.models else "1-20,99"
    
    # Simple check: only despawn table if we're ONLY running model 99
    if model_str == "99":
        rospy.loginfo("Hollywood-only experiment detected (model 99) - despawning table...")
        despawn_table()
    else:
        rospy.loginfo(f"Running models {model_str} - keeping table for standard models")
    
    # Load configuration
    # Try multiple possible paths
    possible_config_paths = [
        os.path.join(parent_dir, args.config),
        os.path.join(current_dir, args.config),
        args.config  # Absolute path
    ]
    
    config_path = None
    for path in possible_config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path is None:
        rospy.logerr(f"Configuration file not found. Tried:")
        for path in possible_config_paths:
            rospy.logerr(f"  {path}")
        return
    
    config = Config(config_path)
    
    # Override iteration display frequency if provided
    if args.iteration_display_freq > 1:
        config.iteration_display_freq = args.iteration_display_freq
        rospy.loginfo(f"Setting iteration display frequency to: every {args.iteration_display_freq} iterations")
    
    # Initialize feature extractor based on config
    if hasattr(config, 'backbone_model'):
        rospy.loginfo(f"Using MultiBackboneViTExtractor with {config.backbone_model}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_size = getattr(config, 'vit_input_size', 224)
        feature_extractor = MultiBackboneViTExtractor(config.backbone_model, device, input_size)
        rospy.loginfo(f"Set input size: {input_size}")
    else:
        rospy.loginfo("Using standard ViTExtractor (legacy fallback)")
        feature_extractor = ViTExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse models to test - only use valid models
    models_to_test = []
    if args.models == "1-20":
        # Get all valid models from 1-20
        models_to_test = [m for m in VALID_MODELS if m <= 20]
    elif args.models == "99":
        models_to_test = [99]
    elif args.models == "all":
        models_to_test = VALID_MODELS.copy()
    else:
        # Try to parse as single number, range, or comma-separated list
        try:
            for part in args.models.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    # Only include valid models in the range
                    for m in range(start, end + 1):
                        if m in VALID_MODELS:
                            models_to_test.append(m)
                else:
                    model_id = int(part)
                    if model_id in VALID_MODELS:
                        models_to_test.append(model_id)
                    else:
                        rospy.logwarn(f"Model {model_id} is not in valid models list, skipping")
        except:
            rospy.logerr(f"Invalid models specification: {args.models}")
            return
    
    rospy.loginfo(f"Will test models: {models_to_test}")
    rospy.loginfo(f"Number of models to test: {len(models_to_test)}")
    
    # Run experiments for each model
    all_results = []
    
    for model_index in models_to_test:
        if model_index not in MODELS:
            rospy.logwarn(f"Model {model_index} not found in MODELS mapping, skipping")
            continue
        
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"Testing Model {model_index}: {MODELS[model_index]}")
        rospy.loginfo(f"{'='*60}")
        
        rospy.loginfo(f"Calling run_experiments_for_model with model_index={model_index}, samples={args.samples}, perturbation={args.perturbation}, cluttered={args.cluttered}")
        results = run_experiments_for_model(config, feature_extractor, model_index, args.samples, args.perturbation, args.cluttered)
        
        rospy.loginfo(f"run_experiments_for_model returned: {results is not None}")
        if results:
            all_results.append(results)
            rospy.loginfo(f"Added results for model {model_index} to all_results (total: {len(all_results)})")
    
    # Save results (generate filename if not specified)
    if all_results:
        if args.output:
            # If output path is absolute, use it directly; otherwise join with parent_dir
            if os.path.isabs(args.output):
                output_path = args.output
            else:
                output_path = os.path.join(parent_dir, args.output)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                rospy.loginfo(f"Created output directory: {output_dir}")
        else:
            # Generate automatic filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            perturbation_str = "_perturbed" if args.perturbation else ""
            models_str = args.models.replace('-', '_').replace(',', '_')
            output_filename = f"results_{config_name}_models{models_str}{perturbation_str}_{timestamp}.npz"
            output_path = os.path.join(parent_dir, "benchmark", "results", output_filename)
            
            # Create benchmark results directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                rospy.loginfo(f"Created output directory: {output_dir}")
        
        # Convert results to numpy arrays for saving
        save_data = {}
        for i, result in enumerate(all_results):
            prefix = f"model_{result['model_index']}_"
            
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 0:
                    try:
                        # Try to convert to numpy array
                        arr = np.array(value)
                        save_data[prefix + key] = arr
                    except (ValueError, TypeError) as e:
                        # If conversion fails (inhomogeneous data), save as object array
                        rospy.logwarn(f"Converting {key} to object array due to inhomogeneous shape")
                        save_data[prefix + key] = np.array(value, dtype=object)
                else:
                    save_data[prefix + key] = value
        
        np.savez(output_path, **save_data)
        rospy.loginfo(f"Results saved to: {output_path}")
    
    # Print summary
    rospy.loginfo(f"\n{'='*60}")
    rospy.loginfo("EXPERIMENT SUMMARY")
    rospy.loginfo(f"{'='*60}")
    rospy.loginfo(f"Total results collected: {len(all_results)}")
    
    for result in all_results:
        model_idx = result['model_index']
        model_name = result['model_name']
        convergence_flags = result['convergence_flags']
        
        if len(convergence_flags) > 0:
            converged_count = sum(convergence_flags)
            convergence_rate = converged_count / len(convergence_flags) * 100
            
            rospy.loginfo(f"\nModel {model_idx} ({model_name}):")
            rospy.loginfo(f"  Samples: {len(convergence_flags)}")
            rospy.loginfo(f"  Convergence rate: {convergence_rate:.1f}%")
            
            if converged_count > 0:
                pos_errors = [e for e, c in zip(result['position_errors'], convergence_flags) if c]
                ori_errors = [e for e, c in zip(result['orientation_errors'], convergence_flags) if c]
                
                if pos_errors:
                    rospy.loginfo(f"  Avg position error: {np.mean(pos_errors):.2f} cm")
                if ori_errors:
                    rospy.loginfo(f"  Avg orientation error: {np.mean(ori_errors):.2f} degrees")
    
    rospy.loginfo("\nExperiment completed!")


if __name__ == '__main__':
    main()