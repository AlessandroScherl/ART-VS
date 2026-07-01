#!/usr/bin/env python3

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetWorldProperties
from geometry_msgs.msg import Pose
import tf.transformations as transformations

# Import centralized model configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from models_config import MODELS
except ImportError:
    # Fallback if import fails
    MODELS = {
        1: "1_Coffee_mug_1",
        2: "1_Coffeemaker",
        3: "1_Cooking_pan",
        4: "1_Toaster",  # NEW
        5: "2_Helmet",  # NEW
        6: "2_Hammer_black",
        7: "2_Scissors",  # NEW
        8: "2_Screwdriver",
        9: "3_Keyboard",
        10: "3_Laptop_1",
        11: "3_Alarmclock",  # NEW
        12: "3_Mouse",  # NEW
        13: "4_Shoe_boat",
        14: "4_Shoe_boot",
        15: "4_Shoe_sandal",
        16: "4_Shoe_sport",
        17: "5_Sonny_School_Bus",
        18: "5_Toy_squirrel",
        19: "5_Toy_transformer",
        20: "5_Toy_turtle",
        21: "4_Backpack"  # NEW
    }

# Z elevation to add to all models (in meters)
Z_ELEVATION = 0.775

# Specific poses for certain models (position and quaternion orientation)
MODEL_POSES = {
    "1_Coffeemaker": {
        "position": [-0.0023754185531288385, 0.16935168206691742, 0.03971210494637489],
        "orientation": [-0.13195763528347015, -0.6942475438117981, 0.6948477029800415, -0.13339509069919586]
    },
    "1_Toaster": {
        "position": [0.0, 0.0, 0.0],  # Default upright position
        "orientation": [0.0, 0.0, 0.0, 1.0]  # No rotation
    },
    "2_Helmet": {
        "position": [0.0, 0.0, 0.0],  # Default position
        "orientation": [0.0, 0.0, 0.0, 1.0]  # Upright
    },
    "2_Hammer_black": {
        "position": [-0.035142723470926285, -0.0026362850330770016, 0.03142233565449715],
        "orientation": [0.481036901473999, 0.5184742212295532, 0.5182544589042664, 0.4808329939842224]
    },
    "2_Scissors": {
        "position": [0.0, 0.0, 0.01],  # Slightly elevated
        "orientation": [0.0, 0.0, 0.0, 1.0]  # Flat on table
    },
    "3_Keyboard": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.7334461212158203, 0.6797475814819336]
    },
    "3_Alarmclock": {
        "position": [0.0, -0.07251602411270142, 0.02076128721237183],  # Specific position (z will be added to Z_ELEVATION)
        "orientation": [-0.705507755279541, 0.0, 0.0, 0.7087022066116333]  # Specific rotation
    },
    "3_Mouse": {
        "position": [0.0, 0.0, 0.0],  # Default position
        "orientation": [0.0, 0.0, 0.0, 1.0]  # Normal orientation
    },
    "4_Backpack": {
        "position": [-6.661338147750939e-16, -0.17445747554302216, 0.110100781917572],  # Specific position (z will be added to Z_ELEVATION)
        "orientation": [-0.7060276865959167, 0.0, 0.0, 0.7081843018531799]  # Specific rotation
    },
    "5_Toy_transformer": {
        "position": [0.0, -0.10612589865922928, 0.052749499678611755],
        "orientation": [-0.7070179581642151, 0.0, 0.0, 0.7071956396102905]
    },
    "5_Toy_squirrel": {
        "position": [-0.07678496837615967, 0.020125215873122215, 7.587407890241593e-05],
        "orientation": [0.49860095977783203, -0.5013985633850098, 0.49951183795928955, -0.5004842877388000]
    }
}

def get_models_base_path():
    """Get the base path for models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ibvs_dir = os.path.dirname(script_dir)
    models_path = os.path.join(ibvs_dir, 'models')
    
    if not os.path.exists(models_path):
        models_path = os.path.join(os.path.dirname(script_dir), 'models')
    
    if not os.path.exists(models_path):
        raise RuntimeError(f"Cannot find models directory. Looked in: {models_path}")
    
    return models_path

def get_model_name_from_sdf(sdf_path):
    """Extract model name from SDF file"""
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model_elem = root.find('.//model')
    return model_elem.get('name') if model_elem is not None else None

def delete_all_spawned_models():
    """Delete all models that were spawned by this script"""
    if not rospy.core.is_initialized():
        rospy.init_node('model_spawner', anonymous=True)
    
    try:
        rospy.wait_for_service('/gazebo/get_world_properties', timeout=2.0)
        get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        resp = get_world_props()
        current_models = resp.model_names
        
        rospy.wait_for_service('/gazebo/delete_model', timeout=2.0)
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        models_base = get_models_base_path()
        
        for model in current_models:
            for folder in MODELS.values():
                sdf_path = os.path.join(models_base, folder, 'model.sdf')
                if os.path.exists(sdf_path):
                    model_name = get_model_name_from_sdf(sdf_path)
                    if model_name == model:
                        print(f"Deleting existing model: {model}")
                        try:
                            delete_model(model)
                        except:
                            pass
                        break
                        
    except Exception as e:
        print(f"Note: Could not check/delete existing models: {e}")

def read_and_fix_sdf(model_folder, make_static=True):
    """Read SDF and fix model:// paths"""
    models_base = get_models_base_path()
    model_path = os.path.join(models_base, model_folder)
    sdf_path = os.path.join(model_path, 'model.sdf')
    
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")
    
    # Read the SDF file
    with open(sdf_path, 'r') as f:
        sdf_content = f.read()
    
    # Parse to get model name
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model_elem = root.find('.//model')
    model_name = model_elem.get('name') if model_elem is not None else 'spawned_model'
    
    # Fix model:// URIs and relative paths
    import re
    
    # First, replace any existing model:// URIs to ensure consistency
    sdf_content = re.sub(
        r'model://[^/]+/',
        f'model://{model_folder}/',
        sdf_content
    )
    
    # Then, fix relative mesh paths (meshes/model.obj -> model://model_folder/meshes/model.obj)
    sdf_content = re.sub(
        r'<uri>meshes/',
        f'<uri>model://{model_folder}/meshes/',
        sdf_content
    )
    
    # Also fix texture references if any
    sdf_content = re.sub(
        r'<uri>materials/',
        f'<uri>model://{model_folder}/materials/',
        sdf_content
    )
    
    # If making static, modify the content
    if make_static:
        # Check if <static> tag exists
        if '<static>' not in sdf_content:
            # Add static tag after <model name=...>
            sdf_content = re.sub(
                r'(<model[^>]*>)',
                r'\1\n    <static>true</static>',
                sdf_content
            )
        else:
            # Update existing static tag
            sdf_content = re.sub(
                r'<static>[^<]*</static>',
                '<static>true</static>',
                sdf_content
            )
    
    return sdf_content, model_name

def spawn_model(model_index, x=None, y=None, z=None, use_custom_pose=True, skip_deletion=False):
    if model_index not in MODELS:
        print(f"Invalid model index. Choose from 1-{len(MODELS)}")
        return False
    
    model_folder = MODELS[model_index]
    
    # Initialize ROS node if needed
    if not rospy.core.is_initialized():
        rospy.init_node('model_spawner', anonymous=True)
    
    # Delete existing models only if not already done
    if not skip_deletion:
        print(f"[DEBUG] About to delete existing models for spawning {model_folder}")
        delete_all_spawned_models()
        print(f"[DEBUG] Finished deleting existing models")
    
    # Read and fix SDF
    try:
        sdf_content, model_name = read_and_fix_sdf(model_folder)
    except Exception as e:
        print(f"Error reading SDF: {e}")
        return False
    
    # Create pose
    pose = Pose()
    
    # Set position
    if use_custom_pose and model_folder in MODEL_POSES:
        pose_data = MODEL_POSES[model_folder]
        if x is None:  # Use custom pose
            pos = pose_data["position"]
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2] + Z_ELEVATION
            
            orient = pose_data["orientation"]
            pose.orientation.x = orient[0]
            pose.orientation.y = orient[1]
            pose.orientation.z = orient[2]
            pose.orientation.w = orient[3]
            
            print(f"Using custom pose for {model_folder}")
            print(f"  Position: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")
        else:
            # Position override provided
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0
    else:
        # Default pose
        if x is None:
            x, y, z = 0.0, 0.0, Z_ELEVATION
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
    
    print(f"Spawning {model_folder} as '{model_name}'...")
    
    # Call spawn service
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
        spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        resp = spawn_model_srv(
            model_name=model_name,
            model_xml=sdf_content,
            robot_namespace="",
            initial_pose=pose,
            reference_frame="world"
        )
        
        if resp.success:
            print(f"✓ Successfully spawned {model_folder}")
            print(f"  Model name in Gazebo: '{model_name}'")
            
            # Check if model actually exists in Gazebo
            try:
                rospy.wait_for_service('/gazebo/get_world_properties', timeout=1.0)
                get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
                resp_world = get_world_props()
                if model_name in resp_world.model_names:
                    print(f"[DEBUG] Confirmed: Model '{model_name}' is in Gazebo world")
                else:
                    print(f"[DEBUG] WARNING: Model '{model_name}' NOT found in Gazebo world after spawning!")
                    print(f"[DEBUG] Current models: {resp_world.model_names}")
            except:
                pass
                
            return True
        else:
            print(f"✗ Failed to spawn model: {resp.status_message}")
            return False
            
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return False
    except Exception as e:
        print(f"Error spawning model: {e}")
        return False

def list_models():
    print("\nAvailable GoogleScanNet models:")
    print("=" * 50)
    models_base = get_models_base_path()
    
    for idx, folder in MODELS.items():
        sdf_path = os.path.join(models_base, folder, 'model.sdf')
        if os.path.exists(sdf_path):
            model_name = get_model_name_from_sdf(sdf_path)
            print(f"{idx:2d}. {folder:25} [{model_name}]")
        else:
            print(f"{idx:2d}. {folder}")

def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python3 spawn_model_service.py list               - List all models")
        print("  python3 spawn_model_service.py <number>           - Spawn at origin")
        print("  python3 spawn_model_service.py <number> <x> <y> <z> - Spawn at position")
        print("\nExample:")
        print("  python3 spawn_model_service.py 1                  - Spawn coffee mug")
        print("  python3 spawn_model_service.py 9 0.5 0 0.2        - Spawn keyboard at (0.5, 0, 0.2)")
        print("\nNote: This script requires ROS services to be running (Gazebo must be started)")
        sys.exit(1)
    
    if sys.argv[1] == "list":
        list_models()
    else:
        try:
            model_idx = int(sys.argv[1])
            x = float(sys.argv[2]) if len(sys.argv) > 2 else None
            y = float(sys.argv[3]) if len(sys.argv) > 3 else None
            z = float(sys.argv[4]) if len(sys.argv) > 4 else None
            success = spawn_model(model_idx, x, y, z)
            sys.exit(0 if success else 1)
        except ValueError:
            print("Error: Please provide valid numbers")
            list_models()
            sys.exit(1)

if __name__ == "__main__":
    main()