# NOTE: This is simulation-only code that interfaces directly with Gazebo via ROS services.
# The rospy dependency is required for ROS service calls (wait_for_service, ServiceProxy, etc.)
# and ROS message types (ModelState, Pose, etc.). Only logging has been migrated to Python logging.
# When running on real robot (ROS2), rospy and gazebo_msgs are not available — all functions
# return None/False gracefully.
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rospy
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import GetModelState, SpawnModel, SetModelState, DeleteModel
    from geometry_msgs.msg import Pose
    import tf_conversions
    _GAZEBO_AVAILABLE = True
except ImportError:
    _GAZEBO_AVAILABLE = False
    logger.info("Gazebo/rospy not available — gazebo_utils functions will return None (real robot mode)")


def get_camera_pose():
    """
    Retrieve the current camera position and orientation from Gazebo.

    Returns:
        tuple: (position as np.array, orientation as np.array) or (None, None) if failed
    """
    if not _GAZEBO_AVAILABLE:
        logger.warning("get_camera_pose() called but Gazebo/rospy not available")
        return None, None
    try:
        rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
    except rospy.ROSException:
        logger.error("Timeout waiting for /gazebo/get_model_state service")
        return None, None
        
    try:
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state('realsense2_camera', '')
        
        if not model_state.success:
            logger.error(f"Failed to get model state: {model_state.status_message}")
            return None, None
            
        position = np.array([
            model_state.pose.position.x,
            model_state.pose.position.y,
            model_state.pose.position.z
        ])
        orientation = np.array([
            model_state.pose.orientation.x,
            model_state.pose.orientation.y,
            model_state.pose.orientation.z,
            model_state.pose.orientation.w
        ])
        return position, orientation
    except rospy.ServiceException as e:
        logger.error(f"Service call failed: {e}")
        return None, None


def set_camera_pose(camera_position, orientation_quaternion):
    """
    Set the camera's pose in Gazebo with the given position and orientation.

    Args:
        camera_position (np.ndarray): The position of the camera.
        orientation_quaternion (np.ndarray): The orientation quaternion for the camera.
    """
    if not _GAZEBO_AVAILABLE:
        logger.warning("set_camera_pose() called but Gazebo/rospy not available")
        return
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Create a new state for the camera
        state = ModelState()
        state.model_name = 'realsense2_camera'
        state.pose.position.x = camera_position[0]
        state.pose.position.y = camera_position[1]
        state.pose.position.z = camera_position[2]
        state.pose.orientation.x = orientation_quaternion[0]
        state.pose.orientation.y = orientation_quaternion[1]
        state.pose.orientation.z = orientation_quaternion[2]
        state.pose.orientation.w = orientation_quaternion[3]
        state.reference_frame = 'world'

        # Set the new state in Gazebo
        set_state(state)
    except rospy.ServiceException as e:
        logger.error(f"Service call failed: {e}")


def manage_gazebo_models(model_index):
    """
    Delete the current model and spawn a new perturbed model in Gazebo.

    Args:
        model_index (int): The index of the perturbed model to spawn (1-500).
    """
    if not _GAZEBO_AVAILABLE:
        logger.warning("manage_gazebo_models() called but Gazebo/rospy not available")
        return
    # Delete the current model
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # If it's the first iteration, delete the original "resized" model
        if model_index == 1:
            model_to_delete = "resized"
        else:
            model_to_delete = f"resized{model_index - 1}"

        delete_model(model_to_delete)
        logger.info(f"Deleted model: {model_to_delete}")
    except rospy.ServiceException as e:
        logger.error(f"Failed to delete model {model_to_delete}: {e}")

    # Spawn the new perturbed model
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Construct the path to the new model
        # Try multiple path strategies for compatibility
        model_path = None
        
        # Strategy 1: Check if running in vision container with mounted path
        vision_container_path = f"/root/vision_ws/src/ibvs/models/viso{model_index}/model.sdf"
        if os.path.exists(vision_container_path):
            model_path = vision_container_path
        
        # Strategy 2: Check standard catkin workspace path
        if not model_path:
            catkin_path = os.path.expanduser(f"~/catkin_ws/src/ibvs/models/viso{model_index}/model.sdf")
            if os.path.exists(catkin_path):
                model_path = catkin_path
        
        # Strategy 3: Use relative path from current file location
        if not model_path:
            current_file_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            relative_model_path = os.path.join(current_file_dir, f"models/viso{model_index}/model.sdf")
            if os.path.exists(relative_model_path):
                model_path = relative_model_path
        
        # Strategy 4: Check environment variable
        if not model_path:
            ibvs_path = os.environ.get('IBVS_PATH')
            if ibvs_path:
                env_model_path = os.path.join(ibvs_path, f"models/viso{model_index}/model.sdf")
                if os.path.exists(env_model_path):
                    model_path = env_model_path

        # Check if we found a valid model path
        if not model_path:
            logger.error(f"Could not find model viso{model_index} in any expected location")
            logger.error("Tried: /root/vision_ws/src/ibvs/models/, ~/catkin_ws/src/ibvs/models/, relative path")
            logger.error("Make sure perturbed models have been generated with generate_perturbed_hollywood.py")
            return
        
        # Log which path we're using
        logger.info(f"Found perturbed model at: {model_path}")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return

        with open(model_path, "r") as f:
            model_xml = f.read()

        # Set the pose for the new model
        initial_pose = Pose()
        initial_pose.position.x = 0
        initial_pose.position.y = 0
        initial_pose.position.z = 0.005

        # Convert Euler angles to quaternion
        quaternion = tf_conversions.transformations.quaternion_from_euler(1.5708, 0, 1.5708)
        initial_pose.orientation.x = quaternion[0]
        initial_pose.orientation.y = quaternion[1]
        initial_pose.orientation.z = quaternion[2]
        initial_pose.orientation.w = quaternion[3]

        # Spawn the model
        new_model_name = f"resized{model_index}"
        spawn_model(new_model_name, model_xml, "", initial_pose, "world")

        logger.info(f"Spawned perturbed model: {new_model_name}")
    except rospy.ServiceException as e:
        logger.error(f"Failed to spawn model {new_model_name}: {e}")
