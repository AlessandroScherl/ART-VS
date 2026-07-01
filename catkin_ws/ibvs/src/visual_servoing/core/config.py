import yaml
import os
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for visual servoing parameters."""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_parameters()
        
    def load_parameters(self):
        """Load parameters from a YAML configuration file."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Camera and image parameters
        self.u_max = config['u_max']  # Image width
        self.v_max = config['v_max']  # Image height
        self.f_x = config['f_x']  # Focal length x
        self.f_y = config['f_y']  # Focal length y
        self.c_x = self.u_max / 2  # Principal point x
        self.c_y = self.v_max / 2  # Principal point y

        # Control parameters
        self.lambda_ = config['lambda_']  # Control gain
        self.max_velocity = config.get('max_velocity', 1.0)
        # min_error and max_error removed - not used in control law
        self.num_pairs = config['num_pairs']

        # Feature extraction parameters  
        self.backbone_model = config.get('backbone_model', 'am-radio')
        self.thresh_filter_keypoints = config['thresh_filter_keypoints']
        self.vit_input_size = config['vit_input_size']
        self.use_feature_binning = config['use_feature_binning']
        self.feature_hierarchy = config.get('feature_hierarchy', 1)  # Hierarchy for log binning (1=9 bins, 2=17 bins)
        self.background_thresh = config.get('background_thresh', 0.5)

        # Tiling parameters
        self.use_tiling = config.get('use_tiling', True)
        self.tile_number = config.get('tile_number', 4)
        self.tile_input_resolution = config.get('tile_input_resolution', 896)
        self.auto_calculate_tiles = config.get('auto_calculate_tiles', True)  # Default to True (auto-calc enabled)
        
        # Hybrid mode parameters
        self.use_hybrid_mode = config.get('use_hybrid_mode', False)
        self.hybrid_switch_threshold = config.get('hybrid_switch_threshold', 0.1)
        
        # NEW: Iteration-based tiling switch (0 = use error-based)
        self.tiling_switch_iteration = config.get('tiling_switch_iteration', 0)

        # Sampling parameters
        self.num_samples = config['num_samples']
        self.num_circles = config['num_circles']
        self.circle_radius_aug = config['circle_radius_aug']

        # Convergence parameters
        self.velocity_convergence_threshold = config['velocity_convergence_threshold']
        self.velocity_threshold_translation = config['velocity_threshold_translation']
        self.velocity_threshold_rotation = config['velocity_threshold_rotation']
        self.error_threshold_ratio = config['error_threshold_ratio']
        self.error_threshold_absolute_translation = config['error_threshold_absolute_translation']
        self.error_threshold_absolute_rotation = config['error_threshold_absolute_rotation']

        # Iteration control
        self.min_iterations = config['min_iterations']
        self.max_iterations = config['max_iterations']

        # EMA parameter
        self.ema_alpha = config.get('ema_alpha', 0.1)

        # Velocity history
        self.max_velocity_vector_history = config.get('max_velocity_vector_history', 200)

        # Set the image path
        current_directory = os.path.dirname(os.path.abspath(self.config_path))
        # Handle both absolute and relative paths
        if os.path.isabs(config['image_path']):
            self.image_path = config['image_path']
        else:
            # Navigate up to find the images folder
            # From visual_servoing/ go up to ibvs/, then into images/goal/goalrgb_1440x1080/
            # current_directory could be configs/ or configs/test2_5models/ etc.
            visual_servoing_dir = os.path.dirname(os.path.dirname(current_directory))
            if not visual_servoing_dir.endswith('visual_servoing'):
                # If in subdirectory like configs/test2_5models, go up one more
                visual_servoing_dir = os.path.dirname(visual_servoing_dir)

            src_dir = os.path.dirname(visual_servoing_dir)
            ibvs_dir = os.path.dirname(src_dir)
            images_dir = os.path.join(ibvs_dir, 'images', 'goal', 'goalrgb_1440x1080')
            self.image_path = os.path.join(images_dir, config['image_path'])

            # Try alternative location if not found
            if not os.path.exists(self.image_path):
                self.image_path = os.path.join(current_directory, config['image_path'])

        # Check if the goal image exists (warn only, don't fail - experiment script will update it)
        if not os.path.exists(self.image_path):
            logger.warning(f"Goal image not found at: {self.image_path} (will be updated by experiment script)")
            # Don't raise error - allow experiment script to override image_path later
        
        # Set the mask path (optional)
        if 'mask_path' in config and config['mask_path'] is not None:
            # Handle both absolute and relative paths
            if os.path.isabs(config['mask_path']):
                self.mask_path = config['mask_path']
            else:
                # Masks are in consolidated location: ibvs/images/masks/
                # Navigate up from visual_servoing to ibvs
                visual_servoing_dir = os.path.dirname(os.path.dirname(current_directory))
                if not visual_servoing_dir.endswith('visual_servoing'):
                    visual_servoing_dir = os.path.dirname(visual_servoing_dir)

                src_dir = os.path.dirname(visual_servoing_dir)
                ibvs_dir = os.path.dirname(src_dir)
                masks_dir = os.path.join(ibvs_dir, 'images', 'masks')
                self.mask_path = os.path.join(masks_dir, config['mask_path'])

                # Fallback: try same directory as config
                if not os.path.exists(self.mask_path):
                    self.mask_path = os.path.join(current_directory, config['mask_path'])

            # Verify mask exists if specified
            if not os.path.exists(self.mask_path):
                logger.warning(f"Mask image not found at: {self.mask_path}, proceeding without mask")
                self.mask_path = None
        else:
            self.mask_path = None
            
        # ROI detection parameters
        self.use_roi_detection = config.get('use_roi_detection', False)
        self.roi_padding_ratio = config.get('roi_padding_ratio', 0.1)
        self.yolo_confidence_threshold = config.get('yolo_confidence_threshold', 0.3)
        self.yolo_model_size = config.get('yolo_model_size', 's')
        self.yolo_detector_type = config.get('yolo_detector_type', 'yoloworld')  # 'yoloworld' or 'yoloe'

        # LangSAM + LightTrack detector parameters (alternative to YOLO-World)
        self.detector_type = config.get('detector_type', 'yoloworld')  # 'langsam_lighttrack' or 'yoloworld'
        # Default weights: LIGHTTRACK_WEIGHTS env var, then native path, then Docker path
        _default_weights = os.environ.get('LIGHTTRACK_WEIGHTS',
                                          os.path.expanduser('~/LightTrack/snapshot/LightTrackM/LightTrackM.pth'))
        if not os.path.exists(_default_weights):
            _default_weights = '/root/vision_ws/src/models/LightTrack/snapshot/LightTrackM/LightTrackM.pth'
        self.lighttrack_weights = config.get('lighttrack_weights', _default_weights)
        self.lighttrack_arch = config.get('lighttrack_arch', 'LightTrackM_Subnet')
        self.cache_timeout = config.get('cache_timeout', 10)  # iterations to use cached bbox before re-detection

        # ROI tiling parameters
        self.use_roi_tiling = config.get('use_roi_tiling', False)
        self.roi_tile_number = config.get('roi_tile_number', 4)
        self.roi_tiling_switch_threshold = config.get('roi_tiling_switch_threshold', 0.2)
        
        # Visualization control
        self.enable_visualization = config.get('enable_visualization', True)
        # Renderer for the per-iteration correspondence overlay:
        #   'matplotlib' (default) - original, slower (~150ms/frame canvas render)
        #   'cv2'                  - fast cv2 drawing, ~10-50x faster (recommended for live demo)
        self.visualization_backend = config.get('visualization_backend', 'matplotlib')
        
        # Rotation compensation parameters
        self.rotation_mode = config.get('rotation_mode', 'discrete')  # 'discrete' or 'continuous'
        self.use_continuous_rotation = config.get('use_continuous_rotation', False)  # Legacy support
        # Override rotation_mode if use_continuous_rotation is set
        if self.use_continuous_rotation:
            self.rotation_mode = 'continuous'
        self.rotation_search_range = config.get('rotation_search_range', 30.0)  # degrees
        self.rotation_gradient_delta = config.get('rotation_gradient_delta', 5.0)  # degrees
        self.rotation_learning_rate = config.get('rotation_learning_rate', 2.0)  # degrees/iter
        self.rotation_convergence_threshold = config.get('rotation_convergence_threshold', 0.001)
        self.rotation_max_iterations = config.get('rotation_max_iterations', 15)

        # ============================================
        # REAL ROBOT INTEGRATION PARAMETERS
        # ============================================
        # Robot mode: 'simulation' (Gazebo/ROS1) or 'real' (ROS2 + MoveIt Servo - see docs/REAL_ROBOT.md)
        self.robot_mode = config.get('robot_mode', 'simulation')

        # Configurable ROS topics (with backward-compatible defaults for simulation)
        self.camera_rgb_topic = config.get('camera_rgb_topic', '/camera/color/image_raw')
        self.camera_depth_topic = config.get('camera_depth_topic', '/camera/depth/image_raw')
        self.velocity_topic = config.get('velocity_topic', '/camera_vel')

        # TF frames for real robot (only used when robot_mode='real')
        self.camera_frame = config.get('camera_frame', 'camera_color_optical_frame')
        self.tool_frame = config.get('tool_frame', 'tool0')
        self.base_frame = config.get('base_frame', 'base')
        # Frame in which velocity commands are expressed (MoveIt Servo command frame)
        self.velocity_frame_id = config.get('velocity_frame_id', 'tool0')

        # Convergence mode: 'full' (position + orientation from TF/Gazebo) or 'visual_only' (feature error only)
        # Real robot uses 'visual_only' since we don't have ground truth pose
        self.convergence_mode = config.get('convergence_mode', 'full' if self.robot_mode == 'simulation' else 'visual_only')

        # Real-robot convergence criteria. Must be set as attributes here or the YAML values
        # are ignored — controller.py reads them via config.__dict__.get(..., default).
        self.real_robot_error_reduction_target = config.get('real_robot_error_reduction_target', 0.90)
        self.real_robot_max_iterations = config.get('real_robot_max_iterations', 300)

        # Velocity convergence (for visual-only mode) - can be disabled for real robot
        # When disabled, only feature stagnation is used for convergence
        self.velocity_convergence_enabled = config.get('velocity_convergence_enabled', True)

        # Detection keyword/text prompt for ROI detection (LangSAM or YOLOWorld)
        # For real robot, set this to describe your target object (e.g., "red mug", "screwdriver")
        self.detection_keyword = config.get('detection_keyword', None)
