"""ROS2 (rclpy) visual servoing controller for real-robot deployment.

The simulation benchmark uses the ROS1 controller in ros_controller.py; this module
is its ROS2 port, publishing TwistStamped commands consumed by MoveIt Servo.
"""
import os
import sys

# Ensure parent directory is in path for local module imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Twist, TwistStamped
from tf2_ros import Buffer as TF2Buffer, TransformListener as TF2TransformListener
import tf2_ros
from ros_interface.tf_compat import quaternion_matrix as _quaternion_matrix
import threading
import copy

# Try to import cv_bridge with fallback
try:
    from cv_bridge import CvBridge
except (ImportError, TypeError) as e:
    print(f"Warning: Native cv_bridge import failed: {e}")
    print("Using cv_bridge wrapper with fallback implementation")
    from ros_interface.cv_bridge_wrapper import CvBridge
from PIL import Image
import numpy as np
import cv2
import time
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# Import from other modules using absolute imports
from core.controller import VisualServoingController
from features.matcher import find_correspondences_batch, scale_points_from_patch
from features.yolo_world_detector import YOLOWorldDetector
try:
    from features.yoloe_detector import YOLOEDetector
except ImportError:
    YOLOEDetector = None
try:
    from features.yolo_fallback_detector import YOLOFallbackDetector
except ImportError:
    YOLOFallbackDetector = None
try:
    from features.yolo_smart_detector import YOLOSmartDetector
except ImportError:
    YOLOSmartDetector = None
try:
    from features.langsam_lighttrack_detector import LangSAMLightTrackDetector, LANGSAM_LIGHTTRACK_AVAILABLE
except ImportError:
    LangSAMLightTrackDetector = None
    LANGSAM_LIGHTTRACK_AVAILABLE = False
from vs_utils.visualization import visualize_correspondences_ros, visualize_correspondences_ros_cv2
from ros_interface.gazebo_utils import get_camera_pose


class ROSVisualServoingController:
    """ROS wrapper for the visual servoing controller."""
    
    def __init__(self, config, feature_extractor, desired_position, desired_orientation,
                 pre_initialized_detector=None, pre_detected_goal_roi_bbox=None, node=None,
                 tf_buffer=None, executor=None):
        self.config = config
        self.node = node
        self._executor = executor  # For pause/resume during CUDA ops
        self.feature_extractor = feature_extractor
        self.desired_position = desired_position
        self.desired_orientation = desired_orientation
        
        # Initialize core controller
        self.vs_controller = VisualServoingController(config)
        
        # ROI tiling tracking
        self.roi_tiling_activated = False
        self.roi_tiling_switch_iteration = None
        
        # Automatic tile number calculation
        self.optimal_tile_number = None  # Will be calculated if hybrid mode is enabled
        
        # Classical method failure tracking
        self.consecutive_no_features_count = 0
        self.max_consecutive_no_features = 5  # Abort after 5 consecutive failures
        
        # ROS components
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_depth = None
        self.latest_pil_image = None
        self.camera_position = None
        self.orientation_quaternion = None
        
        # Feature failure tracking
        self.feature_failure_count = 0
        self.last_num_features = 0  # Track number of features detected

        # Feature error tracking for real-world switching (no ground truth needed)
        self.initial_feature_error = None  # Average pixel error at first iteration
        self.current_feature_error = None  # Current average pixel error

        # Cached goal image processing for performance
        self.goal_image_resized = None
        self.goal_tensor = None
        self.goal_features = None
        self.goal_roi_resized = None
        self.goal_roi_tensor = None
        self.goal_roi_features = None
        self.goal_roi_mask_path = None
        
        # Initialize ROI detector (YOLO-World or LangSAM+LightTrack)
        self.yolo_detector = None
        self.detector_name = "NONE"  # Will be set to actual detector name
        self.yolo_keyword = None
        self.goal_roi_bbox = None  # Cached goal ROI
        self.current_roi_bbox = None  # Cached current ROI
        self.last_valid_roi_iteration = 0  # Track when last valid ROI was detected

        # LightTrack tracking state
        self.tracking_started = False  # True after rotation alignment and tracking initialized

        if pre_initialized_detector is not None:
            # Use detector created early in run_visual_servoing.py (before rotation)
            self.yolo_detector = pre_initialized_detector
            self.detector_name = "LangSAM+LightTrack (pre-initialized)"
            if hasattr(pre_initialized_detector, 'is_tracking') and pre_initialized_detector.is_tracking:
                self.tracking_started = True
            if pre_detected_goal_roi_bbox is not None:
                self.goal_roi_bbox = pre_detected_goal_roi_bbox
            self.node.get_logger().info(f"Using pre-initialized detector: {self.detector_name}")
            self.node.get_logger().info(f"  Tracking active: {self.tracking_started}")
            self.node.get_logger().info(f"  Goal ROI: {self.goal_roi_bbox}")

        elif self.config.use_roi_detection:
            # Check which detector type to use
            detector_type = getattr(self.config, 'detector_type', 'yoloworld')
            yolo_detector_type = getattr(self.config, 'yolo_detector_type', 'yoloworld')

            try:
                # Try LangSAM + LightTrack first if configured
                if detector_type == 'langsam_lighttrack':
                    if LangSAMLightTrackDetector is not None and LANGSAM_LIGHTTRACK_AVAILABLE:
                        self.yolo_detector = LangSAMLightTrackDetector(
                            lighttrack_weights=self.config.lighttrack_weights,
                            lighttrack_arch=self.config.lighttrack_arch,
                            confidence_threshold=self.config.yolo_confidence_threshold,
                            cache_timeout=self.config.cache_timeout,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        self.detector_name = "LangSAM+LightTrack"
                        self.node.get_logger().info("Using LangSAM + LightTrack detector for ROI detection")
                    else:
                        self.node.get_logger().warn("LangSAM+LightTrack not available, falling back to YOLO-World")
                        detector_type = 'yoloworld'

                # Fall back to YOLO-based detectors
                if detector_type != 'langsam_lighttrack' or self.yolo_detector is None:
                    if yolo_detector_type == 'smart' and YOLOSmartDetector is not None:
                        # Use Smart detector
                        self.yolo_detector = YOLOSmartDetector(
                            model_size=self.config.yolo_model_size,
                            confidence_threshold=self.config.yolo_confidence_threshold,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        self.detector_name = "YOLO-Smart"
                        self.node.get_logger().info("Using Smart YOLO detector for ROI detection")
                    elif yolo_detector_type == 'yoloe' and YOLOEDetector is not None:
                        # Try YOLO-E detector
                        try:
                            self.yolo_detector = YOLOEDetector(
                                model_size=self.config.yolo_model_size,
                                confidence_threshold=self.config.yolo_confidence_threshold,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                            self.detector_name = "YOLO-E"
                            self.node.get_logger().info("Using YOLO-E detector for ROI detection")
                        except Exception as e:
                            self.node.get_logger().warn(f"YOLO-E initialization failed: {e}, falling back to YOLOWorld")
                            self.yolo_detector = YOLOWorldDetector(
                                model_size=self.config.yolo_model_size,
                                confidence_threshold=self.config.yolo_confidence_threshold,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                            self.detector_name = "YOLO-World"
                    else:
                        # Try YOLOWorld first
                        try:
                            self.yolo_detector = YOLOWorldDetector(
                                model_size=self.config.yolo_model_size,
                                confidence_threshold=self.config.yolo_confidence_threshold,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                            self.detector_name = "YOLO-World"
                            self.node.get_logger().info("Using YOLOWorld detector for ROI detection")
                        except Exception as e:
                            self.node.get_logger().warn(f"YOLOWorld failed: {e}, trying fallback detector")
                            # Try fallback detector
                            if YOLOFallbackDetector is not None:
                                self.yolo_detector = YOLOFallbackDetector(
                                    model_size=self.config.yolo_model_size,
                                    confidence_threshold=self.config.yolo_confidence_threshold,
                                    device='cuda' if torch.cuda.is_available() else 'cpu'
                                )
                                self.detector_name = "YOLO-Fallback"
                                self.node.get_logger().info("Using YOLO Fallback detector for ROI detection")
                            else:
                                raise
            except Exception as e:
                self.node.get_logger().error(f"Failed to initialize any ROI detector: {e}")
                self.config.use_roi_detection = False

        # Initialize TF2 for real robot mode
        self.tf_listener = None
        if tf_buffer is not None:
            # Reuse existing TF buffer (already populated by caller)
            self.tf_buffer = tf_buffer
            self.node.get_logger().info("TF2: reusing existing buffer from caller")
        elif getattr(self.config, 'robot_mode', 'simulation') == 'real':
            self.tf_buffer = TF2Buffer()
            self.tf_listener = TF2TransformListener(self.tf_buffer, self.node)
            self.node.get_logger().info("TF2 initialized for real robot mode")
        else:
            self.tf_buffer = None

        # Setup ROS communication
        self.setup_ros_communication()
        
        # Load goal image
        self.goal_image = self.load_goal_image(config.image_path)
        self.goal_image_array = np.array(self.goal_image)
        
        # Cache goal image processing for performance
        self._cache_goal_image_processing()

        # Auto-set detection keyword from config (for real robot operation)
        detection_keyword = getattr(config, 'detection_keyword', None)
        if detection_keyword and self.config.use_roi_detection:
            if pre_initialized_detector is not None and self.goal_roi_bbox is not None:
                # Detector and goal ROI already set up - skip redundant goal detection
                self.yolo_keyword = detection_keyword
                if hasattr(self.yolo_detector, 'set_classes'):
                    self.yolo_detector.set_classes([detection_keyword])
                # Cache goal ROI features (mask cropping handled inside _cache_goal_roi_features)
                self._cache_goal_roi_features()
                self._calculate_optimal_tile_number(self.goal_roi_bbox)
                self.node.get_logger().info(f"Using pre-detected goal ROI, keyword: '{detection_keyword}'")
            else:
                self.node.get_logger().info(f"Setting detection keyword from config: '{detection_keyword}'")
                self.set_yolo_keyword(detection_keyword)

        # Wait for first image (use configurable topic).
        # Note: If the executor is paused during init (to avoid GIL contention
        # during model loading), this will timeout. The VS loop will get images
        # once the executor is resumed by the caller.
        rgb_topic = getattr(config, 'camera_rgb_topic', '/camera/color/image_raw')
        self.node.get_logger().info(f"Waiting for the first image on {rgb_topic}...")
        import time as _time
        # Short wait: the executor is intentionally paused here, so this almost always
        # times out and the real image wait happens in the VS loop. 0.5s is enough to
        # catch the case where a frame is already buffered; longer is just dead time.
        deadline = _time.time() + 0.5
        while self.latest_image is None and _time.time() < deadline:
            _time.sleep(0.05)  # Background executor handles callbacks
        if self.latest_image is None:
            self.node.get_logger().warn(
                f"No image yet on {rgb_topic} (executor may be paused). "
                "Will receive images when VS loop starts.")
        else:
            self.node.get_logger().info("First image received!")
    
    def setup_ros_communication(self):
        """Setup ROS publishers and subscribers."""
        # Get configurable topics (with backward-compatible defaults)
        rgb_topic = getattr(self.config, 'camera_rgb_topic', '/camera/color/image_raw')
        depth_topic = getattr(self.config, 'camera_depth_topic', '/camera/depth/image_raw')
        velocity_topic = getattr(self.config, 'velocity_topic', '/camera_vel')
        robot_mode = getattr(self.config, 'robot_mode', 'simulation')

        self.node.get_logger().info(f"Robot mode: {robot_mode}")
        self.node.get_logger().info(f"RGB topic: {rgb_topic}")
        self.node.get_logger().info(f"Depth topic: {depth_topic}")
        self.node.get_logger().info(f"Velocity topic: {velocity_topic}")

        # Subscribers — QoS depth=1 keeps ONLY the latest frame.
        # With depth>1, old frames queue up during ibvs() (no callbacks processed)
        # and spin_for_callbacks delivers stale images instead of fresh ones.
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        _img_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST)
        self.image_sub_rgb = self.node.create_subscription(
            ImageMsg, rgb_topic, self.image_callback_rgb, _img_qos)
        self.image_sub_depth = self.node.create_subscription(
            ImageMsg, depth_topic, self.image_callback_depth, _img_qos)
        self.node.get_logger().info(
            f"Subscriptions created: RGB={self.image_sub_rgb.topic_name}, "
            f"Depth={self.image_sub_depth.topic_name}")

        # Velocity publisher - TwistStamped for MoveIt Servo
        self.pub = self.node.create_publisher(TwistStamped, velocity_topic, 10)
        # Disabled - not needed, reduces overhead (set to None, still referenced in visualization calls)
        self.image_pub = None
        self.goal_image_pub = None
        self.current_image_pub = None
        self.correspondence_pub = self.node.create_publisher(ImageMsg, '/correspondence_visualization', 10)
        self.vit_space_tiled_pub = self.node.create_publisher(ImageMsg, '/vs/vit_space_correspondences_tiled', 10)

        # 100Hz twist republisher for MoveIt Servo watchdog.
        # Uses a standalone Python thread instead of a ROS timer so that it
        # keeps publishing even when the executor is paused (during ibvs()).
        # Without continuous 100Hz publishing, Servo's watchdog (100ms timeout)
        # zeros velocity during the ~200ms ibvs() computation, causing the
        # robot to stutter instead of moving smoothly.
        self.latest_twist_stamped = TwistStamped()
        # Set frame_id so Servo never receives an empty source_frame
        robot_mode = getattr(self.config, 'robot_mode', 'simulation')
        if robot_mode == 'real':
            self.latest_twist_stamped.header.frame_id = getattr(self.config, 'tool_frame', 'tool0')
        self._image_event = threading.Event()  # OS-level wait for fresh camera frame (no GIL contention)
        self._twist_lock = threading.Lock()
        self._twist_thread_active = False  # Activated by publish_twist()
        self._twist_thread_stop = threading.Event()
        # Cached camera→tool0 transform (static via URDF — looked up once, reused).
        # Avoids per-iteration tf_buffer.lookup_transform() stalls during heavy load.
        self._cam_to_tool_cached = None
        self._twist_thread = threading.Thread(
            target=self._twist_republish_loop, daemon=True)
        self._twist_thread.start()

        self.node.get_logger().info("ROS publishers and subscribers initialized")

    def _twist_republish_loop(self):
        """Background thread: republish latest twist at 100Hz, independent of executor."""
        while not self._twist_thread_stop.is_set():
            if self._twist_thread_active:
                try:
                    with self._twist_lock:
                        msg = copy.copy(self.latest_twist_stamped)
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    self.pub.publish(msg)
                except Exception:
                    pass  # Node may be shutting down
            self._twist_thread_stop.wait(0.01)  # 100Hz

    def stop_twist_timer(self):
        """Stop the twist republisher and send zero velocity."""
        self._twist_thread_active = False
        try:
            zero_twist = TwistStamped()
            zero_twist.header.stamp = self.node.get_clock().now().to_msg()
            robot_mode = getattr(self.config, 'robot_mode', 'simulation')
            if robot_mode == 'real':
                zero_twist.header.frame_id = getattr(self.config, 'tool_frame', 'tool0')
            else:
                zero_twist.header.frame_id = getattr(self.config, 'camera_frame', 'camera_color_optical_frame')
            self.pub.publish(zero_twist)
        except Exception:
            pass
        self._twist_thread_stop.set()

    def cleanup(self):
        """Tear down everything this controller created on the shared node.

        Used by the persistent-worker path (run_visual_servoing.py --worker), which
        creates a fresh controller per pick on a long-lived node. Without this, each
        pick would leak a velocity publisher + a 100Hz twist thread + subscriptions,
        and multiple twist threads would fight over the servo topic.

        NOTE: image_sub_depth is intentionally NOT destroyed here — the worker
        reassigns it to a shared, long-lived early depth subscription after construction,
        so destroying it would kill the shared sub. The controller's OWN depth sub is
        already destroyed by the worker right after construction.
        """
        # Stop the 100Hz republisher thread and send a final zero velocity.
        try:
            self.stop_twist_timer()
        except Exception:
            pass
        # Give the thread a moment to exit its 10ms wait loop, then join.
        try:
            if hasattr(self, '_twist_thread') and self._twist_thread.is_alive():
                self._twist_thread.join(timeout=1.0)
        except Exception:
            pass
        # Destroy publishers/subscriptions created in __init__ (and the lazy vit_space_pub).
        for attr in ('image_sub_rgb', 'pub', 'correspondence_pub',
                     'vit_space_tiled_pub', 'vit_space_pub'):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                if attr == 'image_sub_rgb':
                    self.node.destroy_subscription(obj)
                else:
                    self.node.destroy_publisher(obj)
            except Exception:
                pass

    def image_callback_rgb(self, msg):
        """Callback for RGB image messages."""
        # Profile image conversion
        callback_start = time.perf_counter()
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2_time = (time.perf_counter() - callback_start) * 1000

        # Profile PIL conversion
        pil_start = time.perf_counter()
        self.latest_pil_image = Image.fromarray(cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB))
        pil_time = (time.perf_counter() - pil_start) * 1000

        self._image_event.set()  # Wake main thread waiting for fresh frame

        total_time = (time.perf_counter() - callback_start) * 1000
        if total_time > 5:
            self.node.get_logger().debug(f"[PERF] Image callback: cv2={cv2_time:.1f}ms, PIL={pil_time:.1f}ms, total={total_time:.1f}ms")

    def image_callback_depth(self, msg):
        """Callback for depth image messages."""
        try:
            first = self.latest_image_depth is None
            self.latest_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if first:
                self.node.get_logger().info(
                    f"First depth image received: {self.latest_image_depth.shape}, "
                    f"dtype={self.latest_image_depth.dtype}")
        except Exception as e:
            self.node.get_logger().error(f"Depth callback error: {e}")
    
    def load_goal_image(self, image_path):
        """Load the goal image from the specified path."""
        try:
            goal_image = Image.open(image_path)
            goal_image = goal_image.convert('RGB')
            self.node.get_logger().info(f".... got goal image from {image_path}")

            return goal_image
        except Exception as e:
            self.node.get_logger().error(f"Failed to load image at {image_path}: {e}")
            raise
    
    def get_current_image(self):
        """Get the current camera image as PIL Image."""
        if hasattr(self, 'latest_pil_image') and self.latest_pil_image is not None:
            return self.latest_pil_image
        else:
            self.node.get_logger().warn("No current image available")
            return None
    
    def get_goal_image(self):
        """Get the goal image as PIL Image."""
        if hasattr(self, 'goal_image') and self.goal_image is not None:
            return self.goal_image
        else:
            self.node.get_logger().warn("No goal image available")
            return None
    
    
    
    def _cache_goal_image_processing(self):
        """Cache goal image processing to avoid redundant computation."""
        cache_start = time.perf_counter()
        
        # Check if using classical features - no caching needed for classical methods
        if hasattr(self.config, 'backbone_model') and 'classical-' in self.config.backbone_model:
            self.node.get_logger().debug("Classical feature detection - no goal image caching needed")
            return
        
        # Cache resized goal image
        self.node.get_logger().debug("Caching goal image processing...")
        self.goal_image_resized = self.goal_image.resize((self.config.vit_input_size, self.config.vit_input_size))
        
        # Cache preprocessed tensor
        self.goal_tensor = self.feature_extractor.preprocess_pil(self.goal_image_resized)
        
        # Cache goal features
        with torch.no_grad():
            self.goal_features = self.feature_extractor.extract_descriptors(
                self.goal_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )
        
        cache_time = (time.perf_counter() - cache_start) * 1000
        self.node.get_logger().debug(f"Goal image processing cached in {cache_time:.2f}ms")
        self.node.get_logger().debug(f"  - Resized image: {self.goal_image_resized.size}")
        self.node.get_logger().debug(f"  - Tensor shape: {self.goal_tensor.shape}")
        self.node.get_logger().debug(f"  - Features shape: {self.goal_features.shape}")
    
    def _cache_goal_roi_features(self):
        """Cache goal ROI features for ROI-based detection."""
        if self.goal_roi_bbox is None:
            return

        cache_start = time.perf_counter()
        is_classical = 'classical-' in self.config.backbone_model

        # Crop goal ROI
        goal_roi = self.yolo_detector.crop_roi(self.goal_image, self.goal_roi_bbox)

        # Store original ROI size for proper coordinate mapping
        self.goal_roi_original_size = goal_roi.size

        if not is_classical:
            # Resize to VIT input size - force square for compatibility with ViT
            self.goal_roi_resized = goal_roi.resize((self.config.vit_input_size, self.config.vit_input_size))

            # Preprocess and cache tensor
            self.goal_roi_tensor = self.feature_extractor.preprocess_pil(self.goal_roi_resized)

            # Extract and cache features
            with torch.no_grad():
                self.goal_roi_features = self.feature_extractor.extract_descriptors(
                    self.goal_roi_tensor.to(self.feature_extractor.device),
                    layer=11,
                    facet='token',
                    bin=self.config.use_feature_binning,
                    hierarchy=self.config.feature_hierarchy
                )
        else:
            # Classical methods: store the ROI crop for use in detect_features_classical_roi
            self.goal_roi_image = goal_roi

        # Cache ROI mask if needed
        if self.config.mask_path:
            # Load original mask
            mask_img = Image.open(self.config.mask_path).convert('L')

            # Crop mask to match goal ROI
            x1, y1, x2, y2 = self.goal_roi_bbox
            mask_roi = mask_img.crop((x1, y1, x2, y2))

            if not is_classical:
                # Resize mask to VIT input size
                mask_roi_resized = mask_roi.resize(
                    (self.config.vit_input_size, self.config.vit_input_size),
                    Image.Resampling.LANCZOS
                )

                # Save temporary resized mask
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    mask_roi_resized.save(tmp.name, 'JPEG')
                    self.goal_roi_mask_path = tmp.name

        cache_time = (time.perf_counter() - cache_start) * 1000
        self.node.get_logger().debug(f"Goal ROI features cached in {cache_time:.2f}ms")
        self.node.get_logger().debug(f"  - ROI bbox: {self.goal_roi_bbox}")
    
    def get_depth(self, current_points):
        """Get depth values for the current feature points."""
        if self.latest_image_depth is None:
            self.node.get_logger().warn("No depth image received yet")
            return None

        z_values_meter = np.zeros((len(current_points), 1))
        height, width = self.latest_image_depth.shape

        # Log depth image dimensions once
        if not hasattr(self, '_depth_dims_logged'):
            self.node.get_logger().info(f"[Depth] Image dimensions: {width}x{height}")
            self._depth_dims_logged = True

        out_of_bounds_count = 0
        zero_depth_count = 0

        for count, point in enumerate(current_points):
            x, y = int(point[0]), int(point[1])

            # Ensure the point is within the image bounds
            if 0 <= x < width and 0 <= y < height:
                depth_value = self.latest_image_depth[y, x]
                # Convert depth value to meters
                if depth_value != 0:
                    z_values_meter[count] = depth_value / 1000.0
                else:
                    z_values_meter[count] = 0.2  # Default 20cm for zero depth (closer than 100m)
                    zero_depth_count += 1
            else:
                z_values_meter[count] = 0.2  # Default for out of bounds
                out_of_bounds_count += 1

        # Log a depth summary every 50 iterations
        iter_count = self.vs_controller.iteration_count if hasattr(self, 'vs_controller') else 0
        if iter_count % 50 == 0:
            valid_depths = z_values_meter[z_values_meter < 10]  # Exclude default values
            if len(valid_depths) > 0:
                self.node.get_logger().debug(
                    f"[Depth] Iter {iter_count}: mean={np.mean(valid_depths):.3f}m, "
                    f"min={np.min(valid_depths):.3f}m, max={np.max(valid_depths):.3f}m, "
                    f"zero={zero_depth_count}, oob={out_of_bounds_count}")
            else:
                self.node.get_logger().warn(
                    f"[Depth] Iter {iter_count}: no valid depth values (zero={zero_depth_count}, oob={out_of_bounds_count})")

        if out_of_bounds_count > 0:
            self.node.get_logger().warn(f"[Depth] {out_of_bounds_count} points out of bounds (image: {width}x{height})")

        return z_values_meter
    
    def publish_twist(self, v_c):
        """Publish velocity commands via MoveIt Servo.

        For real robot: Transforms velocity from camera frame to tool0 frame,
        then publishes as TwistStamped. MoveIt Servo handles tool0->base internally.
        """
        # Activate the 100Hz republisher thread on the first real velocity.
        if not self._twist_thread_active:
            self._twist_thread_active = True

        v_c_clipped = self.vs_controller.store_applied_velocities(v_c)

        camera_twist = Twist()
        camera_twist.linear.x = v_c_clipped[0]
        camera_twist.linear.y = v_c_clipped[1]
        camera_twist.linear.z = v_c_clipped[2]
        camera_twist.angular.x = v_c_clipped[3]
        camera_twist.angular.y = v_c_clipped[4]
        camera_twist.angular.z = v_c_clipped[5]

        if np.any(np.abs(v_c) > self.config.max_velocity):
            self.node.get_logger().warn("Velocity capped due to exceeding maximum allowed value.")

        robot_mode = getattr(self.config, 'robot_mode', 'simulation')

        if robot_mode == 'real':
            # Transform v_c from camera_color_optical_frame to tool0.
            # The transform is static (URDF-defined) — look up ONCE and cache.
            # Avoids per-iteration tf_buffer.lookup_transform() which can stall
            # under heavy load and cause Servo to halt the robot.
            camera_frame = getattr(self.config, 'camera_frame', 'camera_color_optical_frame')
            tool_frame = getattr(self.config, 'tool_frame', 'tool0')

            if self._cam_to_tool_cached is None:
                # First call: look up the static transform with a generous timeout.
                from rclpy.time import Time
                try:
                    self._cam_to_tool_cached = self.tf_buffer.lookup_transform(
                        tool_frame, camera_frame, Time(),
                        timeout=rclpy.duration.Duration(seconds=2.0))
                    self.node.get_logger().info(
                        f"Cached static transform {camera_frame}→{tool_frame}")
                except Exception as e:
                    self.node.get_logger().error(
                        f"Failed to look up static transform {camera_frame}→{tool_frame}: {e}")
                    # Don't publish — better to skip than to send wrong-frame command
                    return

            tool_twist = self._transform_twist(camera_twist, self._cam_to_tool_cached)

            ts = TwistStamped()
            ts.header.stamp = self.node.get_clock().now().to_msg()
            ts.header.frame_id = tool_frame
            ts.twist = tool_twist
        else:
            # Simulation mode - publish directly as TwistStamped (no transform needed)
            ts = TwistStamped()
            ts.header.stamp = self.node.get_clock().now().to_msg()
            ts.twist = camera_twist

        # Store for the 100Hz republisher safety net
        with self._twist_lock:
            self.latest_twist_stamped = ts

        # Also publish immediately — don't wait for the next 100Hz tick.
        # Belt-and-suspenders against the republisher being GIL-starved.
        try:
            self.pub.publish(ts)
        except Exception:
            pass

    def _transform_twist(self, twist, transform):
        """Transform a Twist message using a TF2 transform.

        Args:
            twist: geometry_msgs/Twist message
            transform: TF2 transform (from lookup_transform)

        Returns:
            Transformed Twist message
        """
        # Extract rotation quaternion
        q = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]

        # Get rotation matrix from quaternion
        rot_matrix = _quaternion_matrix(q)

        # Transform linear velocity
        linear = [twist.linear.x, twist.linear.y, twist.linear.z, 0.0]
        linear_transformed = rot_matrix.dot(linear)

        # Transform angular velocity
        angular = [twist.angular.x, twist.angular.y, twist.angular.z, 0.0]
        angular_transformed = rot_matrix.dot(angular)

        # Create transformed twist
        transformed_twist = Twist()
        transformed_twist.linear.x = linear_transformed[0]
        transformed_twist.linear.y = linear_transformed[1]
        transformed_twist.linear.z = linear_transformed[2]
        transformed_twist.angular.x = angular_transformed[0]
        transformed_twist.angular.y = angular_transformed[1]
        transformed_twist.angular.z = angular_transformed[2]

        return transformed_twist

    def get_camera_pose_tf(self):
        """Get camera pose from TF tree (for real robot mode).

        Returns:
            tuple: (position, quaternion) or (None, None) on failure
                - position: np.array([x, y, z])
                - quaternion: np.array([x, y, z, w])
        """
        if self.tf_buffer is None:
            self.node.get_logger().error("TF buffer not initialized - cannot get camera pose")
            return None, None

        try:
            base_frame = getattr(self.config, 'base_frame', 'base')
            camera_frame = getattr(self.config, 'camera_frame', 'camera_color_optical_frame')

            transform = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time(), rclpy.time.Duration(seconds=0.05))

            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            quaternion = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

            return position, quaternion

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.node.get_logger().error(f"TF lookup error in get_camera_pose_tf: {e}")
            return None, None

    def get_camera_pose(self):
        """Get camera pose based on robot mode.

        In simulation mode: Uses Gazebo service
        In real robot mode: Uses TF2 tree

        Returns:
            tuple: (position, quaternion) or (None, None) on failure
        """
        robot_mode = getattr(self.config, 'robot_mode', 'simulation')

        if robot_mode == 'real':
            return self.get_camera_pose_tf()
        else:
            # Use Gazebo service (imported at top of file)
            return get_camera_pose()
    
    def _viz_correspondences(self, *args, **kwargs):
        """Dispatch correspondence visualization to the configured backend.

        'cv2'        -> fast cv2 drawing (~10-50x faster, recommended for live demo)
        'matplotlib' -> original matplotlib renderer (default; fallback if cv2 misbehaves)
        Switch via `visualization_backend` in the YAML config.
        """
        backend = getattr(self.config, 'visualization_backend', 'matplotlib')
        if backend == 'cv2':
            visualize_correspondences_ros_cv2(*args, **kwargs)
        else:
            visualize_correspondences_ros(*args, **kwargs)

    def draw_points(self, image, current_points, goal_points):
        """Draw current and goal feature points on the image."""
        # Check if visualization is enabled
        if not getattr(self.config, 'enable_visualization', True):
            return
            
        for x, y in current_points:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Current points in green
        for x, y in goal_points:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # Goal points in red

        # Disabled - not needed, reduces overhead
        # ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        # self.image_pub.publish(ros_image)
    
    def check_roi_tiling_switch(self):
        """Check if we should switch to ROI tiling mode.

        Activates tiling when EITHER condition is met (whichever comes first):
        1. Iteration threshold reached (e.g., 150 iterations)
        2. Feature error reduced by threshold (e.g., 80% reduction)
        """
        # Check ROI tiling mode
        if (self.config.use_roi_detection and hasattr(self.config, 'use_roi_tiling')
            and self.config.use_roi_tiling and not self.roi_tiling_activated):

            # Don't check on iteration 0 - we need at least one iteration to establish baseline
            if self.vs_controller.iteration_count == 0:
                return

            activated = False
            reason = ""

            # Condition 1: Iteration-based (e.g., 150 iterations)
            if hasattr(self.config, 'tiling_switch_iteration') and self.config.tiling_switch_iteration > 0:
                if self.vs_controller.iteration_count >= self.config.tiling_switch_iteration:
                    activated = True
                    reason = f"Iteration threshold ({self.config.tiling_switch_iteration}) reached"

            # Condition 2: Error-based (e.g., 80% reduction) - check even if iteration-based is configured
            if not activated and self.initial_feature_error is not None and self.initial_feature_error > 0:
                if self.current_feature_error is not None and self.vs_controller.iteration_count >= 10:
                    feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)
                    roi_threshold = getattr(self.config, 'hybrid_switch_threshold',
                                          getattr(self.config, 'roi_tiling_switch_threshold', 0.2))

                    if feature_error_reduction >= (1.0 - roi_threshold):
                        activated = True
                        reason = f"Feature error reduced by {feature_error_reduction*100:.1f}%"

            # Activate if either condition met
            if activated:
                self.roi_tiling_activated = True
                self.roi_tiling_switch_iteration = self.vs_controller.iteration_count
                print(f"\n>>> SWITCHING TO ROI TILING MODE (iter {self.vs_controller.iteration_count}) - {reason}")
                if self.initial_feature_error and self.current_feature_error:
                    print(f"    Initial error: {self.initial_feature_error:.2f}px, Current: {self.current_feature_error:.2f}px\n")
    
    def check_tiling_switch(self):
        """Check if we should switch to tiled mode for full image (MODE 2)."""
        # Only check if we haven't already switched and hybrid mode is enabled
        if self.tiling_activated or not self.config.use_hybrid_mode:
            return

        # Don't check on iteration 0 - we need at least one iteration to establish baseline
        if self.vs_controller.iteration_count == 0:
            return
        
        # Check if we should use iteration-based switching
        if hasattr(self.config, 'tiling_switch_iteration') and self.config.tiling_switch_iteration > 0:
            # Debug output on first check
            if self.vs_controller.iteration_count == 1:
                print(f"[Tiling Switch] Using iteration-based switching at iteration {self.config.tiling_switch_iteration}")
            
            # Iteration-based switching
            if self.vs_controller.iteration_count >= self.config.tiling_switch_iteration:
                self.tiling_activated = True
                self.tiling_switch_iteration = self.vs_controller.iteration_count
                print(f"\n>>> SWITCHING TO FULL IMAGE TILING MODE (iter {self.vs_controller.iteration_count}) - Iteration threshold reached\n")
            
            # Return early - we're using iteration-based, don't check error-based
            return
            
        # Feature-error-based switching (works in real world without ground truth)
        if self.vs_controller.iteration_count == 1:
            print(f"[Tiling Switch] Using feature-error-based switching (threshold: {(1-getattr(self.config, 'hybrid_switch_threshold', 0.2))*100:.0f}% reduction)")

        # Need initial feature error to calculate reduction
        if (self.initial_feature_error is not None and
            self.initial_feature_error > 0 and
            self.current_feature_error is not None):

            # Calculate feature error reduction (0.0 = no reduction, 1.0 = 100% reduction)
            feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)

            # Get hybrid switch threshold (default 0.2 = switch at 80% reduction)
            threshold = getattr(self.config, 'hybrid_switch_threshold', 0.2)

            # Log progress every 100 iterations
            if self.vs_controller.iteration_count % 100 == 0:
                target_error = self.initial_feature_error * threshold
                print(f"[Tiling Progress] Iter {self.vs_controller.iteration_count}: Error {self.current_feature_error:.1f}px → need {target_error:.1f}px (reduction: {feature_error_reduction*100:.1f}% / {(1-threshold)*100:.0f}%)")

            # Switch when feature error has been reduced by the threshold amount
            if feature_error_reduction >= (1.0 - threshold):
                self.tiling_activated = True
                self.tiling_switch_iteration = self.vs_controller.iteration_count
                print(f"\n>>> SWITCHING TO FULL IMAGE TILING MODE (iter {self.vs_controller.iteration_count}) - Feature error reduced by {feature_error_reduction*100:.1f}%")
                print(f"    Initial error: {self.initial_feature_error:.2f} pixels, Current: {self.current_feature_error:.2f} pixels\n")
    
    def detect_features(self):
        """Main feature detection method."""
        # Performance tracking
        detect_start = time.perf_counter()

        # CRITICAL: Check for classical methods FIRST, before ROI detection branch
        # Classical methods (SIFT/ORB/AKAZE) have their own detection pipeline
        is_classical = hasattr(self.config, 'backbone_model') and 'classical-' in self.config.backbone_model

        if is_classical:
            # Classical methods - route to classical detection (with or without ROI)
            if self.config.use_roi_detection and self.yolo_detector is not None:
                result = self.detect_features_classical_roi()
            else:
                result = self.detect_features_classical()

            # Handle classical result
            if result is not None and result[0] is not None:
                self.consecutive_no_features_count = 0
            else:
                self.node.get_logger().warn("[Classical] No features detected!")
                self.consecutive_no_features_count += 1
                if self.consecutive_no_features_count >= self.max_consecutive_no_features:
                    self.node.get_logger().error(f"[Classical] No features for {self.consecutive_no_features_count} consecutive iterations - aborting")
                    raise RuntimeError("Persistent feature detection failure")

            total_time = (time.perf_counter() - detect_start) * 1000
            self.node.get_logger().debug(f"[PERF] Total detect_features (classical): {total_time:.2f}ms")
            return result

        # ViT methods - existing logic unchanged
        # Check if we should switch to ROI tiling mode
        switch_start = time.perf_counter()
        self.check_roi_tiling_switch()
        switch_time = (time.perf_counter() - switch_start) * 1000
        if switch_time > 1:
            self.node.get_logger().debug(f"[PERF] ROI switch check: {switch_time:.2f}ms")

        # Choose detection method for ViT
        if self.config.use_roi_detection and self.yolo_detector is not None:
            # Check if ROI tiling is enabled and we've reached the error threshold
            if (hasattr(self.config, 'use_roi_tiling') and self.config.use_roi_tiling
                and self.roi_tiling_activated):
                result = self.detect_features_roi_tiled()
            else:
                result = self.detect_features_roi()
        else:
            # MODE 2: Check for full image tiling with hybrid mode
            if (self.config.use_tiling and self.config.use_hybrid_mode):
                # Check if we should switch to tiled mode
                self.check_tiling_switch()

                if self.tiling_activated:
                    result = self.detect_features_tiled()
                else:
                    result = self.detect_features_original()
            elif self.config.use_tiling:
                # Tiling enabled but not hybrid - always use tiled
                result = self.detect_features_tiled()
            else:
                # No tiling - use original
                result = self.detect_features_original()

        total_time = (time.perf_counter() - detect_start) * 1000
        self.node.get_logger().debug(f"[PERF] Total detect_features: {total_time:.2f}ms")
        return result
    
    def detect_features_original(self):
        """Original detect features method without tiling (ViT only).

        Note: Classical methods are handled at the top level of detect_features()
        and will never reach this method.
        """
        if self.latest_image is None:
            return None, None

        # Use cached goal image data - no need to resize or extract features!
        goal_image_resized = self.goal_image_resized
        desc1 = self.goal_features
        
        # Only process current image
        resize_start = time.perf_counter()
        current_image_resized = self.latest_pil_image.resize((self.config.vit_input_size, self.config.vit_input_size))
        resize_time = (time.perf_counter() - resize_start) * 1000
        self.node.get_logger().debug(f"[PERF] Current image resize: {resize_time:.2f}ms (goal cached)")

        with torch.no_grad():
            # Only preprocess current image
            preprocess_start = time.perf_counter()
            current_tensor = self.feature_extractor.preprocess_pil(current_image_resized)
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000
            self.node.get_logger().debug(f"[PERF] Current preprocessing: {preprocess_time:.2f}ms (goal cached)")

            # Only extract features for current image
            extract_start = time.perf_counter()
            desc2 = self.feature_extractor.extract_descriptors(
                current_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )
            extract_time = (time.perf_counter() - extract_start) * 1000
            self.node.get_logger().debug(f"[PERF] Current feature extraction: {extract_time:.2f}ms (goal cached)")
            
            # Check if features are on GPU
            if desc2.is_cuda:
                self.node.get_logger().debug(f"[GPU] Features on GPU, shape: {desc2.shape}")

            # Profile feature matching
            match_start = time.perf_counter()
            self.node.get_logger().debug(f"Calling find_correspondences_batch with mask_path: {self.config.mask_path}")
            points1, points2, sim_selected_12 = find_correspondences_batch(
                desc1, desc2,
                num_pairs=self.config.num_pairs,
                mask_path=self.config.mask_path,
                vit_input_size=self.config.vit_input_size,
                patch_size=self.feature_extractor.get_patch_size()
            )
            match_time = (time.perf_counter() - match_start) * 1000
            self.node.get_logger().debug(f"[PERF] Feature matching: {match_time:.2f}ms")

            if points1 is None or points2 is None:
                self.feature_failure_count += 1
                self.last_num_features = 0  # No features found
                if self.feature_failure_count >= 10:
                    self.node.get_logger().error("Feature detection failed 10 times in a row")
                    raise RuntimeError("Persistent feature detection failure")
                return None, None

            # Reset counter on successful detection
            self.feature_failure_count = 0
            
            # Store actual number of features found
            self.last_num_features = len(points1) if points1 is not None else 0

            # Check for GPU->CPU transfer  
            if torch.is_tensor(points1) and points1.is_cuda:
                size_mb = (points1.numel() + points2.numel()) * 4 / (1024 * 1024)
                self.node.get_logger().debug(f"[GPU->CPU] Points will be transferred: {size_mb:.3f}MB")

            # Convert to VIT input space for visualization (points are in patch coordinates)
            scale = self.config.vit_input_size / int(np.sqrt(desc1.size(-2)))  
            patch_size = self.feature_extractor.get_patch_size()
            points1_vit = points1 * scale + patch_size / 2
            points2_vit = points2 * scale + patch_size / 2

            # Flip points from (x,y) to (y,x) for visualization
            # The visualization function expects points in (y,x) order
            if torch.is_tensor(points1_vit):
                points1_viz = torch.flip(points1_vit, dims=[1])
                points2_viz = torch.flip(points2_vit, dims=[1])
            else:
                points1_viz = np.flip(np.array(points1_vit), axis=1)
                points2_viz = np.flip(np.array(points2_vit), axis=1)

            # Visualize correspondences (only if visualization is enabled)
            if self.config.enable_visualization:
                self._viz_correspondences(
                    goal_image_resized, current_image_resized,
                    points1_viz, points2_viz,
                    None, self.bridge, self.correspondence_pub,
                    self.goal_image_pub, self.current_image_pub,
                    False, None  # No tiling
                )

            # Profile calculate_uv - pass VIT coordinates since calculate_uv expects VIT input size
            calc_start = time.perf_counter()
            result = self.calculate_uv(points1_vit.tolist(), points2_vit.tolist()), sim_selected_12
            calc_time = (time.perf_counter() - calc_start) * 1000
            if calc_time > 5:
                self.node.get_logger().debug(f"[PERF] calculate_uv: {calc_time:.2f}ms")
            return result
    
    def calculate_uv(self, goal_features, current_features):
        """Calculate feature points and scale them to the real image resolution."""
        num_pairs = self.config.num_pairs

        # Profile numpy operations
        numpy_start = time.perf_counter()
        # Keep coordinates in (x,y) format - no flip needed as they match (u,v)
        s_uv_star_resized = np.asarray(goal_features)
        s_uv_resized = np.asarray(current_features)

        # Calculate the scaled feature values
        s_uv = np.zeros([num_pairs, 2], dtype=int)
        s_uv_star = np.zeros([num_pairs, 2], dtype=int)
        numpy_time = (time.perf_counter() - numpy_start) * 1000
        if numpy_time > 1:
            self.node.get_logger().debug(f"[PERF] calculate_uv numpy ops: {numpy_time:.2f}ms")

        if len(goal_features) != num_pairs:
            # Silently adjust - this is normal when mask filters some patches
            num_pairs = len(goal_features)
            if num_pairs < 4:
                self.node.get_logger().warn("Too few features detected (<4). Skipping processing.")
                return s_uv_star, s_uv

        # Scale factors to convert from DINO input size to original image size
        scale_x = self.config.u_max / self.config.vit_input_size
        scale_y = self.config.v_max / self.config.vit_input_size

        for count in range(num_pairs):
            s_uv_star[count, 0] = round(s_uv_star_resized[count, 0] * scale_x)
            s_uv_star[count, 1] = round(s_uv_star_resized[count, 1] * scale_y)
            s_uv[count, 0] = round(s_uv_resized[count, 0] * scale_x)
            s_uv[count, 1] = round(s_uv_resized[count, 1] * scale_y)

        return s_uv_star, s_uv
    
    def calculate_uv_roi(self, goal_features, current_features):
        """Calculate feature points for ROI mode - points are already in full image coordinates."""
        num_pairs = self.config.num_pairs

        
        # NOTE: No flip needed - points are already in (x,y) format which matches (u,v)
        # where x/u is horizontal and y/v is vertical
        s_uv_star = np.asarray(goal_features).astype(int)
        s_uv = np.asarray(current_features).astype(int)
        

        if len(goal_features) != num_pairs:
            # Silently adjust - this is normal when mask filters some patches
            num_pairs = len(goal_features)
            if num_pairs < 4:
                self.node.get_logger().warn("Too few features detected (<4). Skipping processing.")
                return np.zeros([0, 2], dtype=int), np.zeros([0, 2], dtype=int)

        # Points are already in full image coordinates, just ensure they're within bounds
        s_uv_star[:, 0] = np.clip(s_uv_star[:, 0], 0, self.config.u_max - 1)
        s_uv_star[:, 1] = np.clip(s_uv_star[:, 1], 0, self.config.v_max - 1)
        s_uv[:, 0] = np.clip(s_uv[:, 0], 0, self.config.u_max - 1)
        s_uv[:, 1] = np.clip(s_uv[:, 1], 0, self.config.v_max - 1)

        self.last_num_features = num_pairs  # Store actual number of features used
        return s_uv_star[:num_pairs], s_uv[:num_pairs]
    
    def set_yolo_keyword(self, keyword):
        """Set the YOLO detection keyword for the current model."""
        if self.yolo_detector is not None:
            self.yolo_keyword = keyword
            self.yolo_detector.set_classes([keyword])
            
            # Reset current ROI cache when switching models
            self.current_roi_bbox = None
            self.last_valid_roi_iteration = 0
            
            # Detect and cache goal ROI
            detection = self.yolo_detector.detect_and_cache_goal(
                self.goal_image,
                keyword,
                self.config.roi_padding_ratio
            )

            if detection['bbox'] is not None:
                self.goal_roi_bbox = detection['padded_bbox']
                self.node.get_logger().info(f"[{self.detector_name} GOAL DETECTION] Goal image size: {self.goal_image.size}")
                self.node.get_logger().info(f"[{self.detector_name} GOAL DETECTION] Detected bbox: {detection['bbox']}")
                self.node.get_logger().info(f"[{self.detector_name} GOAL DETECTION] Padded bbox: {detection['padded_bbox']}")

                # Calculate what the bbox would be if scaled to 1440 or 1920
                goal_width = self.goal_image.size[0]
                if goal_width == 1920:
                    scale_to_1440 = 1440 / 1920
                    scaled_bbox = [
                        detection['padded_bbox'][0] * scale_to_1440,
                        detection['padded_bbox'][1],
                        detection['padded_bbox'][2] * scale_to_1440,
                        detection['padded_bbox'][3]
                    ]
                    self.node.get_logger().info(f"[{self.detector_name} GOAL DETECTION] If scaled to 1440: {scaled_bbox}")
                elif goal_width == 1440:
                    scale_to_1920 = 1920 / 1440
                    scaled_bbox = [
                        detection['padded_bbox'][0] * scale_to_1920,
                        detection['padded_bbox'][1],
                        detection['padded_bbox'][2] * scale_to_1920,
                        detection['padded_bbox'][3]
                    ]
                    self.node.get_logger().info(f"[{self.detector_name} GOAL DETECTION] If scaled to 1920: {scaled_bbox}")
                # Cache goal ROI features when ROI is detected
                self._cache_goal_roi_features()
                # Calculate optimal tile number if hybrid mode is enabled
                self._calculate_optimal_tile_number(detection['bbox'])
            else:
                self.node.get_logger().warn(f"No detection in goal image for '{keyword}', will use full image")
                self.goal_roi_bbox = None
                # Set goal ROI features to use full image features as fallback
                self.goal_roi_features = self.goal_features
                self.goal_roi_resized = self.goal_image_resized
                self.goal_roi_tensor = self.goal_tensor
                self.goal_roi_mask_path = self.config.mask_path if hasattr(self.config, 'mask_path') else None
    
    def detect_features_roi(self):
            if len(current_descriptors.shape) == 3:
                current_descriptors = current_descriptors.unsqueeze(1)
            
            # Clear batch tensors to free memory
            del goal_batch, current_batch
            torch.cuda.empty_cache() if self.feature_extractor.device.type == 'cuda' else None
            
            # Find correspondences for each tile
            all_matches = []
            num_pairs_per_tile = max(1, self.config.num_pairs // self.tiling_config.slice_number)
            
            for tile_idx in range(self.tiling_config.slice_number):
                desc1_tile = goal_descriptors[tile_idx:tile_idx+1]
                desc2_tile = current_descriptors[tile_idx:tile_idx+1]
                
                # For tiling, we'll apply masking after global coordinate transformation
                # since the mask is defined in the original image space
                points1, points2, similarities = find_correspondences_batch(
                    desc1_tile, desc2_tile,
                    num_pairs=num_pairs_per_tile,
                    distance_threshold=1,
                    mask_path=None,  # Masking will be applied after global transformation
                    patch_size=self.feature_extractor.get_patch_size()
                )
                
                if points1 is not None:
                    if len(similarities.shape) > 1:
                        similarities = similarities.squeeze()
                    
                    all_matches.append({
                        'tile_idx': tile_idx,
                        'points1': points1,
                        'points2': points2,
                        'similarities': similarities
                    })
            
            if not all_matches:
                self.feature_failure_count += 1
                if self.feature_failure_count >= 10:
                    self.node.get_logger().error("Feature detection failed 10 times in a row")
                    raise RuntimeError("Persistent feature detection failure")
                return None, None
            
            # Reset counter on successful detection
            self.feature_failure_count = 0
            
            # Transform all matches to global coordinates
            global_matches = []
            
            for match_data in all_matches:
                tile_idx = match_data['tile_idx']
                points1 = match_data['points1']
                points2 = match_data['points2']
                similarities = match_data['similarities']
                
                for i in range(len(points1)):
                    # Points are in patch coordinates (x,y format)
                    # For a VIT input with 14x14 patches, calculate number of patches
                    # Scale from patch space to VIT input pixel coordinates
                    num_patches = self.config.vit_input_size // 14
                    scale = self.config.vit_input_size / num_patches
                    
                    # Scale points to pixel coordinates within VIT input
                    p1_pixel = points1[i] * scale + scale / 2  # Still in (x,y) format
                    p2_pixel = points2[i] * scale + scale / 2
                    
                    # Debug first point transformation
                    if i == 0 and tile_idx == 0:
                        self.node.get_logger().info(f"[TILING TRANSFORM DEBUG] Tile {tile_idx}:")
                        self.node.get_logger().info(f"  Original patch coord: x={points1[i][0]:.2f}, y={points1[i][1]:.2f}")
                        self.node.get_logger().info(f"  VIT pixel coord: x={p1_pixel[0]:.2f}, y={p1_pixel[1]:.2f}")
                    
                    # Now scale from VIT input size to tile size in the tiled image
                    tile_scale = self.config.slice_resolution / self.config.vit_input_size
                    
                    # Transform to tile pixel coordinates
                    p1_tile_pixel = p1_pixel * tile_scale
                    p2_tile_pixel = p2_pixel * tile_scale
                    
                    # Get tile offsets for goal and current images
                    goal_tile_y, goal_tile_x = goal_tile_coords[tile_idx]
                    current_tile_y, current_tile_x = current_tile_coords[tile_idx]
                    
                    # Convert to global coordinates (still in x,y format)
                    global_p1_x = goal_tile_x + p1_tile_pixel[0]
                    global_p1_y = goal_tile_y + p1_tile_pixel[1]
                    global_p2_x = current_tile_x + p2_tile_pixel[0]
                    global_p2_y = current_tile_y + p2_tile_pixel[1]
                    
                    if i == 0 and tile_idx == 0:
                        self.node.get_logger().info(f"  Tile pixel coord: x={p1_tile_pixel[0]:.2f}, y={p1_tile_pixel[1]:.2f}")
                        self.node.get_logger().info(f"  Tile offset: x={goal_tile_x}, y={goal_tile_y}")
                        self.node.get_logger().info(f"  Global coord: x={global_p1_x:.2f}, y={global_p1_y:.2f}")
                    
                    # Create tensors in (y,x) format for consistency with visualization
                    global_p1 = torch.tensor([global_p1_y, global_p1_x], device=points1.device)
                    global_p2 = torch.tensor([global_p2_y, global_p2_x], device=points2.device)
                    
                    global_matches.append({
                        'p1': global_p1,
                        'p2': global_p2,
                        'similarity': similarities[i],
                        'tile_idx': tile_idx
                    })
            
            # Apply mask filtering if mask is provided
            if self.config.mask_path:
                # Load and resize mask to match tiling input resolution
                from PIL import Image as PILImage
                mask_img = PILImage.open(self.config.mask_path).convert('L')
                mask_img = mask_img.resize((self.tiling_config.input_resolution, 
                                           self.tiling_config.input_resolution), 
                                          PILImage.LANCZOS)
                mask_array = np.array(mask_img) > 128
                
                # Filter global matches based on mask
                filtered_matches = []
                for match in global_matches:
                    # Check if the goal point (p1) is within the mask
                    p1_x = int(match['p1'][1].item() if torch.is_tensor(match['p1'][1]) else match['p1'][1])
                    p1_y = int(match['p1'][0].item() if torch.is_tensor(match['p1'][0]) else match['p1'][0])
                    
                    # Ensure coordinates are within bounds
                    if (0 <= p1_x < self.tiling_config.input_resolution and 
                        0 <= p1_y < self.tiling_config.input_resolution and
                        mask_array[p1_y, p1_x]):
                        filtered_matches.append(match)
                
                global_matches = filtered_matches
            
            # Sort by similarity and take top k matches
            global_matches.sort(key=lambda x: x['similarity'], reverse=True)
            k = min(self.config.num_pairs, len(global_matches))
            top_matches = global_matches[:k]
            
            # Extract points for visualization and processing
            points1_global = torch.stack([m['p1'] for m in top_matches])
            points2_global = torch.stack([m['p2'] for m in top_matches])
            
            # Extract similarity scores
            sim_scores = []
            for m in top_matches:
                sim = m['similarity']
                if torch.is_tensor(sim):
                    sim_scores.append(sim.item() if sim.numel() == 1 else sim.mean().item())
                else:
                    sim_scores.append(float(sim))
            sim_scores = torch.tensor(sim_scores)

            # Convert to UV coordinates for original image
            return self.tiling_handler.calculate_uv_roi(
                points1_global.tolist(), points2_global.tolist()), sim_scores
    
    def detect_features_tiled(self):
        """Detect features using full image tiling (MODE 2)."""
        if self.latest_image is None:
            return None, None
            
        # Get the effective tile number (auto-calculated or from config)
        effective_tile_number = self._get_full_image_tile_number()
        
        grid_size = int(np.sqrt(effective_tile_number))
        total_tiles = grid_size * grid_size
        tile_size = self.config.vit_input_size
        
        # self.node.get_logger().info(f"[Full Image Tiling] Processing {total_tiles} tiles ({grid_size}x{grid_size} grid)")  # Commented out to reduce console output
        
        # First resize full images to tiling resolution
        tiling_resolution = grid_size * tile_size
        goal_image_resized = self.goal_image.resize((tiling_resolution, tiling_resolution))
        current_image_resized = self.latest_pil_image.resize((tiling_resolution, tiling_resolution))
        
        # Create tiles
        goal_tiles = []
        current_tiles = []
        tile_positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                tile_x = j * tile_size
                tile_y = i * tile_size
                
                # Extract tiles
                goal_tile = goal_image_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                current_tile = current_image_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                
                goal_tiles.append(goal_tile)
                current_tiles.append(current_tile)
                tile_positions.append((i, j))
        
        # Batch preprocess all tiles
        goal_tensors = []
        current_tensors = []
        
        for goal_tile, current_tile in zip(goal_tiles, current_tiles):
            goal_tensor = self.feature_extractor.preprocess_pil(goal_tile)
            current_tensor = self.feature_extractor.preprocess_pil(current_tile)
            goal_tensors.append(goal_tensor)
            current_tensors.append(current_tensor)
        
        # Stack into batches
        goal_batch = torch.cat(goal_tensors, dim=0).to(self.feature_extractor.device)
        current_batch = torch.cat(current_tensors, dim=0).to(self.feature_extractor.device)
        
        # Batch extract features
        with torch.no_grad():
            desc1_batch = self.feature_extractor.extract_descriptors(
                goal_batch,
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )
            desc2_batch = self.feature_extractor.extract_descriptors(
                current_batch,
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )

        # Load mask if available
        mask_resized = None
        if self.config.mask_path:
            # Resize mask to tiling resolution
            from PIL import Image
            mask_pil = Image.open(self.config.mask_path).convert('L')
            mask_resized = mask_pil.resize((tiling_resolution, tiling_resolution))
        
        # Process correspondences for each tile
        all_points1 = []
        all_points2 = []
        all_similarities = []
        vit_tiles_data = []  # For visualization
        
        for tile_idx in range(total_tiles):
            i, j = tile_positions[tile_idx]
            
            # Extract features for this tile from the batch
            desc1 = desc1_batch[tile_idx:tile_idx+1]
            desc2 = desc2_batch[tile_idx:tile_idx+1]
            
            # Create mask for this tile if available
            mask_tile_tensor = None
            if self.config.mask_path and mask_resized:
                tile_x = j * tile_size
                tile_y = i * tile_size
                mask_tile = mask_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                
                from features.matcher import create_patch_mask_from_pil
                actual_patch_size = self.feature_extractor.get_patch_size()
                mask_tile_tensor = create_patch_mask_from_pil(
                    mask_tile, 
                    vit_input_size=self.config.vit_input_size,
                    patch_size=actual_patch_size
                ).to(self.feature_extractor.device)
            
            # Find correspondences for this tile
            pairs_per_tile = max(4, self.config.num_pairs // total_tiles)
            self.node.get_logger().debug(f"[Full Image Tiling] Processing tile ({i},{j}) - requesting {pairs_per_tile} pairs")

            points1, points2, sim_scores = find_correspondences_batch(
                desc1, desc2,
                num_pairs=pairs_per_tile,
                mask_tensor=mask_tile_tensor,
                vit_input_size=self.config.vit_input_size,
                patch_size=self.feature_extractor.get_patch_size()
            )

            if points1 is not None and len(points1) > 0:
                self.node.get_logger().debug(f"[Full Image Tiling] Tile ({i},{j}) SUCCESS: Got {len(points1)} correspondences")
                # Convert tensors to numpy if needed
                if torch.is_tensor(points1):
                    points1_np = points1.cpu().numpy()
                    points2_np = points2.cpu().numpy()
                    sim_scores_np = sim_scores.cpu().numpy()
                else:
                    points1_np = points1
                    points2_np = points2
                    sim_scores_np = sim_scores
                
                # Scale points from patch coordinates to pixel coordinates
                # Points are in patch coordinates, need to scale to VIT input size
                patch_size = self.feature_extractor.get_patch_size()
                scale = self.config.vit_input_size / int(np.sqrt(desc1.size(-2)))
                points1_scaled = points1_np * scale + patch_size / 2
                points2_scaled = points2_np * scale + patch_size / 2
                
                # Transform points to global coordinates
                global_offset_x = j * tile_size
                global_offset_y = i * tile_size
                
                points1_global = points1_scaled.copy()
                points2_global = points2_scaled.copy()
                
                points1_global[:, 0] += global_offset_x
                points1_global[:, 1] += global_offset_y
                points2_global[:, 0] += global_offset_x
                points2_global[:, 1] += global_offset_y
                
                all_points1.extend(points1_global.tolist())
                all_points2.extend(points2_global.tolist())
                all_similarities.extend(sim_scores_np.tolist())
                
                # Store visualization data for this tile
                vit_tiles_data.append({
                    'tile_idx': (i, j),
                    'goal_tile': goal_tiles[tile_idx],
                    'current_tile': current_tiles[tile_idx],
                    'points1_vit': points1_scaled.copy(),  # Already in VIT space
                    'points2_vit': points2_scaled.copy()
                })
            else:
                # Tile failed to produce correspondences
                self.node.get_logger().error(f"[Full Image Tiling] ERROR: Tile ({i},{j}) FAILED - no correspondences returned! (requested {pairs_per_tile} pairs)")
                self.node.get_logger().error(f"[Full Image Tiling] This likely means no valid correspondences passed the distance threshold and mask filters. Enable debug logging to see details.")

        if len(all_points1) == 0:
            self.node.get_logger().warn("[Full Image Tiling] No features found in any tile")
            return None, None
        
        # Convert lists to arrays
        all_points1 = np.array(all_points1)
        all_points2 = np.array(all_points2)
        all_similarities = np.array(all_similarities)
        
        # Sort by similarity and select top num_pairs
        sorted_indices = np.argsort(all_similarities)[::-1][:self.config.num_pairs]
        
        final_points1 = all_points1[sorted_indices]
        final_points2 = all_points2[sorted_indices]
        
        # Scale points from tiling resolution to original image resolution
        scale_x = self.config.u_max / tiling_resolution
        scale_y = self.config.v_max / tiling_resolution
        
        # Scale up and convert to UV coordinates
        s_uv_star = np.zeros([len(final_points1), 2], dtype=int)
        s_uv = np.zeros([len(final_points2), 2], dtype=int)
        
        for i in range(len(final_points1)):
            s_uv_star[i, 0] = int(final_points1[i, 0] * scale_x)
            s_uv_star[i, 1] = int(final_points1[i, 1] * scale_y)
            s_uv[i, 0] = int(final_points2[i, 0] * scale_x)
            s_uv[i, 1] = int(final_points2[i, 1] * scale_y)
        
        # Clip to image bounds
        s_uv_star[:, 0] = np.clip(s_uv_star[:, 0], 0, self.config.u_max - 1)
        s_uv_star[:, 1] = np.clip(s_uv_star[:, 1], 0, self.config.v_max - 1)
        s_uv[:, 0] = np.clip(s_uv[:, 0], 0, self.config.u_max - 1)
        s_uv[:, 1] = np.clip(s_uv[:, 1], 0, self.config.v_max - 1)
        
        self.last_num_features = len(s_uv_star)
        
        # Get the best similarities for the selected points
        final_similarities = all_similarities[sorted_indices]
        
        # Visualize all tiles at VIT space if visualization is enabled
        if vit_tiles_data and getattr(self.config, 'enable_visualization', True):
            self._visualize_vit_space_tiled_grid(vit_tiles_data, grid_size, grid_size)
        
        # Return in the expected format: (s_uv_star, s_uv), sim_scores
        return (s_uv_star, s_uv), final_similarities
    
    def detect_features_roi(self):
        """Detect features using ROI-based approach with YOLOWorld."""
        if self.latest_image is None or self.yolo_detector is None or self.latest_pil_image is None:
            return None, None

        if self.vs_controller.iteration_count == 0:
            print(f"[VS-DEBUG] detect_features_roi: calling detector.detect() "
                  f"(state={getattr(self.yolo_detector, 'state', '?')}, "
                  f"keyword='{self.yolo_keyword}')...")

        # Profile YOLO detection
        yolo_start = time.perf_counter()
        current_detection = self.yolo_detector.detect(
            self.latest_pil_image,
            self.yolo_keyword,
            self.config.roi_padding_ratio
        )
        yolo_time = (time.perf_counter() - yolo_start) * 1000
        if self.vs_controller.iteration_count == 0:
            print(f"[VS-DEBUG] detect_features_roi: detection done in {yolo_time:.1f}ms, "
                  f"bbox={current_detection.get('bbox')}")
        self.node.get_logger().debug(f"[PERF] YOLO detection: {yolo_time:.2f}ms")
        
        # Update or use cached ROI
        if current_detection['bbox'] is not None:
            # Valid detection - update cache
            self.current_roi_bbox = current_detection['padded_bbox']
            self.last_valid_roi_iteration = self.vs_controller.iteration_count
            current_roi_bbox = self.current_roi_bbox
        else:
            # No detection - use cached ROI if available
            if self.current_roi_bbox is not None:
                current_roi_bbox = self.current_roi_bbox
                self.node.get_logger().warn(f"No ROI detected, using cached ROI from iteration {self.last_valid_roi_iteration}", throttle_duration_sec=5.0)
            else:
                # No cached ROI available, fall back to full image
                self.node.get_logger().warn("No ROI detected and no cached ROI available, using full image", throttle_duration_sec=5.0)
                return self.detect_features_original()
        
        # Use cached goal ROI features
        goal_roi_bbox = self.goal_roi_bbox if self.goal_roi_bbox is not None else None
        
        # Get cached goal features and image
        if goal_roi_bbox is None:
            # No goal ROI detected - use full image features
            desc1 = self.goal_features
            goal_roi_resized = self.goal_image_resized
            roi_mask_path = self.config.mask_path if hasattr(self.config, 'mask_path') else None
            self.node.get_logger().warn("No goal ROI detected, using full goal image features")
        else:
            desc1 = self.goal_roi_features
            goal_roi_resized = self.goal_roi_resized
            roi_mask_path = self.goal_roi_mask_path
        
        # Only need to extract current image features
        current_roi = self.yolo_detector.crop_roi(self.latest_pil_image, current_roi_bbox)
        # Store original size for proper coordinate mapping
        current_roi_original_size = current_roi.size

        # Profile current image processing
        current_process_start = time.perf_counter()

        # Resize current ROI to VIT input size (forces square, causing aspect ratio distortion)
        current_roi_resized = current_roi.resize((self.config.vit_input_size, self.config.vit_input_size))
        
        with torch.no_grad():
            # Extract features from current ROI only
            current_tensor = self.feature_extractor.preprocess_pil(current_roi_resized)
            
            desc2 = self.feature_extractor.extract_descriptors(
                current_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )

        current_process_time = (time.perf_counter() - current_process_start) * 1000
        self.node.get_logger().debug(f"[PERF] Current image processing: {current_process_time:.2f}ms (goal cached)")
        

        # Find correspondences
        self.node.get_logger().debug(f"Calling find_correspondences_batch with mask_path: {roi_mask_path}")
        points1, points2, sim_selected_12 = find_correspondences_batch(
            desc1, desc2,
            num_pairs=self.config.num_pairs,
            mask_path=roi_mask_path,
            vit_input_size=self.config.vit_input_size,
            patch_size=self.feature_extractor.get_patch_size()
        )
        
        # ROI mask is cached, no need to clean up
        
        if points1 is None or points2 is None:
            self.feature_failure_count += 1
            if self.feature_failure_count >= 10:
                self.node.get_logger().error("Feature detection failed 10 times in a row")
                raise RuntimeError("Persistent feature detection failure")
            return None, None
            
        # Reset counter on successful detection
        self.feature_failure_count = 0
        
        # Scale points from patch space to VIT input size
        # Get actual patch size from the model for correct coordinate scaling
        patch_size = self.feature_extractor.get_patch_size()
        num_patches_total = desc1.size(-2)
        num_patches_per_side = int(np.sqrt(num_patches_total))

        # Use the same calculation as the working old implementation
        scale = self.config.vit_input_size / num_patches_per_side


        points1_vit = points1 * scale + patch_size / 2
        points2_vit = points2 * scale + patch_size / 2

        
        # Visualize correspondences in VIT input space
        if getattr(self.config, 'enable_visualization', True):
            self._visualize_vit_space_correspondences(
                goal_roi_resized, current_roi_resized,
                points1_vit, points2_vit
            )

        # For visualization only - map to full image coordinates
        # ROIs are resized to square but we need to use ORIGINAL dimensions for proper mapping


        if goal_roi_bbox is not None:
            # Use the simple mapping function that assumes square resizing
            # This is correct because ViT always processes 224x224 images
            points1_full_viz = self._map_roi_to_full_image(points1_vit, goal_roi_bbox, None)
        else:
            # Goal points are already in full image coordinates
            points1_full_viz = points1_vit
        # For current ROI, also use the simple mapping
        points2_full_viz = self._map_roi_to_full_image(points2_vit, current_roi_bbox, None)

        
        # Visualize correspondences with ROI boxes
        # Points from regular ROI detection are in (x,y) format
        if getattr(self.config, 'enable_visualization', True):
            # DEBUG: Check if goal and current images have different dimensions
            if self.goal_image.size != self.latest_pil_image.size:
                self.node.get_logger().warn(f"[VIZ DIMENSION ISSUE] Goal image: {self.goal_image.size}, Current image: {self.latest_pil_image.size}")
                self.node.get_logger().warn(f"[VIZ DIMENSION ISSUE] Goal ROI bbox: {goal_roi_bbox}")
                self.node.get_logger().warn(f"[VIZ DIMENSION ISSUE] Current ROI bbox: {current_roi_bbox}")

                # If dimensions mismatch, we need to scale the goal ROI bbox to match current image dimensions
                if goal_roi_bbox is not None:
                    goal_width, goal_height = self.goal_image.size
                    current_width, current_height = self.latest_pil_image.size

                    # Scale goal ROI bbox if widths are different
                    if goal_width != current_width:
                        scale_factor = current_width / goal_width
                        scaled_goal_roi_bbox = [
                            goal_roi_bbox[0] * scale_factor,  # x1
                            goal_roi_bbox[1],  # y1 (height is same)
                            goal_roi_bbox[2] * scale_factor,  # x2
                            goal_roi_bbox[3]   # y2 (height is same)
                        ]
                        self.node.get_logger().info(f"[VIZ FIX] Scaling goal ROI bbox by {scale_factor:.3f}")
                        self.node.get_logger().info(f"[VIZ FIX] Original goal ROI: {goal_roi_bbox}")
                        self.node.get_logger().info(f"[VIZ FIX] Scaled goal ROI: {scaled_goal_roi_bbox}")

                        # Also need to scale the goal points
                        points1_full_viz_scaled = points1_full_viz.clone() if torch.is_tensor(points1_full_viz) else points1_full_viz.copy()
                        if torch.is_tensor(points1_full_viz_scaled):
                            points1_full_viz_scaled[:, 0] *= scale_factor
                        else:
                            points1_full_viz_scaled[:, 0] *= scale_factor

                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            points1_full_viz_scaled, points2_full_viz,
                            scaled_goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                    else:
                        # Heights different but widths same - just use original
                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            points1_full_viz, points2_full_viz,
                            goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                else:
                    # No goal ROI bbox to scale
                    self._visualize_roi_correspondences(
                        self.goal_image, self.latest_pil_image,
                        points1_full_viz, points2_full_viz,
                        goal_roi_bbox, current_roi_bbox,
                        points_are_yx=False
                    )
            else:
                # Dimensions match - use original visualization
                self._visualize_roi_correspondences(
                    self.goal_image, self.latest_pil_image,
                    points1_full_viz, points2_full_viz,
                    goal_roi_bbox, current_roi_bbox,
                    points_are_yx=False
                )
        
        # Map points from VIT space to full image coordinates for servoing
        if goal_roi_bbox is not None:
            points1_full = self._map_roi_to_full_image(points1_vit, goal_roi_bbox, None)
        else:
            # Goal points are already in full image coordinates when no ROI
            # Scale from VIT input size to full image size
            scale_x = self.config.u_max / self.config.vit_input_size
            scale_y = self.config.v_max / self.config.vit_input_size
            points1_full = points1_vit.copy() if isinstance(points1_vit, np.ndarray) else points1_vit.cpu().numpy()
            points1_full[:, 0] *= scale_x
            points1_full[:, 1] *= scale_y
        # For current ROI servoing, use the simple mapping
        points2_full = self._map_roi_to_full_image(points2_vit, current_roi_bbox, None)
        
        
        # Use custom calculate_uv for ROI mode that doesn't double-scale
        return self.calculate_uv_roi(points1_full.tolist(), points2_full.tolist()), sim_selected_12
    
    def detect_features_roi_tiled(self):
        """Detect features using ROI-based tiling approach."""
        if self.latest_image is None or self.yolo_detector is None:
            return None, None
            
        # Profile YOLO detection
        yolo_start = time.perf_counter()
        current_detection = self.yolo_detector.detect(
            self.latest_pil_image,
            self.yolo_keyword,
            self.config.roi_padding_ratio
        )
        yolo_time = (time.perf_counter() - yolo_start) * 1000
        self.node.get_logger().debug(f"[PERF] YOLO detection (ROI tiled): {yolo_time:.2f}ms")
        
        # Update or use cached ROI
        if current_detection['bbox'] is not None:
            # Valid detection - update cache
            self.current_roi_bbox = current_detection['padded_bbox']
            self.last_valid_roi_iteration = self.vs_controller.iteration_count
            current_roi_bbox = self.current_roi_bbox
        else:
            # No detection - use cached ROI if available
            if self.current_roi_bbox is not None:
                current_roi_bbox = self.current_roi_bbox
                self.node.get_logger().warn(f"[Tiled] No ROI detected, using cached ROI from iteration {self.last_valid_roi_iteration}", throttle_duration_sec=5.0)
            else:
                # No cached ROI available, fall back to full image with tiling
                self.node.get_logger().warn("[Tiled] No ROI detected and no cached ROI available, using full image with tiling", throttle_duration_sec=5.0)
                return self.detect_features_tiled()
        
        # Use cached goal ROI or fail for tiling mode
        goal_roi_bbox = self.goal_roi_bbox if self.goal_roi_bbox is not None else None
        
        # ROI tiling REQUIRES both bounding boxes
        if goal_roi_bbox is None:
            self.node.get_logger().error("[ROI Tiled] FATAL: No goal ROI detected - ROI tiling mode requires object detection in goal image!")
            self.node.get_logger().error(f"[ROI Tiled] Failed to detect '{self.yolo_keyword}' in goal image. Cannot use ROI tiling mode.")
            # Try to fall back to regular ROI mode
            self.node.get_logger().warn("[ROI Tiled] Attempting fallback to regular ROI mode...")
            return self.detect_features_roi()
        
        # Crop ROIs
        goal_roi = self.yolo_detector.crop_roi(self.goal_image, goal_roi_bbox)
        self.node.get_logger().debug(f"[ROI Tiled] Using goal ROI bbox: {goal_roi_bbox}")
            
        current_roi = self.yolo_detector.crop_roi(self.latest_pil_image, current_roi_bbox)
        
        # Create ROI tiling configuration
        # Use optimal tile number if calculated, otherwise use config value
        roi_tile_number = self.get_effective_tile_number()

        # Get rectangular grid dimensions (supports non-square grids like 4x1)
        if hasattr(self, 'optimal_grid_width') and hasattr(self, 'optimal_grid_height'):
            grid_width = self.optimal_grid_width
            grid_height = self.optimal_grid_height
        else:
            # Fallback to square grid
            grid_size = int(np.sqrt(roi_tile_number))
            grid_width = grid_height = grid_size

        # Only log the grid configuration once when tiling is first activated or grid changes
        if not hasattr(self, '_last_logged_grid') or self._last_logged_grid != (grid_width, grid_height):
            self.node.get_logger().debug(f"[ROI Tiled] Using {grid_width}x{grid_height} grid ({roi_tile_number} tiles)")
            self._last_logged_grid = (grid_width, grid_height)

        # Resize ROI to rectangular dimensions that match the grid
        # Calculate total size for each dimension
        total_width = self.config.vit_input_size * grid_width
        total_height = self.config.vit_input_size * grid_height
        tile_size = self.config.vit_input_size
        
        # Profile ROI resize for tiling
        roi_resize_start = time.perf_counter()
        goal_roi_resized = goal_roi.resize((total_width, total_height))
        current_roi_resized = current_roi.resize((total_width, total_height))
        roi_resize_time = (time.perf_counter() - roi_resize_start) * 1000
        self.node.get_logger().debug(f"[PERF] ROI resize to {total_width}x{total_height}: {roi_resize_time:.2f}ms")
        
        # Handle mask for ROI tiling mode
        mask_resized = None
        if self.config.mask_path:
            # Try to load object-specific mask first
            mask_path = self.config.mask_path
            if hasattr(self, 'yolo_keyword') and self.yolo_keyword:
                # Check if object-specific mask exists
                import os
                base_dir = os.path.dirname(mask_path)
                object_mask_path = os.path.join(base_dir, f"mask_{self.yolo_keyword}.jpg")
                if os.path.exists(object_mask_path):
                    mask_path = object_mask_path
            
            # Load mask
            mask_img = Image.open(mask_path).convert('L')
            
            if goal_roi_bbox is not None:
                # Crop mask to match goal ROI
                x1, y1, x2, y2 = [int(coord) for coord in goal_roi_bbox]
                mask_cropped = mask_img.crop((x1, y1, x2, y2))
            else:
                # No ROI, use full mask
                mask_cropped = mask_img
            
            # Resize mask to match the resized ROI
            mask_resized = mask_cropped.resize((total_width, total_height))
        
        
        all_points1 = []
        all_points2 = []
        all_sims = []
        
        # Store VIT space data for visualization
        vit_tiles_data = []
        
        # Profile tile processing
        tile_process_start = time.perf_counter()
        total_tiles = grid_width * grid_height
        self.node.get_logger().debug(f"[PERF] Processing {total_tiles} tiles ({grid_width}x{grid_height} grid) - BATCH MODE")
        
        # First, extract all tiles
        goal_tiles = []
        current_tiles = []
        mask_tiles = []
        tile_positions = []
        
        extract_tiles_start = time.perf_counter()
        # Process each tile in the rectangular grid
        for i in range(grid_height):  # Rows
            for j in range(grid_width):  # Columns
                tile_x = j * tile_size
                tile_y = i * tile_size
                
                # Extract tiles
                goal_tile = goal_roi_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                current_tile = current_roi_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                
                goal_tiles.append(goal_tile)
                current_tiles.append(current_tile)
                tile_positions.append((i, j))
                
                # Extract mask tile if available
                if mask_resized is not None:
                    mask_tile = mask_resized.crop((
                        tile_x, tile_y,
                        tile_x + tile_size, tile_y + tile_size
                    ))
                    mask_tiles.append(mask_tile)
        
        extract_tiles_time = (time.perf_counter() - extract_tiles_start) * 1000
        self.node.get_logger().debug(f"[PERF] Extract all tiles: {extract_tiles_time:.2f}ms")
        
        # Batch preprocess all tiles
        preprocess_start = time.perf_counter()
        goal_tensors = []
        current_tensors = []
        
        for goal_tile, current_tile in zip(goal_tiles, current_tiles):
            goal_tensor = self.feature_extractor.preprocess_pil(goal_tile)
            current_tensor = self.feature_extractor.preprocess_pil(current_tile)
            goal_tensors.append(goal_tensor)
            current_tensors.append(current_tensor)
        
        # Stack into batches
        goal_batch = torch.cat(goal_tensors, dim=0).to(self.feature_extractor.device)
        current_batch = torch.cat(current_tensors, dim=0).to(self.feature_extractor.device)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        self.node.get_logger().debug(f"[PERF] Preprocess all tiles: {preprocess_time:.2f}ms")
        
        # Batch extract features
        with torch.no_grad():
            extract_start = time.perf_counter()
            # Extract features for all tiles at once
            desc1_batch = self.feature_extractor.extract_descriptors(
                goal_batch,
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )
            desc2_batch = self.feature_extractor.extract_descriptors(
                current_batch,
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning,
                hierarchy=self.config.feature_hierarchy
            )
            extract_time = (time.perf_counter() - extract_start) * 1000
            self.node.get_logger().debug(f"[PERF] Batch feature extraction ({total_tiles} tiles): {extract_time:.2f}ms")
        
        # Pre-compute mask tensors for all tiles to avoid repeated computation
        mask_tensors = []
        if mask_tiles:
            from features.matcher import create_patch_mask_from_pil
            # Calculate actual patch size from descriptor dimensions
            actual_patch_size = self.config.vit_input_size // int(np.sqrt(desc1_batch[0].size(-2)))
            
            mask_compute_start = time.perf_counter()
            for mask_tile in mask_tiles:
                mask_tensor = create_patch_mask_from_pil(
                    mask_tile, 
                    vit_input_size=self.config.vit_input_size,
                    patch_size=actual_patch_size
                ).to(self.feature_extractor.device)
                mask_tensors.append(mask_tensor)
            mask_compute_time = (time.perf_counter() - mask_compute_start) * 1000
            self.node.get_logger().debug(f"[PERF] Mask tensor computation: {mask_compute_time:.2f}ms")
        
        # Process correspondences for each tile
        correspondence_start = time.perf_counter()
        for tile_idx in range(total_tiles):
            i, j = tile_positions[tile_idx]
            
            # Extract features for this tile from the batch
            desc1 = desc1_batch[tile_idx:tile_idx+1]
            desc2 = desc2_batch[tile_idx:tile_idx+1]
            
            # Use pre-computed mask tensor if available
            tile_mask_tensor = mask_tensors[tile_idx] if tile_idx < len(mask_tensors) else None
            
            # Find correspondences for this tile with mask
            # Use actual number of tiles (grid_width * grid_height) instead of configured tile number
            # since rectangular grids may have different tile counts than expected
            # Ensure at least 1 pair per tile
            pairs_per_tile = max(1, self.config.num_pairs // total_tiles)

            # Log which tile we're processing
            self.node.get_logger().debug(f"[ROI Tiled] Processing tile ({i},{j}) - requesting {pairs_per_tile} pairs")

            points1, points2, sim = find_correspondences_batch(
                desc1, desc2,
                num_pairs=pairs_per_tile,  # Distribute pairs across actual tile count
                mask_tensor=tile_mask_tensor,  # Use pre-computed mask tensor
                vit_input_size=self.config.vit_input_size,
                patch_size=self.feature_extractor.get_patch_size()
            )

            if points1 is not None and points2 is not None:
                self.node.get_logger().debug(f"[ROI Tiled] Tile ({i},{j}) SUCCESS: Got {len(points1)} correspondences")
                # Get tile position for offset calculation
                tile_x = j * tile_size
                tile_y = i * tile_size

                # Points from find_correspondences_batch are in [x, y] format
                # Scale points from patch space to VIT input size
                # Get actual patch size from the model for correct coordinate scaling
                patch_size = self.feature_extractor.get_patch_size()
                scale = self.config.vit_input_size / int(np.sqrt(desc1.size(-2)))
                points1_vit = points1 * scale + patch_size / 2
                points2_vit = points2 * scale + patch_size / 2

                
                # Store for VIT space visualization
                vit_tiles_data.append({
                    'tile_idx': (i, j),
                    'goal_tile': goal_tiles[tile_idx],
                    'current_tile': current_tiles[tile_idx],
                    'points1_vit': points1_vit.clone() if torch.is_tensor(points1_vit) else points1_vit.copy(),
                    'points2_vit': points2_vit.clone() if torch.is_tensor(points2_vit) else points2_vit.copy()
                })
                
                # Map from tile VIT space to resized ROI coordinates
                # Since tiles are already at VIT input size, no scaling needed for tile coordinates
                # Just add the tile offset within the resized ROI
                points1_resized_roi = torch.zeros_like(points1_vit)
                points1_resized_roi[:, 0] = tile_x + points1_vit[:, 0]  # x
                points1_resized_roi[:, 1] = tile_y + points1_vit[:, 1]  # y

                points2_resized_roi = torch.zeros_like(points2_vit)
                points2_resized_roi[:, 0] = tile_x + points2_vit[:, 0]  # x
                points2_resized_roi[:, 1] = tile_y + points2_vit[:, 1]  # y

                
                # Step 3: Map from resized ROI coordinates to original ROI coordinates
                # Then to full image coordinates
                if goal_roi_bbox is not None:
                    goal_x1, goal_y1, goal_x2, goal_y2 = goal_roi_bbox
                    goal_roi_width = goal_x2 - goal_x1
                    goal_roi_height = goal_y2 - goal_y1
                    
                    # Scale from resized ROI (total_width x total_height) to original ROI size
                    scale_x_goal = goal_roi_width / total_width
                    scale_y_goal = goal_roi_height / total_height
                    
                    points1_full = torch.zeros_like(points1_vit)
                    points1_full[:, 0] = goal_x1 + points1_resized_roi[:, 0] * scale_x_goal
                    points1_full[:, 1] = goal_y1 + points1_resized_roi[:, 1] * scale_y_goal
                else:
                    # If no ROI, scale directly to full image dimensions
                    points1_full = torch.zeros_like(points1_vit)
                    points1_full[:, 0] = points1_resized_roi[:, 0] * (self.config.u_max / total_width)
                    points1_full[:, 1] = points1_resized_roi[:, 1] * (self.config.v_max / total_height)
                    
                    # Debug logging for no goal bbox case
                    if tile_idx == 0:  # Only log once
                        self.node.get_logger().warn(f"[ROI Tiled] No goal bbox - scaling from {total_width}x{total_height} to {self.config.u_max}x{self.config.v_max}")
                    
                current_x1, current_y1, current_x2, current_y2 = current_roi_bbox
                current_roi_width = current_x2 - current_x1
                current_roi_height = current_y2 - current_y1
                
                # Scale from resized ROI (total_width x total_height) to original ROI size
                scale_x_current = current_roi_width / total_width
                scale_y_current = current_roi_height / total_height
                
                points2_full = torch.zeros_like(points2_vit)
                points2_full[:, 0] = current_x1 + points2_resized_roi[:, 0] * scale_x_current
                points2_full[:, 1] = current_y1 + points2_resized_roi[:, 1] * scale_y_current
                
                all_points1.append(points1_full)
                all_points2.append(points2_full)
                all_sims.append(sim)
            else:
                # Log which tile failed to produce correspondences
                self.node.get_logger().error(f"[ROI Tiled] ERROR: Tile ({i},{j}) FAILED - no correspondences returned! (requested {pairs_per_tile} pairs)")
                self.node.get_logger().error(f"[ROI Tiled] This likely means no valid correspondences passed the distance threshold and mask filters. Enable debug logging to see details.")

        # Log correspondence processing time
        correspondence_time = (time.perf_counter() - correspondence_start) * 1000
        self.node.get_logger().debug(f"[PERF] Correspondence processing: {correspondence_time:.2f}ms")
        
        # Total tile processing time
        total_tile_time = (time.perf_counter() - tile_process_start) * 1000
        self.node.get_logger().debug(f"[PERF] Total ROI tile processing: {total_tile_time:.2f}ms")
        
        # Combine all points from all tiles
        if len(all_points1) > 0:
            combined_points1 = torch.cat(all_points1, dim=0)
            combined_points2 = torch.cat(all_points2, dim=0)
            combined_sims = torch.cat(all_sims, dim=0)

            # Log if we got fewer correspondences than expected
            if len(combined_points1) < self.config.num_pairs:
                self.node.get_logger().debug(f"[ROI Tiled] Found {len(combined_points1)} correspondences from {total_tiles} tiles (requested {self.config.num_pairs})")

            # Visualize all tiles at VIT space
            if vit_tiles_data and getattr(self.config, 'enable_visualization', True):
                # Pass both dimensions to visualization function
                self._visualize_vit_space_tiled_grid(vit_tiles_data, grid_width, grid_height)


            # Select top matches based on similarity
            num_pairs = min(self.config.num_pairs, len(combined_points1))
            if len(combined_points1) > num_pairs:
                top_indices = torch.topk(combined_sims, num_pairs).indices
                final_points1 = combined_points1[top_indices]
                final_points2 = combined_points2[top_indices]
                final_sims = combined_sims[top_indices]
            else:
                final_points1 = combined_points1
                final_points2 = combined_points2
                final_sims = combined_sims
            
            
            # Visualize correspondences with ROI boxes
            # Points from ROI tiling are in (x,y) format, need to tell visualization function
            if getattr(self.config, 'enable_visualization', True):
                # Check if goal and current images have different dimensions (same fix as regular ROI)
                if self.goal_image.size != self.latest_pil_image.size:
                    goal_width, goal_height = self.goal_image.size
                    current_width, current_height = self.latest_pil_image.size

                    if goal_width != current_width and goal_roi_bbox is not None:
                        scale_factor = current_width / goal_width
                        scaled_goal_roi_bbox = [
                            goal_roi_bbox[0] * scale_factor,
                            goal_roi_bbox[1],
                            goal_roi_bbox[2] * scale_factor,
                            goal_roi_bbox[3]
                        ]
                        # Scale goal points too
                        final_points1_scaled = final_points1.clone() if torch.is_tensor(final_points1) else final_points1.copy()
                        if torch.is_tensor(final_points1_scaled):
                            final_points1_scaled[:, 0] *= scale_factor
                        else:
                            final_points1_scaled[:, 0] *= scale_factor

                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            final_points1_scaled, final_points2,
                            scaled_goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                    else:
                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            final_points1, final_points2,
                            goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                else:
                    # Dimensions match - use original
                    self._visualize_roi_correspondences(
                        self.goal_image, self.latest_pil_image,
                        final_points1, final_points2,
                        goal_roi_bbox, current_roi_bbox,
                        points_are_yx=False  # ROI tiling coordinates are in (x,y) format
                    )

            # Store actual number of features found
            self.last_num_features = len(final_points1)

            return self.calculate_uv_roi(final_points1.tolist(), final_points2.tolist()), final_sims
        else:
            self.node.get_logger().warn("No correspondences found in any tile")
            self.last_num_features = 0
            return None, None
    
    def detect_features_classical(self):
        """Detect features using classical methods (SIFT, ORB, AKAZE)."""
        if self.latest_image is None:
            return None, None
        
        try:
            # Get the classical extractor from the multi-backbone extractor
            if hasattr(self.feature_extractor.extractor, 'classical_extractor'):
                classical_extractor = self.feature_extractor.extractor.classical_extractor
            else:
                self.node.get_logger().error("Classical extractor not found in feature extractor")
                return None, None
            
            # Use the classical detect_features method
            points1, points2, similarities = classical_extractor.detect_features_classical(
                self.goal_image,
                self.latest_pil_image,
                num_pairs=self.config.num_pairs
            )
            
            if points1 is None or points2 is None:
                return None, None
            
            # Store actual number of features found
            self.last_num_features = len(points1)
            
            # Visualize correspondences if enabled
            if getattr(self.config, 'enable_visualization', True):
                # Convert points from (x,y) to (y,x) for visualization
                points1_viz = np.flip(points1, axis=1)
                points2_viz = np.flip(points2, axis=1)
                
                self._viz_correspondences(
                    self.goal_image, self.latest_pil_image,
                    points1_viz, points2_viz,
                    None, self.bridge, self.correspondence_pub,
                    self.goal_image_pub, self.current_image_pub,
                    False, None
                )
            
            # CRITICAL FIX: Classical methods return pixel coordinates directly
            # Convert to integer UV coordinates as expected by the visual servoing pipeline
            s_uv_star = np.round(points1).astype(int)  # Goal features in pixel coords
            s_uv = np.round(points2).astype(int)       # Current features in pixel coords
            
            # Return in the expected format: ((goal_points, current_points), similarities)
            return (s_uv_star, s_uv), similarities
            
        except Exception as e:
            self.node.get_logger().error(f"Error in classical feature detection: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def detect_features_classical_roi(self):
        """
        Classical feature detection within ROI regions.
        Uses same ROI from LangSAM/YOLO as ViT methods (MODE 3/4).

        Classical methods (SIFT/ORB/AKAZE) detect features in cropped ROI images,
        then coordinates are mapped back to full image space.
        """
        if self.latest_image is None:
            return None, None

        try:
            # Get the classical extractor
            if hasattr(self.feature_extractor.extractor, 'classical_extractor'):
                classical_extractor = self.feature_extractor.extractor.classical_extractor
            else:
                self.node.get_logger().error("Classical extractor not found in feature extractor")
                return None, None

            # 1. Get current ROI using the detector
            current_detection = self.yolo_detector.detect(
                self.latest_pil_image,
                self.yolo_keyword,
                self.config.roi_padding_ratio
            )

            # Update or use cached current ROI
            if current_detection['bbox'] is not None:
                self.current_roi_bbox = current_detection['padded_bbox']
                self.last_valid_roi_iteration = self.vs_controller.iteration_count
                current_bbox = self.current_roi_bbox
            else:
                # No detection - use cached ROI if available
                if self.current_roi_bbox is not None:
                    current_bbox = self.current_roi_bbox
                    self.node.get_logger().warn("[Classical ROI] No detection, using cached ROI", throttle_duration_sec=5.0)
                else:
                    # No cached ROI available, fall back to full image
                    self.node.get_logger().warn("[Classical ROI] No ROI detected, falling back to full image", throttle_duration_sec=5.0)
                    return self.detect_features_classical()

            # 2. Get goal ROI (use cached from initialization)
            if self.goal_roi_bbox is None:
                # Detect goal ROI if not cached
                goal_detection = self.yolo_detector.detect(
                    self.goal_image,
                    self.yolo_keyword,
                    self.config.roi_padding_ratio
                )
                if goal_detection['bbox'] is not None:
                    self.goal_roi_bbox = goal_detection['padded_bbox']
                    self.node.get_logger().info(f"[Classical ROI] Detected goal ROI: {self.goal_roi_bbox}")
                else:
                    self.node.get_logger().warn("[Classical ROI] No goal ROI detected, falling back to full image")
                    return self.detect_features_classical()

            goal_bbox = self.goal_roi_bbox

            # 3. Crop ROIs
            goal_roi = self.yolo_detector.crop_roi(self.goal_image, goal_bbox)
            current_roi = self.yolo_detector.crop_roi(self.latest_pil_image, current_bbox)

            # 4. Run classical feature detection on ROIs
            goal_pts, curr_pts, sims = classical_extractor.detect_features_classical_roi(
                goal_roi, current_roi,
                goal_bbox, current_bbox,
                num_pairs=self.config.num_pairs
            )

            if goal_pts is None or curr_pts is None:
                return None, None

            # Store actual number of features found
            self.last_num_features = len(goal_pts)

            # 5. Visualize correspondences if enabled
            if getattr(self.config, 'enable_visualization', True):
                # Pass points in (x,y) format - visualization function handles flip
                # NOTE: Do NOT flip here - _visualize_roi_correspondences handles the
                # coordinate conversion from (x,y) to (y,x) when points_are_yx=False (default)
                self._visualize_roi_correspondences(
                    self.goal_image, self.latest_pil_image,
                    goal_pts, curr_pts,  # Raw (x,y) format
                    goal_bbox, current_bbox
                )

            # 6. Return in the expected format: ((goal_points, current_points), similarities)
            # Convert to integer UV coordinates as expected by the visual servoing pipeline
            s_uv_star = np.round(goal_pts).astype(int)  # Goal features in pixel coords
            s_uv = np.round(curr_pts).astype(int)       # Current features in pixel coords

            # Reset failure count on successful detection
            self.consecutive_no_features_count = 0

            return (s_uv_star, s_uv), sims

        except Exception as e:
            self.node.get_logger().error(f"Error in classical ROI feature detection: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _map_roi_to_full_image(self, points_roi, roi_bbox, roi_size):
        """Map points from ROI coordinates to full image coordinates.

        This handles the case where ROI was resized to square (224x224) for ViT processing.
        Points are in VIT space (224x224) and need to be mapped back to original ROI dimensions.
        """
        if roi_bbox is None:
            # No ROI, points are already in full image space (scaled)
            scale_x = self.config.u_max / self.config.vit_input_size
            scale_y = self.config.v_max / self.config.vit_input_size
            points_full = points_roi.clone()
            points_full[:, 0] *= scale_x
            points_full[:, 1] *= scale_y
            return points_full

        # ROI dimensions in the original image
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1

        # Points are in VIT space (224x224), need to map to original ROI dimensions
        # The ROI was resized from (roi_width x roi_height) to (224 x 224)
        # So we need to scale back by the inverse of that transformation
        scale_x = roi_width / self.config.vit_input_size
        scale_y = roi_height / self.config.vit_input_size

        # Map to full image coordinates
        points_full = torch.zeros_like(points_roi)
        points_full[:, 0] = roi_x1 + (points_roi[:, 0] * scale_x)
        points_full[:, 1] = roi_y1 + (points_roi[:, 1] * scale_y)

        return points_full

    def _map_roi_to_full_image_with_actual_size(self, points_roi, roi_bbox, roi_size):
        """Map points from ROI coordinates to full image coordinates using actual resized dimensions.

        This version properly handles aspect ratio distortion by using the actual resized ROI dimensions
        instead of assuming square (vit_input_size x vit_input_size) resizing.

        Args:
            points_roi: Points in ROI coordinate space (from resized ROI)
            roi_bbox: Original ROI bounding box (x1, y1, x2, y2) in full image coordinates
            roi_size: Actual size of the resized ROI as (width, height)
        """
        if roi_bbox is None:
            # No ROI, points are already in full image space - scale from resized to full
            roi_resized_width, roi_resized_height = roi_size
            scale_x = self.config.u_max / roi_resized_width
            scale_y = self.config.v_max / roi_resized_height
            points_full = points_roi.clone()
            points_full[:, 0] *= scale_x
            points_full[:, 1] *= scale_y
            return points_full

        # ROI dimensions in original image
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1

        # Use ACTUAL resized ROI dimensions to handle aspect ratio distortion properly
        roi_resized_width, roi_resized_height = roi_size

        # Scale from resized ROI space to original ROI space
        scale_x = roi_width / roi_resized_width
        scale_y = roi_height / roi_resized_height


        # Map to full image coordinates
        points_full = torch.zeros_like(points_roi)
        points_full[:, 0] = roi_x1 + (points_roi[:, 0] * scale_x)
        points_full[:, 1] = roi_y1 + (points_roi[:, 1] * scale_y)

        return points_full

    def _visualize_vit_space_tiled_grid(self, vit_tiles_data, grid_width, grid_height):
        """Visualize all tiles in a rectangular grid at VIT space level.

        Args:
            vit_tiles_data: List of dictionaries with tile data
            grid_width: Width of the grid (number of columns)
            grid_height: Height of the grid (number of rows)
        """
        # Early return if visualization is disabled
        if not getattr(self.config, 'enable_visualization', True):
            return
            
        # Create a grid image showing all tiles
        tile_size = self.config.vit_input_size
        grid_img_width = tile_size * grid_width + (grid_width - 1) * 10  # 10 pixel spacing
        grid_img_height = tile_size * grid_height + (grid_height - 1) * 10
        
        # Create blank canvases for goal and current
        goal_grid = np.ones((grid_img_height, grid_img_width, 3), dtype=np.uint8) * 255
        current_grid = np.ones((grid_img_height, grid_img_width, 3), dtype=np.uint8) * 255
        
        # Collect all points for visualization
        all_goal_points = []
        all_current_points = []
        
        for tile_data in vit_tiles_data:
            i, j = tile_data['tile_idx']
            goal_tile = np.array(tile_data['goal_tile'])
            current_tile = np.array(tile_data['current_tile'])
            points1_vit = tile_data['points1_vit']
            points2_vit = tile_data['points2_vit']
            
            # Calculate position in grid
            y_offset = i * (tile_size + 10)
            x_offset = j * (tile_size + 10)
            
            # Place tiles in grid
            goal_grid[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = goal_tile
            current_grid[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = current_tile
            
            # Transform points to grid coordinates
            if torch.is_tensor(points1_vit):
                points1_np = points1_vit.cpu().numpy()
                points2_np = points2_vit.cpu().numpy()
            else:
                points1_np = np.array(points1_vit)
                points2_np = np.array(points2_vit)
            
            # Add offset to points (they're in [x,y] format)
            points1_grid = points1_np.copy()
            points1_grid[:, 0] += x_offset  # x
            points1_grid[:, 1] += y_offset  # y
            
            points2_grid = points2_np.copy()
            points2_grid[:, 0] += x_offset  # x
            points2_grid[:, 1] += y_offset  # y
            
            all_goal_points.append(points1_grid)
            all_current_points.append(points2_grid)
            
        
        # Combine all points
        if all_goal_points:
            combined_goal_points = np.vstack(all_goal_points)
            combined_current_points = np.vstack(all_current_points)
            
            # Convert to [y,x] for visualization
            combined_goal_points_viz = np.flip(combined_goal_points, axis=1)
            combined_current_points_viz = np.flip(combined_current_points, axis=1)
            
            # Convert grids to PIL images
            goal_grid_pil = Image.fromarray(goal_grid)
            current_grid_pil = Image.fromarray(current_grid)
            
            # Fixed output dimensions for consistent video recording
            FIXED_WIDTH = 1200
            FIXED_HEIGHT = 600
            FIXED_DPI = 100

            # Create visualization figure with fixed size
            fig = plt.figure(figsize=(FIXED_WIDTH / FIXED_DPI, FIXED_HEIGHT / FIXED_DPI), dpi=FIXED_DPI)

            # Use fixed subplot positions instead of tight_layout (prevents jiggle)
            ax1 = fig.add_axes([0.02, 0.02, 0.46, 0.96])  # [left, bottom, width, height]
            ax2 = fig.add_axes([0.52, 0.02, 0.46, 0.96])

            ax1.imshow(goal_grid_pil)
            ax2.imshow(current_grid_pil)

            # Draw correspondences
            colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(combined_goal_points_viz)))

            for i, ((y1, x1), (y2, x2), color) in enumerate(zip(combined_goal_points_viz, combined_current_points_viz, colors)):
                ax1.plot(x1, y1, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=0.5)

                ax2.plot(x2, y2, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=0.5)

                # Draw lines between correspondences
                con = ConnectionPatch(
                    xyA=(x1, y1), xyB=(x2, y2),
                    coordsA="data", coordsB="data",
                    axesA=ax1, axesB=ax2, color=color, alpha=0.35
                )
                fig.add_artist(con)

            # Draw grid lines to show tile boundaries
            # Vertical lines (for grid_width)
            for j in range(1, grid_width):
                line_pos = j * (tile_size + 10) - 5
                ax1.axvline(x=line_pos, color='gray', linestyle='--', alpha=0.5)
                ax2.axvline(x=line_pos, color='gray', linestyle='--', alpha=0.5)
            # Horizontal lines (for grid_height)
            for i in range(1, grid_height):
                line_pos = i * (tile_size + 10) - 5
                ax1.axhline(y=line_pos, color='gray', linestyle='--', alpha=0.5)
                ax2.axhline(y=line_pos, color='gray', linestyle='--', alpha=0.5)

            ax1.axis('off')
            ax2.axis('off')

            # NO tight_layout() - we use fixed axes positions for consistent output size

            # Convert to ROS image with fixed dimensions
            fig.canvas.draw()
            img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_data = img_data.reshape((FIXED_HEIGHT, FIXED_WIDTH, 4))[:, :, :3]  # Fixed dimensions, drop alpha

            # Convert RGB to BGR for ROS
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            ros_image = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            self.vit_space_tiled_pub.publish(ros_image)

            plt.close(fig)
    
    def _visualize_roi_correspondences(self, goal_image, current_image, 
                                       points1, points2, goal_roi_bbox, current_roi_bbox,
                                       points_are_yx=False):
        """Visualize correspondences with ROI bounding boxes.
        
        Args:
            points_are_yx: If True, points are already in (y,x) format. If False, they are in (x,y) format.
        """
        # Early return if visualization is disabled
        if not getattr(self.config, 'enable_visualization', True):
            return
            
        # Convert images to numpy
        goal_np = np.array(goal_image)
        current_np = np.array(current_image)
        
        # Draw ROI boxes
        if goal_roi_bbox is not None:
            x1, y1, x2, y2 = [int(x) for x in goal_roi_bbox]
            cv2.rectangle(goal_np, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if current_roi_bbox is not None:
            x1, y1, x2, y2 = [int(x) for x in current_roi_bbox]
            cv2.rectangle(current_np, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Convert back to PIL for visualization
        goal_pil = Image.fromarray(goal_np)
        current_pil = Image.fromarray(current_np)
        
        
        # Handle different point formats
        if points_are_yx:
            # Points are already in (y, x) format from ROI tiling
            points1_viz = points1
            points2_viz = points2
        else:
            # Points are in (x, y) format from regular ROI detection - need to flip
            if torch.is_tensor(points1):
                points1_viz = torch.flip(points1, dims=[1])  # Flip from (x,y) to (y,x)
                points2_viz = torch.flip(points2, dims=[1])
            else:
                points1_viz = np.flip(np.array(points1), axis=1)
                points2_viz = np.flip(np.array(points2), axis=1)
        
        
        # Use existing visualization with flipped coordinates (only if visualization is enabled)
        # For ROI mode, we don't want to show tiling grid lines
        if self.config.enable_visualization:
            self._viz_correspondences(
                goal_pil, current_pil,
                points1_viz, points2_viz,
                None, self.bridge, self.correspondence_pub,
                self.goal_image_pub, self.current_image_pub,
                False, None  # Don't draw grid lines for ROI visualization
            )
    
    def _visualize_vit_space_correspondences(self, goal_roi_resized, current_roi_resized,
                                              points1_vit, points2_vit):
        """Visualize correspondences in VIT input space for debugging."""
        # Early return if visualization is disabled
        if not getattr(self.config, 'enable_visualization', True):
            return

        # Create a new publisher for VIT space visualization if it doesn't exist
        if not hasattr(self, 'vit_space_pub'):
            self.vit_space_pub = self.node.create_publisher(ImageMsg, '/vs/vit_space_correspondences', 1)

        # Fixed output dimensions for consistent video recording
        FIXED_WIDTH = 1000
        FIXED_HEIGHT = 500
        FIXED_DPI = 100

        # Create figure with fixed size and DPI
        fig = plt.figure(figsize=(FIXED_WIDTH / FIXED_DPI, FIXED_HEIGHT / FIXED_DPI), dpi=FIXED_DPI)

        # Use fixed subplot positions instead of tight_layout (prevents jiggle)
        ax1 = fig.add_axes([0.02, 0.02, 0.46, 0.96])  # [left, bottom, width, height]
        ax2 = fig.add_axes([0.52, 0.02, 0.46, 0.96])

        # Display the resized ROIs
        ax1.imshow(goal_roi_resized)
        ax2.imshow(current_roi_resized)

        # Convert points to numpy if they're tensors
        points1_np = points1_vit.cpu().numpy() if torch.is_tensor(points1_vit) else np.array(points1_vit)
        points2_np = points2_vit.cpu().numpy() if torch.is_tensor(points2_vit) else np.array(points2_vit)

        # Generate colors
        colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(points1_np)))

        # Plot points in VIT space (expecting x,y order)
        for i, ((x1, y1), (x2, y2), color) in enumerate(zip(points1_np, points2_np, colors)):
            # Plot points
            ax1.plot(x1, y1, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

            ax2.plot(x2, y2, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

            # Draw lines connecting corresponding points
            con = ConnectionPatch(
                xyA=(x1, y1), xyB=(x2, y2),
                coordsA="data", coordsB="data",
                axesA=ax1, axesB=ax2, color=color, alpha=0.35, linewidth=2
            )
            fig.add_artist(con)

        # Set axis limits and remove scales/ticks
        for ax in [ax1, ax2]:
            ax.set_xlim(0, self.config.vit_input_size)
            ax.set_ylim(self.config.vit_input_size, 0)  # Flip y-axis to match image coordinates
            ax.axis('off')  # Remove scales, ticks, and borders

        # NO tight_layout() - we use fixed axes positions for consistent output size

        # Convert to ROS image with fixed dimensions
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape((FIXED_HEIGHT, FIXED_WIDTH, 4))[:, :, :3]  # Fixed dimensions, drop alpha

        # Convert RGB to BGR for ROS
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        ros_image = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        self.vit_space_pub.publish(ros_image)

        plt.close(fig)
    
    def ibvs(self):
        """Image Based Visual Servoing Method."""
        if self.latest_image is None:
            print("NO LATEST IMAGE SKIPPING")
            return

        start_time = time.time()

        # Get features with error handling
        _t_feat = time.perf_counter()
        result = self.detect_features()
        self._timing_detect = time.perf_counter() - _t_feat

        if result is None or result[0] is None:
            self.node.get_logger().warn("Feature detection failed - skipping this iteration")
            return

        (s_uv_star, s_uv), sim_selected_12 = result
        if s_uv_star is None or s_uv is None or len(s_uv_star) < 4:
            self.node.get_logger().warn("Insufficient features detected - skipping this iteration")
            return

        # Calculate feature error in pixel space for real-world switching
        # This is the average pixel distance between current and goal features
        feature_errors = np.linalg.norm(s_uv - s_uv_star, axis=1)  # Pixel distances for each feature
        self.current_feature_error = np.mean(feature_errors)  # Average pixel error

        # Store initial feature error on first iteration (after first successful feature detection)
        if self.initial_feature_error is None and self.vs_controller.iteration_count <= 1:
            self.initial_feature_error = self.current_feature_error
            self.node.get_logger().info(f"[Feature Error] Initial average pixel error: {self.initial_feature_error:.2f} pixels")

        # Log feature error reduction progress every 50 iterations
        if self.initial_feature_error is not None and self.initial_feature_error > 0:
            if self.vs_controller.iteration_count % 50 == 0:
                feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)
                self.node.get_logger().info(f"[Feature Error] Iter {self.vs_controller.iteration_count}: "
                            f"{self.current_feature_error:.2f} pixels ({feature_error_reduction*100:.1f}% reduction)")

        # Draw points only if visualization is enabled
        if getattr(self.config, 'enable_visualization', True):
            self.draw_points(np.array(self.latest_pil_image), s_uv, s_uv_star)

        # Transform feature points to real-world coordinates
        s_xy, s_star_xy = self.vs_controller.transform_to_real_world(s_uv, s_uv_star)

        # Get depth
        _t_depth = time.perf_counter()
        Z = self.get_depth(s_uv)
        self._timing_depth = time.perf_counter() - _t_depth
        if Z is None:
            self.node.get_logger().warn("Failed to get depth - skipping this iteration")
            return

        # Compute control velocity
        _t_ctrl = time.perf_counter()
        v_c = self.vs_controller.compute_control_velocity(s_xy, s_star_xy, Z)
        self._timing_control = time.perf_counter() - _t_ctrl

        # Update iteration statistics
        self.vs_controller.update_iteration_stats(v_c)

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store iteration time for FPS calculation
        self.vs_controller.iteration_times.append(execution_time)
        # Get current pose error
        if hasattr(self, 'camera_position') and hasattr(self, 'desired_position') and self.camera_position is not None and self.desired_position is not None:
            position_error = np.linalg.norm(self.camera_position - self.desired_position)
            position_error_cm = position_error * 100  # Convert to cm
            
            # Calculate rotation error in degrees
            if hasattr(self, 'orientation_quaternion') and hasattr(self, 'desired_orientation') and self.orientation_quaternion is not None and self.desired_orientation is not None:
                from scipy.spatial.transform import Rotation
                R_current = Rotation.from_quat(self.orientation_quaternion)
                R_desired = Rotation.from_quat(self.desired_orientation)
                R_error = R_desired.inv() * R_current
                rotation_error_deg = np.linalg.norm(R_error.as_rotvec()) * 180 / np.pi
            else:
                rotation_error_deg = 0.0
            
            # Get mode and bbox info
            if self.roi_tiling_activated:
                mode = "ROI-Tiled"
                # Add tile number and VIT size info
                tile_num = self.get_effective_tile_number()
                mode += f" T{tile_num}"
            elif self.tiling_activated:
                mode = "Tiled"
                # Add tile number info - use the effective tile number
                effective_tiles = self._get_full_image_tile_number()
                mode += f" T{effective_tiles}"
                # Add asterisk if auto-calculated
                if not (hasattr(self.config, 'auto_calculate_tiles') and not self.config.auto_calculate_tiles):
                    mode += "*"  # Indicates auto-calculated
            elif self.config.use_roi_detection:
                mode = "ROI"
            else:
                mode = "Original"
            
            # Get current bbox dimensions if available
            bbox_info = ""
            if self.config.use_roi_detection and self.current_roi_bbox is not None:
                x1, y1, x2, y2 = self.current_roi_bbox
                width = int(x2 - x1)
                height = int(y2 - y1)
                bbox_info = f" | BBox: {width}x{height}"
            
            # Add feature count and VIT size to output
            feature_info = f" | F: {self.last_num_features}/{self.config.num_pairs}"
            vit_info = f" | VIT: {self.config.vit_input_size}"

            # Add feature error info (real-world observable metric)
            feature_error_info = ""
            if self.initial_feature_error is not None and self.initial_feature_error > 0 and self.current_feature_error is not None:
                feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)
                feature_error_info = f" | FeatErr: {self.current_feature_error:.1f}px ({feature_error_reduction*100:.0f}%↓)"

            # Check if we should display this iteration (respecting display frequency)
            display_freq = getattr(self.config, 'iteration_display_freq', 1)
            if self.vs_controller.iteration_count % display_freq == 0 or self.vs_controller.iteration_count == 1:
                print(f"Iter: {self.vs_controller.iteration_count:4d} | Time: {execution_time:4.2f}s | Trans: {position_error_cm:5.2f}cm | Rot: {rotation_error_deg:5.2f}° | Mode: {mode}{bbox_info}{feature_info}{vit_info}{feature_error_info}")
        else:
            # Simple output when error info not available
            display_freq = getattr(self.config, 'iteration_display_freq', 1)
            if self.vs_controller.iteration_count % display_freq == 0 or self.vs_controller.iteration_count == 1:
                print(f"Iter: {self.vs_controller.iteration_count:4d} | Time: {execution_time:4.2f}s")
        
        return v_c
    
    def run(self):
        """Main visual servoing control loop."""
        # Print clean start message
        print("\n" + "="*70)
        print("Starting visual servoing...")

        # Reset feature error tracking for new sample
        self.initial_feature_error = None
        self.current_feature_error = None

        # Reset tiling activation state for new sample
        self.tiling_activated = False
        self.tiling_switch_iteration = None

        # Reset ROI tiling activation state
        if hasattr(self, 'roi_tiling_activated'):
            self.roi_tiling_activated = False
            self.roi_tiling_switch_iteration = None

        # Preserve preset initial errors if they exist
        preset_trans_error = None
        preset_rot_error = None
        preset_flag = False
        if hasattr(self.vs_controller, 'initial_errors_preset') and self.vs_controller.initial_errors_preset:
            preset_trans_error = self.vs_controller.initial_error_translation
            preset_rot_error = self.vs_controller.initial_error_rotation
            preset_flag = True
            self.node.get_logger().debug(f"Preserving preset errors: Trans={preset_trans_error:.2f}cm, Rot={preset_rot_error:.1f}°")
        
        # Reset controller state
        self.vs_controller.reset()
        self.feature_failure_count = 0
        self.consecutive_no_features_count = 0  # Reset classical method failure count
        self.tiling_activated = False
        self.tiling_switch_iteration = None
        self.roi_tiling_activated = False
        self.roi_tiling_switch_iteration = None
        
        # Restore preset initial errors after reset
        if preset_flag:
            self.vs_controller.initial_error_translation = preset_trans_error
            self.vs_controller.initial_error_rotation = preset_rot_error
            self.vs_controller.initial_errors_preset = True
            self.node.get_logger().debug(f"Restored preset errors after reset: Trans={preset_trans_error:.2f}cm, Rot={preset_rot_error:.1f}°")
        
        # Reset current ROI cache at start of servoing
        self.current_roi_bbox = None
        self.last_valid_roi_iteration = 0

        # Get initial camera pose and calculate initial errors
        # Use mode-dependent method (Gazebo for simulation, TF2 for real robot)
        self.camera_position, self.orientation_quaternion = self.get_camera_pose()
        if self.camera_position is None or self.orientation_quaternion is None:
            self.node.get_logger().error("Failed to get initial camera pose")
            return None

        # Calculate initial errors (but don't overwrite if they were preset)
        initial_error_translation, initial_error_rotation = self.vs_controller.calculate_end_error(
            self.camera_position, self.orientation_quaternion,
            self.desired_position, self.desired_orientation)
        
        # Check if initial errors were preset (e.g., from true sampled pose before rotation)
        if hasattr(self.vs_controller, 'initial_errors_preset') and self.vs_controller.initial_errors_preset:
            # Use the preset values
            print(f"Using preset initial errors: Trans={self.vs_controller.initial_error_translation:.2f}cm, Rot={self.vs_controller.initial_error_rotation:.1f}°")
            print(f"Current error after rotation: Trans={initial_error_translation:.2f}cm, Rot={initial_error_rotation:.1f}°")
            print(f"Rotation compensation improved by: Trans={self.vs_controller.initial_error_translation - initial_error_translation:.2f}cm, Rot={self.vs_controller.initial_error_rotation - initial_error_rotation:.1f}°")
        else:
            # Set initial errors from current pose (backward compatibility)
            self.vs_controller.initial_error_translation = initial_error_translation
            self.vs_controller.initial_error_rotation = initial_error_rotation
            print(f"Initial error: Trans={initial_error_translation:.2f}cm, Rot={initial_error_rotation:.1f}°")
        
        # Print convergence requirements
        print("\n" + "="*70)
        print("CONVERGENCE REQUIREMENTS:")
        print(f"  To converge, errors must be reduced by 90% from initial:")
        print(f"  • Translation must reach < {self.vs_controller.initial_error_translation * 0.1:.2f}cm (10% of {self.vs_controller.initial_error_translation:.2f}cm)")
        print(f"  • Rotation must reach < {self.vs_controller.initial_error_rotation * 0.1:.2f}° (10% of {self.vs_controller.initial_error_rotation:.2f}°)")
        print("="*70 + "\n")

        # Initialize tracking of lowest errors
        lowest_position_error = float('inf')
        lowest_orientation_error = float('inf')

        _loop_iter = 0
        _has_executor = self._executor is not None and self.node is not None
        _robot_mode = getattr(self.config, 'robot_mode', 'simulation')
        _is_real = _robot_mode == 'real'

        # ── Main-thread spinning ──────────────────────────────────────────
        # During the VS loop we take control of callback processing from the
        # background executor thread.  This eliminates GIL contention:
        #   • ibvs() runs with NO other thread competing for the GIL
        #   • Callbacks (image, TF) are processed in controlled bursts on the
        #     main thread via spin_some(), so no TF backlog can form
        #
        # The 100 Hz twist republisher is a plain Python thread (not an
        # executor callback) — it continues independently.
        if _has_executor and _is_real:
            self._executor.remove_node(self.node)  # Detach from background spin thread
            from rclpy.executors import SingleThreadedExecutor
            self._loop_executor = SingleThreadedExecutor()
            self._loop_executor.add_node(self.node)
            print("[VS-LOOP] Using main-thread executor (no GIL contention)")
        else:
            self._loop_executor = None

        def _spin_for_callbacks(max_seconds=0.25):
            """Spin on the main thread until a fresh image arrives or timeout.
            No background thread involved — zero GIL contention."""
            if self._loop_executor is None:
                return
            self._image_event.clear()
            _deadline = time.perf_counter() + max_seconds
            while time.perf_counter() < _deadline:
                # Process one pending callback (image, TF, timer, …).
                # timeout_sec=0 → return immediately if nothing ready.
                self._loop_executor.spin_once(timeout_sec=0.005)
                if self._image_event.is_set():
                    break  # Got a fresh image — done

        try:
            while rclpy.ok():
                _wall_start = time.perf_counter()

                # --- Phase 0: Wait for initial images ---
                if self.latest_image is None or self.latest_image_depth is None:
                    _waiting_rgb = self.latest_image is None
                    _waiting_depth = self.latest_image_depth is None
                    if _loop_iter == 0 or _loop_iter % 10 == 0:
                        print(f"[VS-DEBUG] Waiting for callbacks (iter {_loop_iter})... "
                              f"RGB={'pending' if _waiting_rgb else 'OK'}, "
                              f"Depth={'pending' if _waiting_depth else 'OK'}")
                    if self._loop_executor:
                        self._loop_executor.spin_once(timeout_sec=0.1)
                    else:
                        time.sleep(0.1)
                    _loop_iter += 1
                    continue

                if _loop_iter == 0 or (self.vs_controller.iteration_count == 0 and _loop_iter < 3):
                    print(f"[VS-DEBUG] Loop iter {_loop_iter}: image OK ({type(self.latest_image).__name__}), calling ibvs()...")

                # --- Phase 1: ibvs() — feature extraction + velocity ---
                # No executor running = zero GIL contention.  CUDA ops get
                # full CPU without TF/image callbacks competing.
                _t_ibvs_start = time.perf_counter()
                try:
                    v_c = self.ibvs()
                    if v_c is None:
                        if _loop_iter < 5:
                            print(f"[VS-DEBUG] ibvs() returned None (iter {_loop_iter})")
                        _spin_for_callbacks(0.1)
                        _loop_iter += 1
                        continue
                except RuntimeError as e:
                    if str(e) == "Persistent feature detection failure":
                        self.node.get_logger().error("Aborting sample due to persistent feature detection failures")
                        return self._create_error_return_tuple()  # finally block handles cleanup
                    raise
                _t_ibvs = time.perf_counter() - _t_ibvs_start

                # --- Phase 2: Publish velocity IMMEDIATELY ---
                # TF buffer was populated by spin_some in previous iteration —
                # lookup is instant (no executor needed).
                _t_pub_start = time.perf_counter()
                self.publish_twist(v_c)
                _t_pub = time.perf_counter() - _t_pub_start

                # --- Phase 3: Get pose (cached TF lookup) ---
                _t_pose_start = time.perf_counter()
                self.camera_position, self.orientation_quaternion = self.get_camera_pose()
                _t_pose = time.perf_counter() - _t_pose_start

                self.vs_controller.position_history.append(self.camera_position)
                self.vs_controller.orientation_history.append(self.orientation_quaternion)

                # Calculate current errors
                current_position_error, current_orientation_error = self.vs_controller.calculate_end_error(
                    self.camera_position, self.orientation_quaternion,
                    self.desired_position, self.desired_orientation)

                # Update lowest errors
                lowest_position_error = min(lowest_position_error, current_position_error)
                lowest_orientation_error = min(lowest_orientation_error, current_orientation_error)

                # --- Phase 4: Spin for fresh image + TF updates ---
                # Process callbacks on the MAIN thread in controlled bursts.
                # Exits as soon as a new camera frame arrives (or 200ms timeout).
                # Robot continues moving via 100Hz twist republisher thread.
                _t_spin_start = time.perf_counter()
                if _is_real:
                    _spin_for_callbacks(0.2)
                _t_spin = time.perf_counter() - _t_spin_start

                # Total wall-clock time for this iteration
                _t_wall_total = time.perf_counter() - _wall_start

                # Timing logging disabled — print I/O causes Servo velocity gaps

                _loop_iter += 1

                # Check if servoing is done
                # Use visual-only convergence for real robot mode
                convergence_mode = getattr(self.config, 'convergence_mode', 'full')
                robot_mode = getattr(self.config, 'robot_mode', 'simulation')

                # Determine convergence mode automatically if not explicitly set
                if convergence_mode == 'full' and robot_mode == 'real':
                    convergence_mode = 'visual_only'

                if convergence_mode == 'visual_only':
                    # Use visual-only convergence (no ground truth pose required)
                    done, converged = self.vs_controller.check_convergence_visual_only(
                        feature_error=self.current_feature_error,
                        velocity=v_c if 'v_c' in dir() else None)
                else:
                    # Standard convergence with ground truth pose
                    done, converged = self.vs_controller.check_convergence(
                        self.camera_position, self.orientation_quaternion,
                        self.desired_position, self.desired_orientation,
                        feature_error=self.current_feature_error)

                if done:
                    self.node.get_logger().info(f"Visual servoing completed after {self.vs_controller.iteration_count} iterations.")
                    self.node.get_logger().info(f"Converged: {converged}")
                    if convergence_mode != 'visual_only':
                        self.node.get_logger().info(f"Final Position Error: {current_position_error:.2f} cm")
                        self.node.get_logger().info(f"Final Orientation Error: {current_orientation_error:.2f} degrees")
                        self.node.get_logger().info(f"Lowest Position Error: {lowest_position_error:.2f} cm")
                        self.node.get_logger().info(f"Lowest Orientation Error: {lowest_orientation_error:.2f} degrees")
                    else:
                        self.node.get_logger().info(f"Final Feature Error: {self.current_feature_error:.2f} pixels (visual-only mode)")

                    return self._create_return_tuple(converged, current_position_error,
                                                   current_orientation_error, lowest_position_error,
                                                   lowest_orientation_error)

        except Exception as e:
            import traceback
            self.node.get_logger().error(f"Error in run loop: {str(e)}")
            self.node.get_logger().error(f"Traceback:\n{traceback.format_exc()}")
            return self._create_error_return_tuple()
        finally:
            # Restore node to background executor so callbacks resume for
            # post-VS operations (e.g. moving home, next sample).
            if self._loop_executor is not None:
                try:
                    self._loop_executor.remove_node(self.node)
                except Exception:
                    pass
                self._loop_executor = None
            if _has_executor and _is_real:
                try:
                    self._executor.add_node(self.node)
                except Exception:
                    pass
    
    def _create_return_tuple(self, converged, current_position_error, current_orientation_error,
                           lowest_position_error, lowest_orientation_error):
        """Create return tuple with all tracking data."""
        return (self.camera_position, self.orientation_quaternion, converged,
                current_position_error, current_orientation_error,
                np.array(self.vs_controller.position_history), 
                np.array(self.vs_controller.orientation_history),
                self.vs_controller.iteration_count,
                lowest_position_error, lowest_orientation_error,
                np.array(self.vs_controller.average_velocities),
                np.array(self.vs_controller.velocity_mean_100),
                np.array(self.vs_controller.velocity_mean_10),
                np.array(self.vs_controller.applied_velocity_x),
                np.array(self.vs_controller.applied_velocity_y),
                np.array(self.vs_controller.applied_velocity_z),
                np.array(self.vs_controller.applied_velocity_roll),
                np.array(self.vs_controller.applied_velocity_pitch),
                np.array(self.vs_controller.applied_velocity_yaw),
                np.array(self.vs_controller.iteration_times))  # Add iteration times
    
    def _create_error_return_tuple(self):
        """Helper method to create return tuple for error cases."""
        return self._create_return_tuple(False, float('inf'), float('inf'), 
                                       float('inf'), float('inf'))
    
    def _calculate_optimal_tile_number(self, bbox):
        """
        Calculate optimal tile grid dimensions based on ROI bounding box.
        Supports rectangular grids (e.g., 4x2) for better resolution utilization.

        Args:
            bbox: Tuple (x1, y1, x2, y2) representing the bounding box
        """
        self.node.get_logger().info(f"[Auto Tile Calculation] Called with bbox: {bbox}")

        # Only calculate if hybrid mode is enabled and auto calculation is not disabled
        if not getattr(self.config, 'use_hybrid_mode', False):
            self.node.get_logger().info("[Auto Tile Calculation] Skipped - use_hybrid_mode is False")
            return

        # Check if auto calculation is explicitly disabled
        if hasattr(self.config, 'auto_calculate_tiles') and not self.config.auto_calculate_tiles:
            self.node.get_logger().info("Automatic tile calculation is disabled by config")
            return

        # Extract bbox dimensions
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Get VIT input size
        vit_input_size = self.config.vit_input_size

        # Calculate tiles needed for each dimension independently
        import math
        tiles_width = math.ceil(bbox_width / vit_input_size)
        tiles_height = math.ceil(bbox_height / vit_input_size)

        # Apply constraints per dimension
        min_tiles_per_dim = 1
        max_tiles_per_dim = 5  # Maximum to prevent memory issues

        tiles_width = max(min_tiles_per_dim, min(tiles_width, max_tiles_per_dim))
        tiles_height = max(min_tiles_per_dim, min(tiles_height, max_tiles_per_dim))

        # Store rectangular grid dimensions
        self.optimal_grid_width = tiles_width
        self.optimal_grid_height = tiles_height
        self.optimal_tile_number = tiles_width * tiles_height
        
        # Calculate coverage efficiency with rectangular grid
        coverage_width_pixels = tiles_width * vit_input_size
        coverage_height_pixels = tiles_height * vit_input_size
        coverage_width_ratio = coverage_width_pixels / bbox_width
        coverage_height_ratio = coverage_height_pixels / bbox_height

        self.node.get_logger().info(f"[Auto Tile Calculation] ROI bbox: {bbox_width:.0f}x{bbox_height:.0f}, VIT size: {vit_input_size}")
        self.node.get_logger().info(f"[Auto Tile Calculation] Optimal grid: {tiles_width}x{tiles_height} = {self.optimal_tile_number} tiles")
        self.node.get_logger().info(f"[Auto Tile Calculation] Coverage: {coverage_width_pixels}x{coverage_height_pixels} pixels ({coverage_width_ratio:.1f}x width, {coverage_height_ratio:.1f}x height)")
        
        # Note: This overrides config.tile_number when use_hybrid_mode=true AND use_roi_detection=true
        # To disable auto calculation, set auto_calculate_tiles=false in config
    
    def get_effective_tile_number(self):
        """
        Get the effective tile number to use.
        Returns optimal_tile_number if calculated, otherwise config.tile_number.
        
        This method is called when ROI tiling is active to determine the tile grid size.
        """
        if self.optimal_tile_number is not None:
            return self.optimal_tile_number
        return getattr(self.config, 'roi_tile_number', 4)
    
    def _get_full_image_tile_number(self):
        """
        Get the tile number for full image tiling (MODE 2).
        
        If auto_calculate_tiles is enabled (or not explicitly disabled),
        calculates optimal tile number based on full image resolution.
        Otherwise returns config.tile_number.
        
        Returns:
            int: Number of tiles to use (must be a perfect square)
        """
        # Check if auto calculation is explicitly disabled
        if hasattr(self.config, 'auto_calculate_tiles') and not self.config.auto_calculate_tiles:
            return self.config.tile_number
            
        # Calculate optimal tiles based on full image resolution
        import math
        
        # Full image dimensions
        img_width = self.config.u_max  # 1920
        img_height = self.config.v_max  # 1080
        vit_input_size = self.config.vit_input_size
        
        # Calculate tiles needed for each dimension
        tiles_width = math.ceil(img_width / vit_input_size)
        tiles_height = math.ceil(img_height / vit_input_size)
        
        # Take the maximum to ensure full coverage in both dimensions
        # We need a square grid that can cover both width and height
        grid_size = max(tiles_width, tiles_height)
        
        # The total number of tiles is grid_size squared
        optimal_tiles_square = grid_size ** 2
        
        self.node.get_logger().info(f"[Auto Tile Calculation - Full Image] Image: {img_width}x{img_height}, "
                     f"VIT: {vit_input_size}, Tiles needed: {tiles_width}x{tiles_height}, "
                     f"Using: {grid_size}x{grid_size} = {optimal_tiles_square} tiles")
        
        return optimal_tiles_square
    
    def __del__(self):
        """Cleanup temporary files on object destruction."""
        # Clean up temporary ROI mask file if it exists
        if hasattr(self, 'goal_roi_mask_path') and self.goal_roi_mask_path:
            if self.goal_roi_mask_path != getattr(self.config, 'mask_path', None):
                try:
                    import os
                    if os.path.exists(self.goal_roi_mask_path):
                        os.unlink(self.goal_roi_mask_path)
                        self.node.get_logger().info(f"Cleaned up temporary mask file: {self.goal_roi_mask_path}")
                except Exception as e:
                    self.node.get_logger().warn(f"Failed to clean up temporary mask: {e}")
