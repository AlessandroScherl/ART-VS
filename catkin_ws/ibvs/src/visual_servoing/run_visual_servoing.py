#!/usr/bin/env python3
"""
Real Robot Visual Servoing Entry Point

This script runs visual servoing on a real UR5 robot with RealSense D435i camera.
It uses the modular visual servoing infrastructure with config-based switching
between simulation and real robot modes.

Usage:
    python3 run_visual_servoing.py --config configs/real_robot/config_real_robot_dinov3.yaml
    python3 run_visual_servoing.py --config configs/real_robot/config_real_robot_sift.yaml --skip-rotation

Prerequisites (ROS2 Jazzy, see docs/REAL_ROBOT.md for the full startup sequence):
    1. Robot driver:  ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=<ROBOT_IP> launch_rviz:=false
    2. MoveIt+Servo:  ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_servo:=true
    3. Camera:        ../robot/start_camera.sh
    4. Image cropper: python3 ../robot/crop_image_node.py
"""

import os
import sys
import signal
import time
import argparse
import threading
import subprocess
import numpy as np

# Use locally cached models when available (skip slow network checks).
# Models are downloaded on first use; subsequent runs load from cache.
# Note: not forcing offline mode — allows first-time model downloads.

# Ensure parent directory is in path for local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import rclpy
import rclpy.time
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image as ImageMsg
import torch
from PIL import Image

# Import visualization function
from vs_utils.visualization import visualize_correspondences

# TF2 for real robot pose acquisition
from tf2_ros import Buffer as TF2Buffer, TransformListener as TF2TransformListener
import tf2_ros
from ros_interface.tf_compat import quaternion_matrix as _quaternion_matrix

# Import from our modules
from core.config import Config
from ros_interface.ros2_controller import ROSVisualServoingController
from features.multi_backbone_extractor import MultiBackboneViTExtractor
from features.matcher import find_correspondences_batch, create_patch_mask_from_pil

# LangSAM + LightTrack for early ROI detection (before rotation compensation)
try:
    from features.langsam_lighttrack_detector import LangSAMLightTrackDetector
    LANGSAM_AVAILABLE = True
except ImportError:
    LangSAMLightTrackDetector = None
    LANGSAM_AVAILABLE = False

# cv_bridge for image conversion
try:
    from cv_bridge import CvBridge
except (ImportError, TypeError):
    from ros_interface.cv_bridge_wrapper import CvBridge


class RealRobotVisualServoing:
    """Main class for running visual servoing on a real robot."""

    def __init__(self, config_path, skip_rotation=False):
        self.config_path = config_path
        self.skip_rotation = skip_rotation
        self.bridge = CvBridge()
        self.latest_image = None
        self.velocity_pub = None
        self.viz_crop_pub = None  # Crop-level correspondence visualization
        self.viz_full_pub = None  # Full-image correspondence visualization
        self.running = True
        self.node = None  # Will be created in run()
        self.config = None  # Store config reference for stop_robot

        # TF2 for pose acquisition
        self.tf_buffer = None
        self.tf_listener = None

        # Persistent-worker caches: expensive model objects loaded ONCE and reused
        # across picks (they are pure torch/CUDA, independent of the ROS node, so they
        # survive node teardown between picks). Avoids ~30s model reload per pick.
        self._worker_mode = False
        self._cached_extractor = None
        self._cached_detector = None
        self._cached_goal_image = None
        self._cached_goal_roi_bbox = None  # None=not yet detected, False=detected-but-empty

    def publish_correspondence_visualization(self, goal_image, current_image, points1, points2,
                                             input_size, angle, similarity, goal_features):
        """
        Publish correspondence visualization using existing visualization function.

        Args:
            goal_image: PIL Image (resized to input_size)
            current_image: PIL Image (resized to input_size)
            points1: Patch indices from matcher in (x, y) = (col, row) format
            points2: Patch indices from matcher in (x, y) = (col, row) format
            input_size: ViT input size used (e.g., 518)
            angle: Rotation angle being tested
            similarity: Similarity score
            goal_features: Descriptor tensor to get actual patch count
        """
        if self.viz_crop_pub is None or len(points1) == 0:
            return

        # Convert to numpy
        if torch.is_tensor(points1):
            p1 = points1.cpu().numpy()
            p2 = points2.cpu().numpy()
        else:
            p1 = np.array(points1)
            p2 = np.array(points2)

        # Compute pixel coordinates from patch indices
        # Formula: pixel = patch_index * scale + scale/2 (center of patch)
        num_patches_total = goal_features.size(-2)
        num_patches_per_side = int(np.sqrt(num_patches_total))
        scale = input_size / num_patches_per_side

        # Scale from patch indices to pixel coordinates
        # Points are in (x, y) format from matcher
        offset = scale / 2
        points1_pixels = p1 * scale + offset
        points2_pixels = p2 * scale + offset

        # Flip from (x, y) to (y, x) for visualization function
        # visualize_correspondences unpacks as (y, x) at line 38
        points1_viz = points1_pixels[:, ::-1].copy()  # [x,y] -> [y,x]
        points2_viz = points2_pixels[:, ::-1].copy()

        # Create visualization using existing function
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = visualize_correspondences(goal_image, current_image, points1_viz, points2_viz)

        # Convert figure to image and publish
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

        try:
            ros_image = self.bridge.cv2_to_imgmsg(img_data, encoding="rgb8")
            self.viz_crop_pub.publish(ros_image)
        except Exception as e:
            if self.node:
                self.node.get_logger().warn(f"Failed to publish crop visualization: {e}")

        plt.close(fig)

    def publish_correspondence_visualization_full(self, full_goal, full_current,
                                                   points1, points2,
                                                   crop_input_size, angle, similarity,
                                                   goal_features,
                                                   goal_roi_bbox, current_roi_bbox,
                                                   goal_padded_size=None, current_padded_size=None):
        """
        Publish visualization showing correspondences on FULL images.
        Points are in ROI-crop patch space and get mapped to full-image pixel space
        with rotation-aware coordinate transforms.

        IMPORTANT: When ROI crops are padded to square for rotation, the mapping must
        account for the padding offset. Points in 518×518 space now represent the
        padded square, not the original ROI directly.

        Coordinate mapping (rotation-aware, with padding support):
            1. patch_index -> pixel_in_518x518_padded_crop  (scale by patch_size, offset by half)
            2. pixel_in_518_padded -> pixel_in_original_padded_crop  (scale by padded_size / 518)
            3. pixel_in_padded_crop -> pixel_in_UNPADDED_crop  (subtract padding offset)
            4. pixel_in_unpadded_crop -> pixel_in_full_image  (+ bbox offset, with rotation if needed)

        Args:
            full_goal: Full (uncropped) goal PIL Image (never rotated)
            full_current: Full (uncropped) current PIL Image (already rotated if angle!=0)
            points1: Patch indices in goal crop space (x, y) = (col, row)
            points2: Patch indices in current crop space (x, y) = (col, row)
            crop_input_size: Size crops were resized to for feature extraction (518)
            angle: Rotation angle being tested
            similarity: Similarity score
            goal_features: Descriptor tensor to get actual patch count
            goal_roi_bbox: [x1, y1, x2, y2] goal ROI in full image coords (float)
            current_roi_bbox: [x1, y1, x2, y2] current ROI in UNROTATED full image coords (float)
            goal_padded_size: Side length of padded goal square (None if no padding)
            current_padded_size: Side length of padded current square (None if no padding)
        """
        if self.viz_full_pub is None or len(points1) == 0:
            return

        import cv2

        # Convert points to numpy
        if torch.is_tensor(points1):
            p1 = points1.cpu().numpy()
            p2 = points2.cpu().numpy()
        else:
            p1 = np.array(points1)
            p2 = np.array(points2)

        # Get patch grid info from features
        num_patches_total = goal_features.size(-2)
        num_patches_per_side = int(np.sqrt(num_patches_total))
        scale = crop_input_size / num_patches_per_side
        offset = scale / 2

        # Step 1: patch indices -> pixel coords in resized crop (518x518)
        p1_crop_px = p1 * scale + offset
        p2_crop_px = p2 * scale + offset

        # Map goal points to full image (goal is never rotated, angle=0)
        p1_full = self._map_crop_points_to_rotated_full(
            p1_crop_px, crop_input_size, goal_roi_bbox, 0, full_goal.size,
            padded_size=goal_padded_size)

        # Map current points to rotated full image (uses actual angle)
        p2_full = self._map_crop_points_to_rotated_full(
            p2_crop_px, crop_input_size, current_roi_bbox, angle, full_current.size,
            padded_size=current_padded_size)

        # Get full image dimensions
        goal_w, goal_h = full_goal.size   # PIL: (width, height)
        curr_w, curr_h = full_current.size

        # Filter points outside visible image area (can happen for non-zero angles)
        valid_goal = ((p1_full[:, 0] >= 0) & (p1_full[:, 0] < goal_w) &
                      (p1_full[:, 1] >= 0) & (p1_full[:, 1] < goal_h))
        valid_curr = ((p2_full[:, 0] >= 0) & (p2_full[:, 0] < curr_w) &
                      (p2_full[:, 1] >= 0) & (p2_full[:, 1] < curr_h))
        valid_mask = valid_goal & valid_curr

        n_total = len(p1_full)
        n_visible = valid_mask.sum()
        if n_visible == 0:
            print(f"  [VIZ] All {n_total} points outside image bounds, skipping")
            return

        p1_full = p1_full[valid_mask]
        p2_full = p2_full[valid_mask]

        # Draw ROI bounding boxes on full images
        goal_display = np.array(full_goal.copy())
        curr_display = np.array(full_current.copy())

        # Goal bbox: always axis-aligned (goal is never rotated)
        gx1, gy1, gx2, gy2 = [int(x) for x in goal_roi_bbox]
        cv2.rectangle(goal_display, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)

        # Current bbox: draw as rotated polygon (correct for any angle)
        curr_corners = self._rotate_bbox(current_roi_bbox, angle, full_current.size)
        pts = curr_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(curr_display, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Convert back to PIL for visualization function
        goal_pil = Image.fromarray(goal_display)
        curr_pil = Image.fromarray(curr_display)

        # Resize to display size
        display_w = 518
        display_h = 518

        goal_display_pil = goal_pil.resize((display_w, display_h))
        curr_display_pil = curr_pil.resize((display_w, display_h))

        # Scale points to display coordinates
        p1_display = p1_full * np.array([display_w / goal_w, display_h / goal_h])
        p2_display = p2_full * np.array([display_w / curr_w, display_h / curr_h])

        # Flip from (x, y) to (y, x) for visualize_correspondences which unpacks as (y, x)
        p1_viz = p1_display[:, ::-1].copy()
        p2_viz = p2_display[:, ::-1].copy()

        print(f"  [VIZ] {n_total} points mapped to full image ({n_visible} visible), angle={angle}°")

        # Create visualization using existing function
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = visualize_correspondences(goal_display_pil, curr_display_pil, p1_viz, p2_viz)

        # Convert figure to image and publish
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

        try:
            ros_image = self.bridge.cv2_to_imgmsg(img_data, encoding="rgb8")
            self.viz_full_pub.publish(ros_image)
        except Exception as e:
            if self.node:
                self.node.get_logger().warn(f"Failed to publish full-image visualization: {e}")

        plt.close(fig)

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully - stop the robot."""
        print("\n\nReceived interrupt signal. Stopping robot...")
        self.running = False
        self.stop_robot()

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        if self.velocity_pub is not None and self.node is not None:
            ts = TwistStamped()
            ts.header.stamp = self.node.get_clock().now().to_msg()
            ts.header.frame_id = getattr(self.config, 'velocity_frame_id', 'tool0') if self.config else 'tool0'
            for _ in range(5):  # Publish multiple times to ensure it's received
                self.velocity_pub.publish(ts)
                time.sleep(0.05)
            print("Robot stopped (zero velocity published)")

    def image_callback(self, msg):
        """Store latest image for rotation alignment."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            import cv2
            self.latest_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            if self.node:
                self.node.get_logger().warn(f"Image callback error: {e}")

    def get_camera_pose_tf(self, config):
        """Get camera pose from TF tree with retry + spinning."""
        if self.tf_buffer is None:
            return None, None

        base_frame = getattr(config, 'base_frame', 'base')
        camera_frame = getattr(config, 'camera_frame', 'camera_color_optical_frame')

        # Retry with spinning — lookup_transform timeout doesn't spin the node
        deadline = time.time() + 5.0
        last_error = None
        while time.time() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame, camera_frame, rclpy.time.Time(),
                    rclpy.time.Duration(seconds=0.1))

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

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                last_error = e
                # Wait for background executor to receive more TF messages
                time.sleep(0.2)

        if self.node and last_error:
            self.node.get_logger().error(f"TF lookup error after retries: {last_error}")
        return None, None

    @staticmethod
    def _pad_to_square(img, fill_color=(128, 128, 128)):
        """
        Pad a PIL Image to a square by adding borders.

        This is critical for rotation alignment: when rotating a tall/narrow ROI crop
        by 90° or 270°, the content would be clipped if we don't pad first.

        Example problem without padding:
            Original ROI (200x400 tall bottle):
            - At 0°/180°: Full bottle visible
            - At 90°/270°: Rotating clips the bottle at both ends!

        Solution: Pad to square (400x400) first, then rotate safely.

        Args:
            img: PIL Image to pad
            fill_color: RGB tuple for padding color (default gray)

        Returns:
            PIL Image padded to square dimensions
        """
        w, h = img.size
        if w == h:
            return img

        size = max(w, h)

        # Create square canvas with fill color
        padded = Image.new('RGB', (size, size), fill_color)

        # Center the original image in the square
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded.paste(img, (x_offset, y_offset))

        return padded

    @staticmethod
    def _pad_mask_to_square(mask_tensor, original_size, padded_size, vit_input_size, patch_size):
        """
        Pad a mask tensor to match a square-padded image.

        When we pad an image to square, the mask must be padded correspondingly
        so that valid patches still align with the object in the padded image.

        Args:
            mask_tensor: Original mask tensor (1, num_patches) or (1, 1, num_patches)
            original_size: (width, height) of original image before padding
            padded_size: Side length of the square-padded image
            vit_input_size: Size images are resized to for ViT (e.g., 518)
            patch_size: ViT patch size (14 for DINOv2, 16 for DINOv3)

        Returns:
            Padded mask tensor matching the square image's patch grid
        """
        if mask_tensor is None:
            return None

        orig_w, orig_h = original_size

        # Number of patches in each dimension for the padded square
        num_patches_per_side = vit_input_size // patch_size

        # Calculate patch-space offsets (how many patches of padding on each side)
        # The image is centered in the square, so padding is split between sides
        x_offset_pixels = (padded_size - orig_w) // 2
        y_offset_pixels = (padded_size - orig_h) // 2

        # Convert pixel offsets to patch offsets
        # After resizing to vit_input_size, calculate the patch offset
        scale = vit_input_size / padded_size
        x_offset_patches = int(round(x_offset_pixels * scale / patch_size))
        y_offset_patches = int(round(y_offset_pixels * scale / patch_size))

        # Original patch grid dimensions (before padding)
        orig_patches_x = int(round(orig_w / padded_size * num_patches_per_side))
        orig_patches_y = int(round(orig_h / padded_size * num_patches_per_side))

        # Ensure we don't exceed bounds
        orig_patches_x = min(orig_patches_x, num_patches_per_side - 2 * x_offset_patches)
        orig_patches_y = min(orig_patches_y, num_patches_per_side - 2 * y_offset_patches)

        # Reshape original mask to 2D if needed
        mask_flat = mask_tensor.view(-1)
        orig_num_patches = int(np.sqrt(len(mask_flat)))

        # Create new mask grid (all False = padding area invalid)
        new_mask = torch.zeros((num_patches_per_side, num_patches_per_side),
                               dtype=mask_tensor.dtype, device=mask_tensor.device)

        # Reshape original mask to 2D grid
        try:
            orig_mask_2d = mask_flat[:orig_num_patches * orig_num_patches].view(orig_num_patches, orig_num_patches)
        except:
            # If mask doesn't reshape cleanly, return None to skip masking
            return None

        # Place original mask in the center of the new mask
        # We need to resample if the patch counts don't match exactly
        if orig_patches_x > 0 and orig_patches_y > 0:
            # Simple nearest-neighbor resampling of the mask
            for py in range(orig_patches_y):
                for px in range(orig_patches_x):
                    # Map to original mask coordinates
                    src_y = min(int(py * orig_num_patches / orig_patches_y), orig_num_patches - 1)
                    src_x = min(int(px * orig_num_patches / orig_patches_x), orig_num_patches - 1)

                    # Map to new mask coordinates (with offset)
                    dst_y = y_offset_patches + py
                    dst_x = x_offset_patches + px

                    if 0 <= dst_y < num_patches_per_side and 0 <= dst_x < num_patches_per_side:
                        new_mask[dst_y, dst_x] = orig_mask_2d[src_y, src_x]

        return new_mask.view(1, -1)

    def find_best_rotation(self, extractor, goal_image, current_image, config,
                           num_pairs=64, pair_weight_alpha=0.1,
                           mask_tensor=None,
                           full_goal_image=None, full_current_image=None,
                           goal_roi_bbox=None, current_roi_bbox=None):
        """
        Find the best discrete rotation (0, 90, 180, 270 degrees) for alignment.

        Uses a combined scoring that considers both feature similarity AND the number
        of matched pairs. More pairs = higher confidence in the similarity estimate.

        Combined score formula:
            combined_score = similarity * (pairs / max_pairs) ^ pair_weight_alpha

        This ensures that a rotation with slightly lower similarity but significantly
        more matched pairs can win over one with marginally higher similarity but
        fewer pairs (which may be less reliable).

        IMPORTANT: ROI crops are padded to square before rotation to prevent clipping.
        A tall bottle ROI (200x400) rotated 90° would clip content without padding.

        Args:
            extractor: Feature extractor
            goal_image: PIL Image of goal (may be ROI crop or full image)
            current_image: PIL Image of current view (may be ROI crop or full image)
            config: Configuration object
            num_pairs: Number of feature correspondences to use (default 64 for robust matching)
            pair_weight_alpha: Exponent for pair count weighting (default 0.3).
                               Higher = more weight on pair count.
                               0 = ignore pair count (original behavior).
                               1 = linear weighting by pair ratio.
            mask_tensor: Pre-computed mask tensor for ROI-cropped images
            full_goal_image: Full (uncropped) goal PIL Image for visualization (None = use crop)
            full_current_image: Full (uncropped) current PIL Image for visualization (None = use crop)
            goal_roi_bbox: [x1, y1, x2, y2] bbox of goal ROI in full image coords
            current_roi_bbox: [x1, y1, x2, y2] bbox of current ROI in full image coords

        Returns:
            best_angle: The angle that gives best feature match (0, 90, 180, or 270)
            best_score: The combined score at the best angle
        """
        from torchvision import transforms

        angles = [0, 90, 180, 270]
        similarities = []  # Raw mean similarity for each rotation
        pair_counts = []   # Number of matched pairs for each rotation

        # Use higher resolution (518x518) for rotation alignment - better matching accuracy
        # This is independent of config.vit_input_size which may be smaller (e.g., 256)
        input_size = 518
        print(f"  Using {input_size}x{input_size} resolution for rotation alignment with {num_pairs} feature pairs")

        # IMPORTANT: Use simple preprocessing (ToTensor + Normalize only) like the reference script
        # The extractor's preprocess_pil may apply Resize+CenterCrop which would cause
        # a mismatch between visualization images and the images the model actually sees.
        # By using simple preprocessing, the manually resized images == what the model sees.
        simple_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # =====================================================================
        # CRITICAL FIX: Pad ROI crops to square before rotation
        # =====================================================================
        # Problem: A tall bottle ROI (e.g., 200x400) rotated by 90° clips content
        #          because the horizontal bottle doesn't fit in the 200x400 frame.
        # Solution: Pad to square (400x400) first, then all rotations preserve full content.

        goal_orig_size = goal_image.size
        current_orig_size = current_image.size

        goal_padded = self._pad_to_square(goal_image)
        current_padded = self._pad_to_square(current_image)

        goal_padded_size = goal_padded.size[0]  # Square, so width == height
        current_padded_size = current_padded.size[0]

        # Log padding info if images were actually padded
        if goal_orig_size != goal_padded.size:
            print(f"  [PADDING] Goal ROI: {goal_orig_size} -> {goal_padded.size} (padded to square)")
        if current_orig_size != current_padded.size:
            print(f"  [PADDING] Current ROI: {current_orig_size} -> {current_padded.size} (padded to square)")

        # Resize padded goal image to model input size
        goal_resized = goal_padded.resize((input_size, input_size))

        # Preprocess goal image using simple transform (no resize/crop)
        goal_tensor = simple_preprocess(goal_resized).unsqueeze(0).to(extractor.device)

        with torch.no_grad():
            goal_features = extractor.extract_descriptors(
                goal_tensor,
                layer=11, facet='token', bin=config.use_feature_binning,
                hierarchy=config.feature_hierarchy
            )

        # Setup mask for rotation alignment
        # Priority: mask_tensor (pre-cropped to ROI) > mask_path from config
        rotation_mask_tensor = mask_tensor  # May be None (from parameter)
        rotation_mask_path = None

        if rotation_mask_tensor is not None:
            print(f"  [MASK] Using pre-computed ROI-cropped mask tensor "
                  f"({rotation_mask_tensor.sum().item()}/{rotation_mask_tensor.numel()} valid patches)")
        else:
            # config.mask_path is already resolved to absolute by Config class
            rotation_mask_path = getattr(config, 'mask_path', None)

            if rotation_mask_path and os.path.exists(rotation_mask_path):
                print(f"  [MASK] Using mask: {rotation_mask_path}")
            elif rotation_mask_path:
                print(f"  [MASK] WARNING: Mask file not found: {rotation_mask_path}")
            else:
                print(f"  [MASK] No mask configured - matches will be across entire image")

        # DINOv3 uses patch_size=16, DINOv2 uses patch_size=14
        patch_size = 16 if 'dinov3' in config.backbone_model else 14

        # If mask was provided, pad it to match the square-padded goal image
        if rotation_mask_tensor is not None and goal_orig_size != goal_padded.size:
            rotation_mask_tensor = self._pad_mask_to_square(
                rotation_mask_tensor, goal_orig_size, goal_padded_size, input_size, patch_size
            )
            if rotation_mask_tensor is not None:
                print(f"  [MASK] Padded mask to match square image "
                      f"({rotation_mask_tensor.sum().item()}/{rotation_mask_tensor.numel()} valid patches)")

        for angle in angles:
            # Rotate the PADDED current image (critical fix for tall/narrow ROIs)
            # Without padding, rotating a 200x400 image by 90° clips the content.
            # With padding to 400x400, rotation preserves all content.
            if angle == 0:
                rotated = current_padded
            else:
                rotated = current_padded.rotate(-angle, expand=False)  # PIL rotates counter-clockwise

            # Resize rotated image to model input size
            rotated_resized = rotated.resize((input_size, input_size))

            # Preprocess using simple transform (same as goal - no resize/crop)
            rotated_tensor = simple_preprocess(rotated_resized).unsqueeze(0).to(extractor.device)

            with torch.no_grad():
                rotated_features = extractor.extract_descriptors(
                    rotated_tensor,
                    layer=11, facet='token', bin=config.use_feature_binning,
                    hierarchy=config.feature_hierarchy
                )

            points1, points2, sims = find_correspondences_batch(
                goal_features, rotated_features,
                num_pairs=num_pairs,
                distance_threshold=1,
                mask_path=rotation_mask_path,
                mask_tensor=rotation_mask_tensor,
                vit_input_size=input_size,
                patch_size=patch_size
            )

            # Handle case where no correspondences found
            if points1 is None or points2 is None or sims is None:
                print(f"  Rotation {angle:3d}°: No correspondences found")
                similarities.append(0.0)
                pair_counts.append(0)
                continue

            # Calculate mean similarity of matched pairs
            n_pairs = len(sims)
            if n_pairs > 0:
                similarity = sims.mean().item() if torch.is_tensor(sims) else np.mean(sims)
            else:
                similarity = 0.0

            similarities.append(similarity)
            pair_counts.append(n_pairs)
            print(f"  Rotation {angle:3d}°: similarity = {similarity:.4f} ({n_pairs} pairs)")

            # Create and publish visualizations
            if len(points1) > 0:
                # Always publish crop-level visualization (reliable ground truth reference)
                if self.viz_crop_pub is not None:
                    self.publish_correspondence_visualization(
                        goal_resized, rotated_resized, points1, points2,
                        input_size, angle, similarity, goal_features
                    )

                # Also publish full-image visualization if in ROI mode
                if (self.viz_full_pub is not None and
                        full_goal_image is not None and full_current_image is not None):
                    if angle == 0:
                        rotated_full = full_current_image
                    else:
                        rotated_full = full_current_image.rotate(-angle, expand=False)
                    self.publish_correspondence_visualization_full(
                        full_goal_image, rotated_full,
                        points1, points2,
                        input_size, angle, similarity, goal_features,
                        goal_roi_bbox, current_roi_bbox,
                        goal_padded_size=goal_padded_size,
                        current_padded_size=current_padded_size
                    )

                time.sleep(0.2)  # Brief pause to allow viewing each rotation

        # Calculate combined scores that weight similarity by relative pair count
        # Formula: combined = similarity * (pairs / max_pairs) ^ alpha
        # This gives higher confidence to rotations with more matched pairs
        max_pairs = max(pair_counts) if pair_counts else 1
        combined_scores = []

        print(f"\n  Pair-weighted scoring (alpha={pair_weight_alpha}):")
        for i, (angle, sim, pairs) in enumerate(zip(angles, similarities, pair_counts)):
            if max_pairs > 0 and pairs > 0:
                pair_weight = (pairs / max_pairs) ** pair_weight_alpha
                combined = sim * pair_weight
            else:
                pair_weight = 0.0
                combined = 0.0
            combined_scores.append(combined)
            print(f"    {angle:3d}°: sim={sim:.4f} × weight={pair_weight:.3f} = combined={combined:.4f}")

        best_idx = np.argmax(combined_scores)
        best_angle = angles[best_idx]
        best_score = combined_scores[best_idx]

        # Also report raw similarity for reference
        print(f"\n  Winner: {best_angle}° (combined={best_score:.4f}, raw_sim={similarities[best_idx]:.4f}, pairs={pair_counts[best_idx]})")

        return best_angle, best_score

    def rotate_robot(self, angle_degrees, config):
        """
        Rotate the robot around the camera optical axis using MoveIt planning.

        Computes a new tool0 pose such that the camera rotates by `angle_degrees`
        around its own optical (Z) axis. This keeps the target object centered
        in the field of view, unlike a pure wrist_3 rotation which would shift
        the view due to the camera mounting offset.

        Args:
            angle_degrees: Target rotation in degrees (positive = CCW in image)
            config: Configuration object
        """
        if angle_degrees == 0:
            return

        from scipy.spatial.transform import Rotation as Rot
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot'))
        from moveit_robot_interface import MoveItRobotInterface

        print(f"\n  Rotating robot by {angle_degrees}° around camera optical axis...")
        angle_rad = np.radians(angle_degrees)

        # Step 1: Switch to scaled_joint_trajectory_controller for MoveIt planning
        print(f"  Switching to trajectory controller...")
        try:
            subprocess.run([
                'ros2', 'control', 'switch_controllers',
                '--deactivate', 'forward_position_controller',
                '--activate', 'scaled_joint_trajectory_controller'
            ], check=True, capture_output=True, text=True, timeout=10)
            print(f"  Controllers switched")
        except subprocess.CalledProcessError as e:
            print(f"  Controller switch note: {e.stderr.strip()}")

        # Step 2: Create temporary MoveIt interface (its own node/executor)
        print(f"  Creating MoveIt interface...")
        robot = MoveItRobotInterface(node_name="rotation_planner", init_rclpy=False)

        try:
            # Step 3: Get current poses from TF
            time.sleep(0.5)  # Let TF populate
            tool0_pose = robot.get_tcp_pose()  # [x, y, z, rx, ry, rz] in base frame
            if tool0_pose is None:
                print(f"  ERROR: Could not get current TCP pose")
                return

            print(f"  Current tool0: pos=[{tool0_pose[0]:.4f}, {tool0_pose[1]:.4f}, {tool0_pose[2]:.4f}]")

            # Get camera optical frame pose in base frame
            from rclpy.time import Time as RclTime
            camera_frame = getattr(config, 'camera_frame', 'camera_color_optical_frame')
            cam_tf = robot.tf_buffer.lookup_transform(
                'base', camera_frame, RclTime(),
                timeout=rclpy.time.Duration(seconds=2.0)
            )
            cam_pos = np.array([
                cam_tf.transform.translation.x,
                cam_tf.transform.translation.y,
                cam_tf.transform.translation.z
            ])
            cam_quat = np.array([
                cam_tf.transform.rotation.x,
                cam_tf.transform.rotation.y,
                cam_tf.transform.rotation.z,
                cam_tf.transform.rotation.w
            ])
            cam_rot = Rot.from_quat(cam_quat)

            # Camera optical Z-axis in base frame (the axis we rotate around)
            optical_z = cam_rot.as_matrix()[:, 2]

            print(f"  Camera pos: [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
            print(f"  Optical Z in base: [{optical_z[0]:.4f}, {optical_z[1]:.4f}, {optical_z[2]:.4f}]")

            # Step 4: Compute rotation around optical axis
            # Rotation matrix for angle_rad around optical_z axis
            delta_rot = Rot.from_rotvec(angle_rad * optical_z)

            # Current tool0 pose
            tool0_pos = np.array(tool0_pose[:3])
            tool0_rot = Rot.from_rotvec(tool0_pose[3:])

            # Rotate tool0 position around camera center (keeps camera center fixed)
            tool0_offset = tool0_pos - cam_pos  # vector from camera to tool0
            new_tool0_offset = delta_rot.apply(tool0_offset)
            new_tool0_pos = cam_pos + new_tool0_offset

            # Rotate tool0 orientation
            new_tool0_rot = delta_rot * tool0_rot
            new_tool0_rotvec = new_tool0_rot.as_rotvec()

            new_pose = [
                new_tool0_pos[0], new_tool0_pos[1], new_tool0_pos[2],
                new_tool0_rotvec[0], new_tool0_rotvec[1], new_tool0_rotvec[2]
            ]
            print(f"  Target tool0: pos=[{new_pose[0]:.4f}, {new_pose[1]:.4f}, {new_pose[2]:.4f}]")

            # Step 5: Execute motion
            print(f"  Planning and executing rotation...")
            success = robot.move_to_tcp_pose(new_pose, speed=0.3, accel=0.3)
            if success:
                print(f"  Motion executed successfully")
            else:
                print(f"  WARNING: Motion may not have completed successfully")

        except Exception as e:
            print(f"  ERROR during rotation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Step 6: Clean up MoveIt interface
            # Brief settle to let pymoveit2 action client finish before destroying
            time.sleep(0.2)
            try:
                robot.destroy()
            except Exception:
                pass  # Executor thread cleanup race — harmless

        # Step 7: Switch back to forward_position_controller for Servo
        print(f"  Switching back to Servo controller...")
        try:
            subprocess.run([
                'ros2', 'control', 'switch_controllers',
                '--deactivate', 'scaled_joint_trajectory_controller',
                '--activate', 'forward_position_controller'
            ], check=True, capture_output=True, text=True, timeout=10)
        except subprocess.CalledProcessError as e:
            print(f"  Controller switch note: {e.stderr.strip()}")

        # Re-enable Servo command type (controller switch resets it)
        try:
            subprocess.run([
                'ros2', 'service', 'call', '/servo_node/switch_command_type',
                'moveit_msgs/srv/ServoCommandType', '{command_type: 1}'
            ], capture_output=True, text=True, timeout=10)
            print(f"  Servo re-enabled")
        except Exception:
            pass

        # Wait for stabilization
        print(f"  Rotation complete. Waiting for stabilization...")
        time.sleep(1.0)

    def _transform_twist(self, twist, transform):
        """Transform a Twist message using a TF2 transform."""
        q = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]

        rot_matrix = _quaternion_matrix(q)

        linear = [twist.linear.x, twist.linear.y, twist.linear.z, 0.0]
        linear_transformed = rot_matrix.dot(linear)

        angular = [twist.angular.x, twist.angular.y, twist.angular.z, 0.0]
        angular_transformed = rot_matrix.dot(angular)

        transformed_twist = Twist()
        transformed_twist.linear.x = linear_transformed[0]
        transformed_twist.linear.y = linear_transformed[1]
        transformed_twist.linear.z = linear_transformed[2]
        transformed_twist.angular.x = angular_transformed[0]
        transformed_twist.angular.y = angular_transformed[1]
        transformed_twist.angular.z = angular_transformed[2]

        return transformed_twist

    @staticmethod
    def predict_post_rotation_center(pre_bbox, best_angle, image_size):
        """
        Predict where an object's bbox center will be after camera rotation.

        When find_best_rotation determines best_angle, the robot rotates by
        -best_angle. The net effect is image content rotates CW by best_angle.

        Args:
            pre_bbox: [x1, y1, x2, y2] bbox before rotation
            best_angle: Discrete rotation angle (0, 90, 180, 270)
            image_size: (width, height) of the image

        Returns:
            (predicted_cx, predicted_cy) tuple
        """
        pre_cx = (pre_bbox[0] + pre_bbox[2]) / 2
        pre_cy = (pre_bbox[1] + pre_bbox[3]) / 2

        img_cx = image_size[0] / 2
        img_cy = image_size[1] / 2

        theta = np.radians(best_angle)
        dx = pre_cx - img_cx
        dy = pre_cy - img_cy

        # CW rotation in image coordinates
        predicted_cx = img_cx + dx * np.cos(theta) + dy * np.sin(theta)
        predicted_cy = img_cy - dx * np.sin(theta) + dy * np.cos(theta)

        return (predicted_cx, predicted_cy)

    @staticmethod
    def _map_crop_points_to_rotated_full(points_518, crop_input_size,
                                          crop_bbox, angle, full_image_size,
                                          padded_size=None):
        """
        Map points from 518x518 rotated-padded crop space to rotated full-image pixel space.

        Understanding the coordinate spaces:
        - Points are extracted from a 518×518 image that was:
          1. Cropped from full image (crop_bbox)
          2. Padded to square (padded_size × padded_size)
          3. Rotated by -angle (PIL convention)
          4. Resized to 518×518
        - The visualization shows:
          - Goal: full image (never rotated)
          - Current: full image rotated by -angle around image center

        The key insight: points in 518 space are in the ROTATED padded crop.
        To map to the rotated full image, we need to:
          1. Scale 518 → padded_size (rotated padded crop space)
          2. Inverse-rotate around padded crop center → unrotated padded crop
          3. Remove padding offset → unrotated original crop
          4. Add bbox offset → unrotated full image
          5. Rotate around full image center → rotated full image

        For angle=0, steps 2 and 5 are identity.

        Args:
            points_518: Nx2 numpy array, pixel coordinates in 518x518 crop space (x, y)
            crop_input_size: int, the resize target (518)
            crop_bbox: [x1, y1, x2, y2] bbox of the crop in the UNROTATED full image
            angle: rotation angle in degrees (0, 90, 180, 270)
            full_image_size: (width, height) of the full image
            padded_size: Side length of the padded square (None if no padding was applied)

        Returns:
            Nx2 numpy array, pixel coordinates in the rotated full image (x, y)
        """
        x1, y1, x2, y2 = [int(c) for c in crop_bbox]
        crop_w = x2 - x1
        crop_h = y2 - y1
        img_w, img_h = full_image_size

        # Determine padded size
        if padded_size is None:
            padded_size = max(crop_w, crop_h)

        # Calculate padding offsets (original crop was centered in padded square)
        pad_offset_x = (padded_size - crop_w) / 2.0
        pad_offset_y = (padded_size - crop_h) / 2.0

        # Step 1: Scale from 518 to padded_size
        # Points are in rotated-padded space
        p_rotated_padded = points_518 * (padded_size / crop_input_size)

        if angle == 0:
            # No rotation - just remove padding and add bbox offset
            p_unpadded = p_rotated_padded.copy()
            p_unpadded[:, 0] -= pad_offset_x
            p_unpadded[:, 1] -= pad_offset_y
            p_full = p_unpadded + np.array([x1, y1])
            return p_full

        # For non-zero angles:
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Step 2: Inverse-rotate around padded crop center
        # The crop was rotated using crop.rotate(-angle). PIL.rotate(-angle) applies
        # a visual clockwise rotation, which in image coords is a CCW rotation by +angle.
        # To UNDO this, we need the INVERSE: CCW by -angle (= CW by +angle in image coords).
        # Inverse formula: x' = cx + dx*cos(a) + dy*sin(a)
        #                  y' = cy - dx*sin(a) + dy*cos(a)
        pcx, pcy = padded_size / 2.0, padded_size / 2.0
        dx = p_rotated_padded[:, 0] - pcx
        dy = p_rotated_padded[:, 1] - pcy
        # Inverse rotation to undo the crop rotation
        p_unrot_padded_x = pcx + dx * cos_a + dy * sin_a
        p_unrot_padded_y = pcy - dx * sin_a + dy * cos_a

        # Step 3: Remove padding offset → coordinates in unrotated original crop
        p_unrot_crop_x = p_unrot_padded_x - pad_offset_x
        p_unrot_crop_y = p_unrot_padded_y - pad_offset_y

        # Step 4: Add bbox offset → coordinates in unrotated full image
        p_unrot_full_x = p_unrot_crop_x + x1
        p_unrot_full_y = p_unrot_crop_y + y1

        # Step 5: Rotate around full image center → coordinates in rotated full image
        # The visualization does full_img.rotate(-angle). PIL.rotate(-angle) applies
        # a visual clockwise rotation, which in image coords (+y down) is actually
        # a COUNTER-CLOCKWISE rotation by +angle mathematically.
        # Forward CCW formula: x' = cx + dx*cos(a) - dy*sin(a)
        #                      y' = cy + dx*sin(a) + dy*cos(a)
        icx, icy = img_w / 2.0, img_h / 2.0
        fdx = p_unrot_full_x - icx
        fdy = p_unrot_full_y - icy
        # Forward CCW rotation (matches PIL.rotate(-angle) behavior)
        p_rot_full_x = icx + fdx * cos_a - fdy * sin_a
        p_rot_full_y = icy + fdx * sin_a + fdy * cos_a

        return np.stack([p_rot_full_x, p_rot_full_y], axis=-1)

    @staticmethod
    def _rotate_bbox(bbox, angle, image_size):
        """Rotate a bbox's 4 corners and return their positions in the rotated image."""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)

        if angle == 0:
            return corners

        img_w, img_h = image_size
        cx, cy = img_w / 2.0, img_h / 2.0
        angle_rad = np.radians(angle)  # PIL.rotate(-angle) internal = radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        dx = corners[:, 0] - cx
        dy = corners[:, 1] - cy
        # Forward CCW rotation (matches PIL.rotate(-angle) behavior in image coords)
        rot_x = cx + dx * cos_a - dy * sin_a
        rot_y = cy + dx * sin_a + dy * cos_a

        return np.stack([rot_x, rot_y], axis=-1)

    def run(self):
        """One-shot entry point: load models, run a single servo cycle, exit."""

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        # Initialize ROS2
        rclpy.init()
        self.node = Node('visual_servoing_real_robot')

        try:
            self._run_inner()
        finally:
            if hasattr(self, 'node') and self.node is not None:
                self.node.destroy_node()
            rclpy.shutdown()

    def run_worker(self):
        """Persistent-worker entry point: load models ONCE, then run one servo cycle
        per 'SERVO' line received on stdin. Avoids the ~30s model reload per pick.

        Driven by run_multi_pick_demo.py --warm. Protocol:
          stdout sentinels: 'WORKER_READY' (once, after models load),
                            'PICK_DONE' (after each cycle, following the EVAL_ lines)
          stdin commands:   'SERVO' (run one cycle) | 'QUIT' (exit)

        Each cycle uses a FRESH ROS node/executor/subscriptions (cheap) but reuses the
        cached DINOv3 extractor + LangSAM detector (the expensive, ROS-independent parts).
        """
        import sys as _sys
        self._worker_mode = True

        signal.signal(signal.SIGINT, self.signal_handler)
        rclpy.init()
        print("WORKER_READY", flush=True)

        try:
            for raw in _sys.stdin:
                cmd = raw.strip().upper()
                if cmd in ("QUIT", "EXIT"):
                    break
                if cmd != "SERVO":
                    continue

                # Reset per-cycle state; models stay cached on self.
                self.running = True
                self.latest_image = None
                self._latest_depth = None
                self.node = Node('visual_servoing_real_robot')
                try:
                    res = self._run_inner()
                    converged, iterations = res if res else (False, 0)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print("EVAL_CONVERGED: False")
                    print("EVAL_ITERATIONS: 0")
                finally:
                    # Stop this cycle's executor + spin thread, then drop the node so the
                    # next cycle starts with a clean ROS graph (no accumulated entities).
                    try:
                        if getattr(self, '_executor', None) is not None:
                            self._executor.shutdown()
                    except Exception:
                        pass
                    try:
                        if getattr(self, '_spin_thread', None) is not None:
                            self._spin_thread.join(timeout=2.0)
                    except Exception:
                        pass
                    try:
                        if self.node is not None:
                            self.node.destroy_node()
                    except Exception:
                        pass
                    self.node = None
                    self._executor = None
                    self._spin_thread = None

                print("PICK_DONE", flush=True)
        finally:
            rclpy.shutdown()

    def _run_inner(self):
        """Inner run logic, called within rclpy lifecycle."""
        print("\n" + "="*70)
        print("Real Robot Visual Servoing")
        print("="*70)

        # Load configuration
        print(f"\nLoading config: {self.config_path}")
        config = Config(self.config_path)
        self.config = config

        # Verify real robot mode
        robot_mode = getattr(config, 'robot_mode', 'simulation')
        if robot_mode != 'real':
            self.node.get_logger().warn(f"Config robot_mode is '{robot_mode}', expected 'real'")
            self.node.get_logger().warn("Proceeding anyway, but TF-based pose acquisition may fail in simulation")

        print(f"  Robot mode: {robot_mode}")
        print(f"  Backbone model: {config.backbone_model}")
        print(f"  VIT input size: {config.vit_input_size}")
        print(f"  Convergence mode: {getattr(config, 'convergence_mode', 'visual_only')}")

        # Initialize TF2
        print("\nInitializing TF2...")
        self.tf_buffer = TF2Buffer()
        self.tf_listener = TF2TransformListener(self.tf_buffer, self.node)

        # Start background executor so all callbacks (image, TF, timers) fire continuously.
        # This replaces manual spin_once calls throughout the code.
        from rclpy.executors import MultiThreadedExecutor
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Wait for TF buffer to populate via background executor
        time.sleep(1.5)

        # Setup velocity publisher (TwistStamped for servo node)
        velocity_topic = getattr(config, 'velocity_topic', '/servo_node/delta_twist_cmds')
        self.velocity_pub = self.node.create_publisher(TwistStamped, velocity_topic, 10)
        print(f"  Publishing velocities to: {velocity_topic}")

        # Setup visualization publishers for rotation alignment
        self.viz_crop_pub = self.node.create_publisher(ImageMsg, '/correspondence_visualization_crop', 1)
        self.viz_full_pub = self.node.create_publisher(ImageMsg, '/correspondence_visualization', 1)
        print(f"  Publishing crop visualizations to: /correspondence_visualization_crop")
        print(f"  Publishing full-image visualizations to: /correspondence_visualization")

        # Setup image subscriber for rotation alignment
        rgb_topic = getattr(config, 'camera_rgb_topic', '/camera/color/image_cropped')
        self.image_sub = self.node.create_subscription(ImageMsg, rgb_topic, self.image_callback, 1)
        print(f"  Subscribing to images: {rgb_topic}")

        # Subscribe to depth EARLY (while executor is running) so DDS discovery
        # completes before we pause the executor.  The controller's own depth
        # subscription (created during executor pause) won't be discovered by DDS,
        # so this early sub feeds depth frames to the controller once it exists.
        depth_topic = getattr(config, 'camera_depth_topic', '/camera/depth/image_raw')
        self._latest_depth = None
        self._depth_controller_ref = None  # Set after controller creation
        self._early_depth_bridge = CvBridge()
        def _depth_cb(msg):
            try:
                depth = self._early_depth_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                self._latest_depth = depth
                # Forward to controller if it exists
                if self._depth_controller_ref is not None:
                    self._depth_controller_ref.latest_image_depth = depth
            except Exception as e:
                self.node.get_logger().error(f"Early depth callback error: {e}")
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        _depth_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST)
        self._early_depth_sub = self.node.create_subscription(ImageMsg, depth_topic, _depth_cb, _depth_qos)
        print(f"  Subscribing to depth: {depth_topic}")

        # Wait for first image (background executor handles callbacks)
        print("\nWaiting for first camera image...")
        deadline = time.time() + 10.0
        while self.latest_image is None and time.time() < deadline and self.running:
            time.sleep(0.1)

        if self.latest_image is None:
            self.node.get_logger().error("Timeout waiting for camera image!")
            return
        print(f"  Image received: {self.latest_image.size}")

        # Get initial camera pose from TF
        print("\nGetting initial camera pose from TF...")
        position, orientation = self.get_camera_pose_tf(config)
        if position is None:
            self.node.get_logger().error("Failed to get camera pose from TF!")
            return
        print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print(f"  Orientation: [{orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}, {orientation[3]:.3f}]")

        # For real robot, use current pose as "desired pose"
        # Visual-only convergence doesn't use pose error, just feature error
        desired_position = position.copy()
        desired_orientation = orientation.copy()

        # Activate MoveIt Servo BEFORE model loading, so rotation alignment
        # can publish twist commands. Servo requires:
        # (1) forward_position_controller active, (2) command_type=1 (TwistStamped).
        if getattr(config, 'robot_mode', 'simulation') == 'real':
            print("\nActivating MoveIt Servo for twist commands...")
            try:
                subprocess.run([
                    'ros2', 'control', 'switch_controllers',
                    '--deactivate', 'scaled_joint_trajectory_controller',
                    '--activate', 'forward_position_controller'
                ], check=True, capture_output=True, text=True, timeout=10)
                print("  Controllers switched (forward_position_controller active)")
            except subprocess.CalledProcessError as e:
                print(f"  Controller switch note: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                print("  WARNING: Controller switch timed out")

            try:
                result = subprocess.run([
                    'ros2', 'service', 'call', '/servo_node/switch_command_type',
                    'moveit_msgs/srv/ServoCommandType', '{command_type: 1}'
                ], check=True, capture_output=True, text=True, timeout=10)
                print(f"  Servo twist commands enabled")
            except subprocess.CalledProcessError as e:
                print(f"  WARNING: Servo command type switch failed: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                print(f"  WARNING: Servo command type switch timed out")

        # Pause executor during model loading to avoid GIL contention.
        # The background executor processes TF callbacks at 100+ Hz, and each
        # callback grabs the GIL. This competes with torch model loading on the
        # main thread, turning a 2s load into 14+ seconds (or infinite hang).
        # We have the image and TF pose we need, so it's safe to pause.
        print("\nPausing executor for model loading...")
        self._executor.remove_node(self.node)

        # Create feature extractor
        print(f"\nInitializing feature extractor ({config.backbone_model})...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")

        # Check if classical method (SIFT/ORB/AKAZE)
        is_classical = 'classical-' in config.backbone_model

        # Reuse a cached extractor in worker mode (avoids reloading the ViT every pick).
        if self._cached_extractor is not None:
            extractor = self._cached_extractor
            print(f"  Feature extractor ready (reused cached model)")
        else:
            extractor = MultiBackboneViTExtractor(
                model_name=config.backbone_model,
                device=device,
                input_size=config.vit_input_size
            )
            self._cached_extractor = extractor
            print(f"  Feature extractor ready")

        # Load goal image for rotation alignment (cache — never changes between picks).
        if self._cached_goal_image is not None:
            goal_image = self._cached_goal_image
        else:
            goal_image = Image.open(config.image_path).convert('RGB')
            self._cached_goal_image = goal_image
            print(f"\nGoal image loaded: {config.image_path}")
            print(f"  Size: {goal_image.size}")

        # ================================================================
        # PHASE 1: Early detector creation (before rotation)
        # ================================================================
        early_detector = None
        goal_roi_bbox = None

        if not is_classical and config.use_roi_detection:
            detector_type = getattr(config, 'detector_type', 'yoloworld')
            if detector_type == 'langsam_lighttrack' and LANGSAM_AVAILABLE:
                if self._cached_detector is not None:
                    # Reuse loaded LangSAM (GroundingDINO+SAM2 stay on GPU); just reset
                    # per-run tracking state so it re-detects this pick's scene fresh.
                    early_detector = self._cached_detector
                    early_detector.reset_tracking()
                    print("\n  Reusing cached LangSAM+LightTrack detector (tracking reset)")
                else:
                    print("\n  Creating LangSAM+LightTrack detector early...")
                    try:
                        early_detector = LangSAMLightTrackDetector(
                            lighttrack_weights=config.lighttrack_weights,
                            lighttrack_arch=config.lighttrack_arch,
                            confidence_threshold=config.yolo_confidence_threshold,
                            cache_timeout=config.cache_timeout,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        self._cached_detector = early_detector
                        print(f"  Detector ready")
                    except Exception as e:
                        print(f"  WARNING: Failed to create early detector: {e}")
                        early_detector = None

        # ================================================================
        # PHASE 2: Detect ROI on goal + current images
        # ================================================================
        current_roi_bbox = None
        pre_rotation_center = None
        use_roi_for_rotation = False

        if early_detector is not None:
            detection_keyword = getattr(config, 'detection_keyword', None)
            padding_ratio = getattr(config, 'roi_padding_ratio', 0.1)

            if detection_keyword:
                print(f"\n" + "-"*70)
                print("EARLY ROI DETECTION (for rotation compensation)")
                print("-"*70)

                # Detect goal ROI (once — goal image is constant across picks).
                if self._cached_goal_roi_bbox is not None:
                    goal_roi_bbox = self._cached_goal_roi_bbox or None
                    print(f"  Goal ROI (cached): {goal_roi_bbox}")
                else:
                    print(f"  Detecting goal ROI (keyword: '{detection_keyword}')...")
                    goal_detection = early_detector.detect_and_cache_goal(
                        goal_image, detection_keyword, padding_ratio
                    )
                    if goal_detection['bbox'] is not None:
                        goal_roi_bbox = goal_detection['padded_bbox']
                        self._cached_goal_roi_bbox = goal_roi_bbox
                        print(f"  Goal ROI: {goal_roi_bbox}")
                    else:
                        self._cached_goal_roi_bbox = False
                        print(f"  WARNING: No goal ROI detected")

                # Detect current ROI
                print(f"  Detecting current ROI...")
                current_detection = early_detector.detect(
                    self.latest_image, keyword=detection_keyword,
                    padding_ratio=padding_ratio
                )
                if current_detection['bbox'] is not None:
                    current_roi_bbox = current_detection['padded_bbox']
                    # Save pre-rotation center for spatial prior after rotation re-detection.
                    # With multiple same-category objects, this ensures we re-lock onto the
                    # SAME instance after rotation compensation (not a neighbor).
                    bbox_unpadded = current_detection['bbox']
                    pre_rotation_center = (
                        (bbox_unpadded[0] + bbox_unpadded[2]) / 2,
                        (bbox_unpadded[1] + bbox_unpadded[3]) / 2
                    )
                    print(f"  Current ROI: {current_roi_bbox}")
                    print(f"  Pre-rotation center: ({pre_rotation_center[0]:.1f}, {pre_rotation_center[1]:.1f})")
                else:
                    print(f"  WARNING: No current ROI detected")

                use_roi_for_rotation = (goal_roi_bbox is not None and current_roi_bbox is not None)
            else:
                print("\n  No detection_keyword configured, skipping early ROI detection")

        # ================================================================
        # PHASE 3: Prepare ROI crops + cropped mask for rotation
        # ================================================================
        # Snapshot the current image NOW so it stays consistent with the
        # current_roi_bbox detected in Phase 2 (self.latest_image keeps updating
        # from the camera callback, so without this the full image could be a
        # different frame than the one the bbox was computed on).
        current_image_snapshot = self.latest_image.copy()
        rotation_goal = goal_image
        rotation_current = current_image_snapshot
        rotation_mask_tensor = None

        if use_roi_for_rotation:
            print(f"\n  Preparing ROI crops for rotation alignment...")
            print(f"  Goal image: {goal_image.size}, Current image snapshot: {current_image_snapshot.size}")

            # Crop images to ROI
            rotation_goal = goal_image.crop(tuple(int(x) for x in goal_roi_bbox))
            rotation_current = current_image_snapshot.crop(tuple(int(x) for x in current_roi_bbox))
            print(f"  Goal ROI crop: {rotation_goal.size}, Current ROI crop: {rotation_current.size}")

            # Crop mask to goal ROI (mask must still be used on cropped ROI)
            # Note: config.mask_path is already resolved to absolute by Config class
            mask_path = getattr(config, 'mask_path', None)
            if mask_path and os.path.exists(mask_path):
                full_mask = Image.open(mask_path).convert('L')
                x1, y1, x2, y2 = (int(x) for x in goal_roi_bbox)
                cropped_mask = full_mask.crop((x1, y1, x2, y2))

                rotation_input_size = 518  # Same as find_best_rotation uses
                patch_size = 16 if 'dinov3' in config.backbone_model else 14
                rotation_mask_tensor = create_patch_mask_from_pil(
                    cropped_mask, vit_input_size=rotation_input_size, patch_size=patch_size
                )
                print(f"  Mask cropped to ROI -> tensor "
                      f"({rotation_mask_tensor.sum().item()}/{rotation_mask_tensor.numel()} valid patches)")
            elif mask_path:
                print(f"  WARNING: Mask not found: {mask_path}")

        # ================================================================
        # PHASE 4: Rotation alignment
        # ================================================================
        best_angle = 0
        if not self.skip_rotation and not is_classical:
            print("\n" + "-"*70)
            print("ROTATION ALIGNMENT")
            print("-"*70)

            if use_roi_for_rotation:
                print("Using CROPPED ROI images (background eliminated)")
            else:
                print("Using FULL images (no ROI available)")

            print("Testing 4 discrete rotations (0, 90, 180, 270 degrees)...")

            best_angle, best_score = self.find_best_rotation(
                extractor, rotation_goal, rotation_current, config,
                mask_tensor=rotation_mask_tensor,
                full_goal_image=goal_image if use_roi_for_rotation else None,
                full_current_image=current_image_snapshot if use_roi_for_rotation else None,
                goal_roi_bbox=goal_roi_bbox if use_roi_for_rotation else None,
                current_roi_bbox=current_roi_bbox if use_roi_for_rotation else None
            )

            print(f"\n  Best rotation: {best_angle}° (score: {best_score:.4f})")

            if best_angle != 0:
                # Negate the angle: find_best_rotation returns the image rotation needed,
                # but robot rotation direction is opposite to image rotation direction.
                robot_angle = -best_angle

                # Normalize to [-180, 180] to use shorter rotation path
                if robot_angle > 180:
                    robot_angle -= 360
                elif robot_angle < -180:
                    robot_angle += 360

                # Executor stays paused — publish() works without it,
                # and running the executor causes GIL starvation in sleep().
                self.rotate_robot(robot_angle, config)
                print(f"  Robot rotated by {robot_angle}° to align with goal image")
            else:
                print(f"  No rotation needed - already aligned")
        elif self.skip_rotation:
            print("\n  Rotation alignment skipped (--skip-rotation flag)")
        elif is_classical:
            print("\n  Rotation alignment skipped (classical methods don't use rotation compensation)")

        # ================================================================
        # PHASE 5: Post-rotation re-detection with spatial prior
        # ================================================================
        if early_detector is not None and current_roi_bbox is not None:
            detection_keyword = getattr(config, 'detection_keyword', None)
            padding_ratio = getattr(config, 'roi_padding_ratio', 0.1)

            if best_angle != 0:
                print(f"\n  Re-detecting after rotation...")

                # Resume executor briefly to get fresh image after rotation.
                self._executor.add_node(self.node)
                time.sleep(0.5)  # Wait for fresh frame after rotation
                self._executor.remove_node(self.node)

                # Transform spatial prior to post-rotation coordinates.
                # After robot rotates by -best_angle, the image effectively rotates
                # by best_angle around its center. We must rotate the pre-rotation
                # detection center by the same angle so the prior points to the
                # correct instance in the post-rotation image.
                post_rotation_prior = None
                if pre_rotation_center is not None:
                    import math
                    img_w, img_h = self.latest_image.size  # (1440, 1080)
                    img_cx, img_cy = img_w / 2, img_h / 2
                    dx = pre_rotation_center[0] - img_cx
                    dy = pre_rotation_center[1] - img_cy
                    theta = math.radians(best_angle)
                    new_dx = dx * math.cos(theta) - dy * math.sin(theta)
                    new_dy = dx * math.sin(theta) + dy * math.cos(theta)
                    post_rotation_prior = (img_cx + new_dx, img_cy + new_dy)
                    print(f"  Spatial prior: ({pre_rotation_center[0]:.1f}, {pre_rotation_center[1]:.1f}) "
                          f"-> rotated {best_angle}° -> ({post_rotation_prior[0]:.1f}, {post_rotation_prior[1]:.1f})")

                # Fresh LangSAM detection on the post-rotation image
                early_detector.state = 'DETECT'
                early_detector.is_tracking = False

                post_detection = early_detector.detect(
                    self.latest_image, keyword=detection_keyword,
                    padding_ratio=padding_ratio,
                    spatial_prior=post_rotation_prior
                )

                if post_detection['bbox'] is not None:
                    post_bbox = post_detection['bbox']
                    post_center = ((post_bbox[0] + post_bbox[2]) / 2, (post_bbox[1] + post_bbox[3]) / 2)
                    print(f"  Detected object at: center=({post_center[0]:.1f}, {post_center[1]:.1f}), "
                          f"bbox={[int(x) for x in post_bbox]}")

                    early_detector.start_tracking(self.latest_image, post_detection['bbox'])
                    print(f"  LightTrack re-initialized on post-rotation frame")
                else:
                    print(f"  WARNING: No detection after rotation, tracker not initialized")
            else:
                # No rotation needed - initialize tracker on current frame
                current_det = early_detector.detect(
                    self.latest_image, keyword=detection_keyword,
                    padding_ratio=padding_ratio,
                    spatial_prior=pre_rotation_center
                )
                if current_det['bbox'] is not None:
                    early_detector.start_tracking(self.latest_image, current_det['bbox'])
                    print(f"  LightTrack initialized (no rotation needed)")

        # ================================================================
        # PHASE 6: Create controller with pre-initialized detector
        # ================================================================
        self.node.destroy_subscription(self.image_sub)

        print("\n" + "-"*70)
        print("VISUAL SERVOING")
        print("-"*70)
        print("Creating visual servoing controller...")

        controller = ROSVisualServoingController(
            config=config,
            feature_extractor=extractor,
            desired_position=desired_position,
            desired_orientation=desired_orientation,
            pre_initialized_detector=early_detector,
            pre_detected_goal_roi_bbox=goal_roi_bbox,
            node=self.node,
            tf_buffer=self.tf_buffer,
            executor=self._executor
        )

        # The controller created its own depth subscription, but it was made while
        # the node was detached from the executor — DDS never discovers it.
        # Destroy it and use the EARLY depth subscription instead (created before
        # the executor pause, so DDS already matched publisher → subscriber).
        self.node.destroy_subscription(controller.image_sub_depth)
        controller.image_sub_depth = self._early_depth_sub
        # Wire the early depth callback to feed the controller
        self._depth_controller_ref = controller
        # Seed with any depth frame already received
        if self._latest_depth is not None:
            controller.latest_image_depth = self._latest_depth
            print(f"  Depth seeded from early subscription: {self._latest_depth.shape}")

        # Resume executor AFTER controller init — all heavy model/feature processing
        # is done, so TF callbacks at 100+ Hz won't cause GIL contention anymore.
        print("\nResuming executor for visual servoing loop...")
        self._executor.add_node(self.node)

        # Wait for DDS to discover new subscriptions and deliver first messages.
        print("Waiting for RGB + depth images...")
        _wait_start = time.time()
        _depth_ok = False
        while time.time() - _wait_start < 10.0:
            if controller.latest_image is not None and not _depth_ok:
                if controller.latest_image_depth is not None:
                    _depth_ok = True
            if controller.latest_image is not None and _depth_ok:
                break
            time.sleep(0.1)
        _elapsed = time.time() - _wait_start
        if controller.latest_image is not None and _depth_ok:
            print(f"  RGB + depth ready in {_elapsed:.1f}s")
        else:
            rgb_status = "OK" if controller.latest_image is not None else "MISSING"
            depth_status = "OK" if controller.latest_image_depth is not None else "MISSING"
            print(f"  WARNING: After {_elapsed:.1f}s — RGB={rgb_status}, Depth={depth_status}")
            if depth_status == "MISSING":
                # Debug: list topics visible to this node
                print(f"  Depth topic: {getattr(config, 'camera_depth_topic', '?')}")
                print(f"  Node subscriptions: {[s.topic_name for s in self.node.subscriptions]}")

        # Servo already activated earlier (before model loading / rotation).

        # Run visual servoing
        print("\nStarting visual servoing loop...")
        print("Press Ctrl+C to stop\n")

        converged = False
        iterations = 0
        try:
            result = controller.run()

            if result is not None:
                print("\n" + "="*70)
                print("VISUAL SERVOING COMPLETE")
                print("="*70)
                converged = result[2] if len(result) > 2 else False
                print(f"  Converged: {converged}")
                iterations = result[7] if len(result) > 7 else 0
                print(f"  Iterations: {iterations}")

                # Explicit stdout prints for evaluation script parsing
                # These use a specific EVAL_ prefix that the evaluation script looks for
                # Important: These are print() not rospy.loginfo() so they go to stdout
                print(f"EVAL_CONVERGED: {converged}")
                print(f"EVAL_ITERATIONS: {iterations}")
            else:
                print("\n  Visual servoing returned None (may have failed)")
                # Still emit EVAL_ lines for evaluation script
                print("EVAL_CONVERGED: False")
                print("EVAL_ITERATIONS: 0")

        except Exception as e:
            self.node.get_logger().error(f"Error during visual servoing: {e}")
            import traceback
            traceback.print_exc()
            # Emit EVAL_ lines on error
            print("EVAL_CONVERGED: False")
            print("EVAL_ITERATIONS: 0")
        finally:
            # Ensure robot is stopped
            self.stop_robot()
            # In worker mode, tear down everything this controller put on the node
            # (100Hz twist thread + pubs/subs) so the next pick starts clean.
            if self._worker_mode:
                try:
                    controller.cleanup()
                except Exception as e:
                    print(f"  Controller cleanup warning: {e}")
            print("\nCleanup complete.")

        return converged, iterations


def main():
    parser = argparse.ArgumentParser(
        description='Run visual servoing on real UR5 robot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with DINOv3 config
    python3 run_visual_servoing.py --config configs/real_robot/config_real_robot_dinov3.yaml

    # Run with SIFT (classical method)
    python3 run_visual_servoing.py --config configs/real_robot/config_real_robot_sift.yaml

    # Skip rotation alignment
    python3 run_visual_servoing.py --config configs/real_robot/config_real_robot_dinov3.yaml --skip-rotation

Prerequisites (ROS2 Jazzy, see docs/REAL_ROBOT.md for the full startup sequence):
    1. Robot driver:  ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=<ROBOT_IP> launch_rviz:=false
    2. MoveIt+Servo:  ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_servo:=true
    3. Camera:        ../robot/start_camera.sh
    4. Image cropper: python3 ../robot/crop_image_node.py
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--skip-rotation',
        action='store_true',
        help='Skip rotation alignment phase'
    )

    parser.add_argument(
        '--worker',
        action='store_true',
        help='Persistent-worker mode: load models once, run one servo cycle per "SERVO" '
             'line on stdin (used by run_multi_pick_demo.py --warm). Avoids per-pick reload.'
    )

    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Run visual servoing
    vs = RealRobotVisualServoing(args.config, skip_rotation=args.skip_rotation)
    if args.worker:
        vs.run_worker()
    else:
        vs.run()


if __name__ == "__main__":
    main()
