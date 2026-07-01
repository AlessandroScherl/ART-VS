#!/usr/bin/env python3
"""
Fixed rotation finder that uses the same feature extraction as visual servoing.
Uses existing methods from feature extractors instead of reinventing.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)
from features.matcher import find_correspondences_batch


class UnifiedRotationFinder:
    """
    Finds optimal rotation alignment using either discrete or continuous search.
    Works by virtually rotating images and comparing features.
    Uses the same feature extraction methods as visual servoing.
    """
    
    def __init__(self, feature_extractor, config, ros_controller=None):
        """
        Initialize rotation finder.
        
        Args:
            feature_extractor: The feature extractor instance (e.g., ViTExtractor or MultiBackboneExtractor)
            config: Configuration object with rotation parameters
            ros_controller: Optional ROSVisualServoingController for visualization
        """
        self.feature_extractor = feature_extractor
        self.config = config
        self.device = feature_extractor.device
        self.ros_controller = ros_controller
        
        # Rotation mode: 'discrete' or 'continuous'
        self.rotation_mode = getattr(config, 'rotation_mode', 'discrete')
        
        # Parameters for continuous mode - can be configured for speed vs accuracy
        # Default values (accurate but slow)
        self.coarse_step = getattr(config, 'rotation_coarse_step', 5.0)  # Coarse search step
        self.fine_range = getattr(config, 'rotation_fine_range', 10.0)   # Fine search ±range
        self.fine_step = getattr(config, 'rotation_fine_step', 0.5)      # Fine search step
        
        # Log the parameters being used
        if self.rotation_mode == 'continuous':
            logger.info(f"Continuous rotation parameters: coarse_step={self.coarse_step}°, "
                         f"fine_range=±{self.fine_range}°, fine_step={self.fine_step}°")
            total_coarse = 360 / self.coarse_step
            total_fine = 2 * self.fine_range / self.fine_step
            logger.info(f"Estimated evaluations: ~{int(total_coarse + 3*total_fine)} "
                         f"(coarse: {int(total_coarse)}, fine: ~{int(3*total_fine)})")
        
    def rotate_image(self, image: Image.Image, angle: float) -> Image.Image:
        """Rotate image by angle degrees around center."""
        return image.rotate(-angle, expand=False, fillcolor=(255, 255, 255))
    
    def compute_similarity_like_vs(self, goal_image: Image.Image, current_image: Image.Image) -> float:
        """
        Compute similarity exactly like visual servoing does.
        This mimics detect_features() -> mean similarity flow.
        """
        # Store current config settings
        original_num_pairs = self.config.num_pairs
        original_use_roi = getattr(self.config, 'use_roi_detection', False)
        original_use_roi_tiling = getattr(self.config, 'use_roi_tiling', False)
        original_use_tiling = getattr(self.config, 'use_tiling', False)
        
        # Temporarily adjust settings for rotation search
        self.config.num_pairs = 48  # More pairs for robust matching
        self.config.use_roi_detection = False  # Always use full image
        self.config.use_roi_tiling = False
        self.config.use_tiling = False  # Disable tiling for rotation search
        
        try:
            # Resize images as in visual servoing
            vit_input_size = self.config.vit_input_size
            goal_resized = goal_image.resize((vit_input_size, vit_input_size))
            current_resized = current_image.resize((vit_input_size, vit_input_size))
            
            with torch.no_grad():
                # Preprocess images and move to device
                goal_tensor = self.feature_extractor.preprocess_pil(goal_resized).to(self.device)
                current_tensor = self.feature_extractor.preprocess_pil(current_resized).to(self.device)
                
                # Extract features using the SAME method as visual servoing
                # Extract features using the same method as visual servoing
                desc1 = self.feature_extractor.extract_descriptors(
                    goal_tensor,
                    layer=11,  # This parameter is ignored for non-DINOv2 models
                    facet='token',  # Use 'token' instead of 'key' for consistency
                    bin=self.config.use_feature_binning
                )
                desc2 = self.feature_extractor.extract_descriptors(
                    current_tensor,
                    layer=11,  # This parameter is ignored for non-DINOv2 models
                    facet='token',  # Use 'token' instead of 'key' for consistency
                    bin=self.config.use_feature_binning
                )
                
                # Find correspondences as in visual servoing
                # Use goal mask if available (only masks goal image features)
                goal_mask_path = getattr(self.config, 'mask_path', None)
                if goal_mask_path and os.path.exists(goal_mask_path):
                    pass  # Mask will be used in find_correspondences_batch
                else:
                    goal_mask_path = None
                    
                points1, points2, sim_selected_12 = find_correspondences_batch(
                    desc1, desc2,
                    num_pairs=self.config.num_pairs,
                    mask_path=goal_mask_path,  # Apply mask to goal features only
                    vit_input_size=self.config.vit_input_size,
                    patch_size=self.feature_extractor.get_patch_size()
                )
                
                # Return mean similarity (this is what best_pose_finder checks)
                if sim_selected_12 is not None and len(sim_selected_12) > 0:
                    mean_similarity = sim_selected_12.mean().item()
                else:
                    mean_similarity = 0.0
                
        finally:
            # Restore original settings
            self.config.num_pairs = original_num_pairs
            self.config.use_roi_detection = original_use_roi
            if hasattr(self.config, 'use_roi_tiling'):
                self.config.use_roi_tiling = original_use_roi_tiling
            if hasattr(self.config, 'use_tiling'):
                self.config.use_tiling = original_use_tiling
        
        return mean_similarity
    
    def visualize_rotation_correspondences_simple(self, goal_image: Image.Image, current_image: Image.Image, 
                                                  angle: float, similarity: float, points1=None, points2=None):
        """Simple visualization that shows images with correspondences, matching VS visualization style."""
        if not getattr(self.config, 'enable_visualization', True) or self.ros_controller is None:
            return
            
        try:
            # Use the standard visualization function with computed correspondences
            from vs_utils.visualization import visualize_correspondences_ros
            
            # Rotate current image
            rotated_image = self.rotate_image(current_image, angle)
            
            # If we don't have points, compute them
            if points1 is None or points2 is None:
                # Get correspondences for this rotation
                vit_input_size = self.config.vit_input_size
                goal_resized = goal_image.resize((vit_input_size, vit_input_size))
                rotated_resized = rotated_image.resize((vit_input_size, vit_input_size))
                
                with torch.no_grad():
                    # Preprocess images
                    goal_tensor = self.feature_extractor.preprocess_pil(goal_resized).to(self.device)
                    rotated_tensor = self.feature_extractor.preprocess_pil(rotated_resized).to(self.device)
                    
                    # Extract features using the same method as above
                    desc1 = self.feature_extractor.extract_descriptors(
                        goal_tensor, 
                        layer=11,  # Ignored for non-DINOv2
                        facet='token',  # Changed from 'key' to 'token' for consistency
                        bin=self.config.use_feature_binning
                    )
                    desc2 = self.feature_extractor.extract_descriptors(
                        rotated_tensor, 
                        layer=11,  # Ignored for non-DINOv2
                        facet='token',  # Changed from 'key' to 'token' for consistency
                        bin=self.config.use_feature_binning
                    )
                    
                    # Use goal mask if available
                    goal_mask_path = getattr(self.config, 'mask_path', None)
                    if goal_mask_path and os.path.exists(goal_mask_path):
                        pass  # Mask will be used in find_correspondences_batch
                    else:
                        goal_mask_path = None
                        
                    # Find correspondences with error handling
                    # Check if descriptors are valid
                    if desc1.shape[2] == 0 or desc2.shape[2] == 0:
                        logger.warning(f"Empty descriptors: desc1 patches={desc1.shape[2]}, desc2 patches={desc2.shape[2]}")
                        points1 = points2 = None
                    else:
                        try:
                            points1, points2, sims = find_correspondences_batch(
                                desc1, desc2,
                                num_pairs=min(self.config.num_pairs, 16),
                                mask_path=goal_mask_path,
                                vit_input_size=self.config.vit_input_size,
                                patch_size=self.feature_extractor.get_patch_size()
                            )
                        except Exception as e:
                            logger.warning(f"Error in find_correspondences_batch: {e}")
                            points1 = points2 = None
                    
                    if points1 is None or points2 is None or len(points1) == 0:
                        # No matches - create empty arrays
                        points1 = np.zeros((0, 2), dtype=np.float32)
                        points2 = np.zeros((0, 2), dtype=np.float32)
                    else:
                        # Move to CPU first if on CUDA
                        if torch.is_tensor(points1):
                            points1 = points1.cpu().numpy()
                        if torch.is_tensor(points2):
                            points2 = points2.cpu().numpy()
                        
                        # Convert to float for scaling
                        points1 = points1.astype(np.float32)
                        points2 = points2.astype(np.float32)
                        
                        # Scale points to full resolution
                        # Points are in VIT space (e.g., 518x518), need to scale to full image (e.g., 1920x1080)
                        scale_x = current_image.size[0] / vit_input_size  # width scaling
                        scale_y = current_image.size[1] / vit_input_size  # height scaling
                        
                        pass  # Scale factors calculated
                        
                        # Debug: Show min/max coordinates before scaling
                        if len(points1) > 0:
                            logger.debug(f"[Rotation] Points before scaling - min: ({points1.min():.1f}), max: ({points1.max():.1f})")
                            
                            # Check if these are patch indices that need converting to pixel coordinates
                            max_coord = max(points1[:, 0].max(), points1[:, 1].max(), points2[:, 0].max(), points2[:, 1].max())
                            
                            # Calculate expected grid size from vit_input_size and patch_size
                            patch_size = self.feature_extractor.get_patch_size()
                            num_patches = vit_input_size // patch_size
                            
                            logger.debug(f"[Rotation] max_coord={max_coord:.1f}, num_patches={num_patches}, patch_size={patch_size}, vit_input_size={vit_input_size}")
                            
                            if max_coord < num_patches:  # These are patch indices
                                logger.debug(f"[Rotation] Converting patch indices to pixels using scale_points_from_patch")
                                # Use the same scaling function as in matcher.py
                                from features.matcher import scale_points_from_patch
                                points1 = scale_points_from_patch(points1, vit_input_size, num_patches)
                                points2 = scale_points_from_patch(points2, vit_input_size, num_patches)
                                logger.debug(f"[Rotation] After patch->pixel conversion in VIT space - max: ({points1.max():.1f})")
                        
                        # Scale x,y coordinates separately
                        points1[:, 0] *= scale_x  # x coordinates
                        points1[:, 1] *= scale_y  # y coordinates
                        points2[:, 0] *= scale_x  # x coordinates
                        points2[:, 1] *= scale_y  # y coordinates
                        
                        # Debug: Show min/max coordinates after scaling
                        if len(points1) > 0:
                            logger.debug(f"[Rotation] Points after scaling to full res - x: ({points1[:, 0].min():.1f}, {points1[:, 0].max():.1f}), y: ({points1[:, 1].min():.1f}, {points1[:, 1].max():.1f})")
                    
                    # Convert points for visualization (flip x,y to y,x)
                    try:
                        if len(points1) > 0:
                            points1_viz = np.flip(points1, axis=1)
                            points2_viz = np.flip(points2, axis=1)
                        else:
                            points1_viz = points1
                            points2_viz = points2
                    except Exception as flip_error:
                        logger.debug(f"Could not flip points: {flip_error}. Using original arrays.")
                        points1_viz = points1
                        points2_viz = points2
            else:
                points1_viz = points1
                points2_viz = points2
            
            # Create visualization matching the feature matching style (visualization.py)
            import cv2
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import ConnectionPatch

            # Fixed output dimensions for consistent video recording (same as visualize_correspondences_ros)
            FIXED_WIDTH = 1200
            FIXED_HEIGHT = 600
            FIXED_DPI = 100

            # Create figure with fixed size and DPI
            fig = plt.figure(figsize=(FIXED_WIDTH / FIXED_DPI, FIXED_HEIGHT / FIXED_DPI), dpi=FIXED_DPI)

            # Use fixed subplot positions instead of tight_layout (prevents jiggle)
            ax1 = fig.add_axes([0.02, 0.02, 0.46, 0.96])  # [left, bottom, width, height]
            ax2 = fig.add_axes([0.52, 0.02, 0.46, 0.96])

            ax1.imshow(goal_image)
            ax2.imshow(rotated_image)

            # Plot correspondences if we have any
            if len(points1_viz) > 0:
                try:
                    colors = plt.cm.plasma(np.linspace(0.05, 0.95, max(1, len(points1_viz))))

                    for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1_viz, points2_viz, colors)):
                        # Plot points in goal image
                        ax1.plot(x1, y1, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

                        # Plot points in current (rotated) image
                        ax2.plot(x2, y2, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

                        # Draw correspondence lines
                        con = ConnectionPatch(
                            xyA=(x1, y1), xyB=(x2, y2),
                            coordsA="data", coordsB="data",
                            axesA=ax1, axesB=ax2, color=color, alpha=0.35
                        )
                        fig.add_artist(con)
                except Exception as e:
                    logger.warning(f"Error plotting correspondences: {e}")

            # Add angle text (no similarity metric)
            ax2.text(0.95, 0.05, f"{angle:.0f}°",
                    transform=ax2.transAxes,
                    fontsize=14,
                    color='white',
                    ha='right',
                    va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

            # Set axis limits to match image dimensions
            ax1.set_xlim(0, goal_image.size[0])
            ax1.set_ylim(goal_image.size[1], 0)  # Invert y-axis for image coordinates
            ax2.set_xlim(0, rotated_image.size[0])
            ax2.set_ylim(rotated_image.size[1], 0)  # Invert y-axis for image coordinates

            ax1.axis('off')
            ax2.axis('off')

            # NO tight_layout() - we use fixed axes positions for consistent output size

            # Convert figure to image with fixed dimensions
            fig.canvas.draw()
            img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_data = img_data.reshape((FIXED_HEIGHT, FIXED_WIDTH, 4))[:, :, :3]  # Fixed dimensions, drop alpha

            # Convert RGB to BGR for ROS (cv_bridge expects BGR)
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            ros_image = self.ros_controller.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")

            # Publish
            self.ros_controller.correspondence_pub.publish(ros_image)
            
            plt.close(fig)
            
        except Exception as e:
            import traceback
            logger.warning(f"Failed visualization with correspondences: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    def visualize_rotation_correspondences(self, goal_image: Image.Image, current_image: Image.Image, 
                                         angle: float, similarity: float):
        """
        Visualize correspondences for a specific rotation angle.
        Only runs if enable_visualization is True and ros_controller is available.
        """
        if not getattr(self.config, 'enable_visualization', True):
            return
            
        if self.ros_controller is None:
            return
            
        try:
            from vs_utils.visualization import visualize_correspondences_ros

            pass  # Starting visualization
            
            # Rotate current image
            rotated_image = self.rotate_image(current_image, angle)
            pass  # Image rotated
            
            # Resize images for feature extraction
            vit_input_size = self.config.vit_input_size
            goal_resized = goal_image.resize((vit_input_size, vit_input_size))
            rotated_resized = rotated_image.resize((vit_input_size, vit_input_size))
            
            with torch.no_grad():
                # Preprocess images
                goal_tensor = self.feature_extractor.preprocess_pil(goal_resized).to(self.device)
                rotated_tensor = self.feature_extractor.preprocess_pil(rotated_resized).to(self.device)
                
                # Extract features
                if hasattr(self.feature_extractor, 'extract_descriptors'):
                    desc1 = self.feature_extractor.extract_descriptors(
                        goal_tensor, layer=11, facet='key', 
                        bin=self.config.use_feature_binning
                    )
                    desc2 = self.feature_extractor.extract_descriptors(
                        rotated_tensor, layer=11, facet='key',
                        bin=self.config.use_feature_binning
                    )
                else:
                    desc1 = self.feature_extractor(goal_resized)
                    desc2 = self.feature_extractor(rotated_resized)
                
                # Use goal mask if available
                goal_mask_path = getattr(self.config, 'mask_path', None)
                if goal_mask_path and os.path.exists(goal_mask_path):
                    pass  # mask_path will be used by find_correspondences_batch
                else:
                    goal_mask_path = None
                    
                # Find correspondences for visualization
                pass  # Finding correspondences
                points1, points2, _ = find_correspondences_batch(
                    desc1, desc2,
                    num_pairs=min(self.config.num_pairs, 16),  # Limit for cleaner visualization
                    mask_path=goal_mask_path,
                    vit_input_size=self.config.vit_input_size,
                    patch_size=self.feature_extractor.get_patch_size()
                )
                
                pass  # Matches found
                
                # Check if we have any matches
                if points1 is None or points2 is None:
                    logger.warning(f"No valid patches found for angle {angle:.1f}° (find_correspondences returned None)")
                    # Create properly shaped empty arrays
                    empty_points = np.zeros((0, 2), dtype=np.float32)
                    # Still visualize to show the rotated image
                    visualize_correspondences_ros(
                        goal_image, rotated_image,
                        empty_points, empty_points,
                        None, self.ros_controller.bridge, self.ros_controller.correspondence_pub,
                        self.ros_controller.goal_image_pub, self.ros_controller.current_image_pub,
                        False, None
                    )
                    return
                    
                if len(points1) == 0:
                    logger.warning(f"No matches found for angle {angle:.1f}° - visualizing empty")
                    # Create properly shaped empty arrays
                    empty_points = np.zeros((0, 2), dtype=np.float32)
                    # Still visualize to show the images
                    visualize_correspondences_ros(
                        goal_image, rotated_image,
                        empty_points, empty_points,
                        None, self.ros_controller.bridge, self.ros_controller.correspondence_pub,
                        self.ros_controller.goal_image_pub, self.ros_controller.current_image_pub,
                        False, None
                    )
                    return
                
                # Scale points to full resolution for visualization
                scale_factor = current_image.size[0] / vit_input_size
                points1_scaled = points1 * scale_factor
                points2_scaled = points2 * scale_factor
                
                pass  # Points scaled
                
                # Convert points for visualization (flip x,y to y,x)
                points1_viz = np.flip(points1_scaled, axis=1)
                points2_viz = np.flip(points2_scaled, axis=1)
                
                pass  # Points prepared for visualization
                
                # Visualize using existing function (title will be added inside if needed)
                pass  # Calling visualization function
                try:
                    # Double-check all parameters before calling
                    pass  # Parameters checked
                    
                    visualize_correspondences_ros(
                        goal_image, rotated_image,
                        points1_viz, points2_viz,
                        None, self.ros_controller.bridge, self.ros_controller.correspondence_pub,
                        self.ros_controller.goal_image_pub, self.ros_controller.current_image_pub,
                        False, None
                    )
                    logger.info(f"Published rotation visualization for angle {angle:.1f}°")
                except Exception as viz_error:
                    logger.error(f"Error in visualize_correspondences_ros: {viz_error}")
                    raise  # Re-raise to see full traceback
                
        except Exception as e:
            import traceback
            logger.warning(f"Failed to visualize rotation correspondences: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def find_best_rotation_discrete(self, goal_image: Image.Image, current_image: Image.Image) -> Dict:
        """
        Find best rotation using discrete angles [0, 90, 180, 270].
        Mimics the original best_pose_finder behavior but with virtual rotation.
        """
        logger.info("Starting discrete rotation search [0°, 90°, 180°, 270°]...")
        
        start_time = time.time()
        best_angle = 0
        best_similarity = float('-inf')
        all_similarities = {}
        
        # Test discrete angles
        for angle in [0, 90, 180, 270]:
            # Rotate current image
            rotated_image = self.rotate_image(current_image, angle)
            
            # Compute similarity with goal using VS method
            similarity = self.compute_similarity_like_vs(goal_image, rotated_image)
            all_similarities[angle] = similarity
            
            logger.info(f"  Angle {angle}°: similarity = {similarity:.4f}")
            
            # Visualize this rotation if enabled (use simple version to avoid issues)
            self.visualize_rotation_correspondences_simple(goal_image, current_image, angle, similarity)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_angle = angle
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Best discrete angle: {best_angle}° (similarity: {best_similarity:.4f})")
        
        return {
            'best_angle': float(best_angle),
            'best_similarity': best_similarity,
            'all_similarities': all_similarities,
            'method': 'discrete',
            'time_seconds': elapsed_time
        }
    
    def find_best_rotation_continuous(self, goal_image: Image.Image, current_image: Image.Image) -> Dict:
        """
        Find best rotation using continuous global search.
        Based on rotation_alignment_global.py implementation.
        """
        logger.info("Starting continuous rotation search...")
        start_time = time.time()
        
        # Stage 1: Coarse global search
        logger.info(f"Stage 1: Coarse search (0-360° with {self.coarse_step}° steps)")
        total_coarse = len(np.arange(0, 360, self.coarse_step))
        logger.info(f"  Testing {total_coarse} angles...")
        
        coarse_angles = np.arange(0, 360, self.coarse_step)
        coarse_similarities = []
        
        for i, angle in enumerate(coarse_angles):
            if i % 5 == 0:  # Progress every 5 angles
                logger.info(f"  Progress: {i+1}/{total_coarse} angles tested...")
            rotated_image = self.rotate_image(current_image, angle)
            sim = self.compute_similarity_like_vs(goal_image, rotated_image)
            coarse_similarities.append(sim)
            
            # Visualize coarse search if enabled (only every 4th angle to reduce spam)
            if getattr(self.config, 'enable_visualization', True) and int(angle) % 20 == 0:
                self.visualize_rotation_correspondences_simple(goal_image, current_image, angle, sim)
        
        coarse_similarities = np.array(coarse_similarities)
        
        # Find top candidates
        n_candidates = 3
        top_indices = np.argsort(coarse_similarities)[-n_candidates:][::-1]
        
        logger.info(f"  Top {n_candidates} coarse candidates:")
        for idx in top_indices:
            logger.info(f"    {coarse_angles[idx]:.0f}°: {coarse_similarities[idx]:.4f}")
        
        # Stage 2: Fine search around candidates
        logger.info(f"Stage 2: Fine search (±{self.fine_range}° with {self.fine_step}° steps)")
        total_fine_per_candidate = int(2 * self.fine_range / self.fine_step) + 1
        logger.info(f"  Testing ~{total_fine_per_candidate} angles per candidate...")
        
        best_angle = None
        best_similarity = -1.0
        
        for idx, candidate_idx in enumerate(top_indices):
            candidate_angle = coarse_angles[candidate_idx]
            logger.info(f"  Fine search {idx+1}/{len(top_indices)} around {candidate_angle:.0f}°...")
            
            # Define fine search range
            fine_start = candidate_angle - self.fine_range
            fine_end = candidate_angle + self.fine_range
            fine_angles = np.arange(fine_start, fine_end + self.fine_step, self.fine_step)
            
            # Fine search
            for angle in fine_angles:
                # Normalize angle to 0-360 range
                angle_norm = angle % 360
                
                rotated_image = self.rotate_image(current_image, angle_norm)
                sim = self.compute_similarity_like_vs(goal_image, rotated_image)
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_angle = angle_norm
                    
                    # Visualize when we find a new best during fine search
                    if getattr(self.config, 'enable_visualization', True):
                        self.visualize_rotation_correspondences_simple(goal_image, current_image, angle_norm, sim)
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Best continuous angle: {best_angle:.1f}° (similarity: {best_similarity:.4f})")
        
        return {
            'best_angle': best_angle,
            'best_similarity': best_similarity,
            'method': 'continuous',
            'time_seconds': elapsed_time,
            'coarse_angles': coarse_angles.tolist(),
            'coarse_similarities': coarse_similarities.tolist()
        }
    
    def find_best_rotation(self, goal_image: Image.Image, current_image: Image.Image) -> Dict:
        """
        Main entry point: find best rotation using configured method.
        
        Args:
            goal_image: Reference/goal image
            current_image: Current camera image to align
            
        Returns:
            Dictionary with best_angle, best_similarity, and metadata
        """
        if self.rotation_mode == 'continuous':
            return self.find_best_rotation_continuous(goal_image, current_image)
        else:
            return self.find_best_rotation_discrete(goal_image, current_image)