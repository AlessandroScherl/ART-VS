#!/usr/bin/env python3
"""
Classical feature extractors (SIFT, ORB, AKAZE) for visual servoing.
Mimics the implementation from ibvs_standard.py - no tiling, no rotation compensation.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)
from typing import Tuple, Optional, Dict

from features.base_extractor import BaseFeatureExtractor


class ClassicalFeatureExtractor(BaseFeatureExtractor):
    """
    Wrapper for classical computer vision feature detectors.
    Supports SIFT, ORB, and AKAZE.
    """
    
    SUPPORTED_METHODS = {
        'sift': 'SIFT (Scale-Invariant Feature Transform)',
        'orb': 'ORB (Oriented FAST and Rotated BRIEF)',
        'akaze': 'AKAZE (Accelerated-KAZE)'
    }
    
    def __init__(self, method: str = 'sift', device: str = 'cuda'):
        """
        Initialize classical feature extractor.
        
        Args:
            method: One of 'sift', 'orb', 'akaze'
            device: Device to use (kept for compatibility, classical methods run on CPU)
        """
        super().__init__(device)
        self.method = method.lower()
        
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. Choose from {list(self.SUPPORTED_METHODS.keys())}")
        
        # Initialize the detector
        self._init_detector()
        
        # Store for matching
        self.goal_keypoints = None
        self.goal_descriptors = None
        self.current_keypoints = None
        self.current_descriptors = None
        
        # Cache FLANN matcher for SIFT (avoid recreating)
        self.flann_matcher = None
        if self.method == 'sift':
            self._init_flann_matcher()
        
    def _init_detector(self):
        """Initialize the feature detector based on method."""
        if self.method == 'sift':
            # Optimized SIFT parameters for speed
            self.detector = cv2.SIFT_create(
                nfeatures=300,  # Reduced from 500 for faster detection
                nOctaveLayers=3,  # Default is 3, could reduce to 2 for more speed
                contrastThreshold=0.08,  # Higher = fewer but stronger keypoints (default 0.04)
                edgeThreshold=10,  # Default is 10
                sigma=1.6  # Gaussian blur, default 1.6
            )
            self.norm_type = cv2.NORM_L2
        elif self.method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.norm_type = cv2.NORM_HAMMING
        elif self.method == 'akaze':
            self.detector = cv2.AKAZE_create()
            self.norm_type = cv2.NORM_HAMMING
        
        logger.info(f"Initialized {self.method.upper()} detector")
    
    def _init_flann_matcher(self):
        """Initialize and cache FLANN matcher for SIFT."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE, 
            trees=5  # 5 trees is good balance
        )
        search_params = dict(
            checks=32  # Reduced from 50 for faster search
        )
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        logger.info("Initialized FLANN matcher (cached for reuse)")
    
    def preprocess_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL image for classical feature detection.
        
        Args:
            pil_image: PIL Image
            
        Returns:
            Grayscale numpy array
        """
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        return gray
    
    def extract_keypoints_and_descriptors(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Extract keypoints and descriptors from image.
        
        Args:
            image: Grayscale image
            
        Returns:
            keypoints, descriptors
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray, num_pairs: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features between two sets of descriptors.
        Mimics the matching logic from ibvs_standard.py.
        
        Args:
            des1: Descriptors from goal image
            des2: Descriptors from current image
            num_pairs: Number of matches to use
            
        Returns:
            points1, points2, similarities (normalized distances)
        """
        if des1 is None or des2 is None:
            return None, None, None
        
        # Use FLANN matcher for SIFT (much faster than BruteForce)
        if self.method == 'sift':
            # Use cached FLANN matcher (avoid recreating)
            if self.flann_matcher is None:
                self._init_flann_matcher()
            
            # Find k=2 nearest neighbors for Lowe's ratio test
            try:
                matches_knn = self.flann_matcher.knnMatch(des1, des2, k=2)
            except cv2.error as e:
                logger.warning(f"FLANN matching failed: {e}")
                return None, None, None
            
            # Apply Lowe's ratio test (0.75 — slightly relaxed for real-world conditions)
            matches = []
            for match_pair in matches_knn:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        matches.append(m)
                elif len(match_pair) == 1:
                    # Only one match found, use it if distance is reasonable
                    if match_pair[0].distance < 100:  # Threshold for SIFT
                        matches.append(match_pair[0])
        else:
            # Use BruteForce for ORB/AKAZE (binary descriptors)
            bf = cv2.BFMatcher(self.norm_type, crossCheck=True)
            matches = bf.match(des1, des2)
        
        if len(matches) < 4:  # Minimum required for visual servoing
            logger.warning(f"Insufficient matches found: {len(matches)} < 4")
            return None, None, None
        
        # Get all distances
        distances = np.array([m.distance for m in matches])
        
        # Normalize and invert distances to [0,1] range where 1 is best match
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist > min_dist:
            normalized_distances = 1 - (distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.ones_like(distances)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Determine number of pairs to use
        available_matches = len(matches)
        num_pairs_to_use = min(num_pairs, available_matches)
        
        if num_pairs_to_use < 4:
            logger.warning("Not enough good matches for visual servoing")
            return None, None, None
        
        # Select top matches
        matches = matches[:num_pairs_to_use]
        
        # Extract matched points
        points1 = np.float32([self.goal_keypoints[m.queryIdx].pt for m in matches])
        points2 = np.float32([self.current_keypoints[m.trainIdx].pt for m in matches])
        
        # Get normalized similarities for selected matches
        similarities = np.array([normalized_distances[i] for i in range(num_pairs_to_use)])
        
        return points1, points2, similarities
    
    def detect_features_classical(self, goal_image: Image.Image, current_image: Image.Image, 
                                 num_pairs: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main feature detection method that mimics ibvs_standard.py detect_features().
        
        Args:
            goal_image: PIL Image of goal
            current_image: PIL Image of current view
            num_pairs: Number of feature pairs to extract
            
        Returns:
            goal_points, current_points, similarities
        """
        # Preprocess images
        goal_gray = self.preprocess_pil(goal_image)
        current_gray = self.preprocess_pil(current_image)
        
        # Extract keypoints and descriptors
        self.goal_keypoints, self.goal_descriptors = self.extract_keypoints_and_descriptors(goal_gray)
        self.current_keypoints, self.current_descriptors = self.extract_keypoints_and_descriptors(current_gray)
        
        # Log detection results
        # Commented out to reduce spam
        # logger.info(
        #     f"{self.method.upper()} found {len(self.goal_keypoints)} keypoints in goal image "
        #     f"and {len(self.current_keypoints)} keypoints in current image"
        # )
        
        # Match features
        points1, points2, similarities = self.match_features(
            self.goal_descriptors, 
            self.current_descriptors,
            num_pairs
        )
        
        # Commented out to reduce spam
        # if points1 is not None:
        #     logger.info(f"Using {len(points1)} matched features for visual servoing")
        
        return points1, points2, similarities
    
    # Methods for compatibility with BaseFeatureExtractor interface
    def extract_descriptors(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        This method is not used for classical features.
        Classical methods use detect_features_classical() instead.
        """
        raise NotImplementedError("Classical features use detect_features_classical() method")
    
    def get_descriptor_dim(self) -> int:
        """Get the dimension of the feature descriptors."""
        if self.method == 'sift':
            return 128
        elif self.method == 'orb':
            return 32
        elif self.method == 'akaze':
            return 61
        return 0
    
    def get_patch_size(self) -> int:
        """Classical methods don't have a patch size."""
        return 1  # Return 1 to avoid division by zero
    
    def get_supported_methods(self) -> Dict[str, str]:
        """Get dictionary of supported methods."""
        return self.SUPPORTED_METHODS
    
    def get_feature_info(self) -> Dict:
        """Get information about the feature extractor."""
        return {
            'method': self.method,
            'descriptor_dim': self.get_descriptor_dim(),
            'detector_type': 'classical',
            'norm_type': 'L2' if self.method == 'sift' else 'Hamming'
        }

    def detect_features_classical_roi(self, goal_roi: Image.Image, current_roi: Image.Image,
                                       goal_bbox: list, current_bbox: list,
                                       num_pairs: int = 24,
                                       normalize_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect and match features within ROI regions.

        CRITICAL: Both ROIs are resized to the same dimensions before feature
        detection to ensure proper matching. Without this, different ROI sizes
        cause scale mismatches that break feature matching.

        Features are detected in normalized-size ROI images, then coordinates
        are mapped back to full image space.

        Args:
            goal_roi: Cropped goal image (PIL)
            current_roi: Cropped current image (PIL)
            goal_bbox: [x1, y1, x2, y2] of goal ROI in full image
            current_bbox: [x1, y1, x2, y2] of current ROI in full image
            num_pairs: Number of feature pairs to return
            normalize_size: Size to normalize both ROIs to (default 512)

        Returns:
            goal_points: Points in FULL IMAGE coordinates
            current_points: Points in FULL IMAGE coordinates
            similarities: Match scores (normalized, 1=best)
        """
        # Store original ROI sizes for coordinate mapping
        goal_roi_original_size = goal_roi.size  # (width, height)
        current_roi_original_size = current_roi.size

        # 1. Resize both ROIs to the same size for consistent feature matching
        # This is critical - without it, different ROI sizes cause scale mismatches
        goal_roi_normalized = goal_roi.resize((normalize_size, normalize_size))
        current_roi_normalized = current_roi.resize((normalize_size, normalize_size))

        # 2. Run feature detection on normalized-size ROIs
        goal_points_norm, current_points_norm, similarities = self.detect_features_classical(
            goal_roi_normalized, current_roi_normalized, num_pairs
        )

        if goal_points_norm is None or current_points_norm is None:
            return None, None, None

        # 3. Map normalized coordinates back to original ROI coordinates
        # Points are in normalized space (0 to normalize_size)
        # Need to scale back to original ROI dimensions
        goal_scale_x = goal_roi_original_size[0] / normalize_size
        goal_scale_y = goal_roi_original_size[1] / normalize_size
        current_scale_x = current_roi_original_size[0] / normalize_size
        current_scale_y = current_roi_original_size[1] / normalize_size

        # Scale points back to original ROI coordinates
        goal_points_roi = goal_points_norm.copy()
        goal_points_roi[:, 0] *= goal_scale_x
        goal_points_roi[:, 1] *= goal_scale_y

        current_points_roi = current_points_norm.copy()
        current_points_roi[:, 0] *= current_scale_x
        current_points_roi[:, 1] *= current_scale_y

        # 4. Map ROI coordinates → Full image coordinates
        goal_offset = np.array([goal_bbox[0], goal_bbox[1]])
        current_offset = np.array([current_bbox[0], current_bbox[1]])

        goal_points_full = goal_points_roi + goal_offset
        current_points_full = current_points_roi + current_offset

        return goal_points_full, current_points_full, similarities