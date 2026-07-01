#!/usr/bin/env python3
"""
YOLOWorld detector for ROI-based visual servoing.
Detects objects in images and returns bounding boxes for region of interest extraction.
"""

import numpy as np
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)
try:
    from ultralytics import YOLOWorld
    YOLOWORLD_AVAILABLE = True
except ImportError:
    # YOLOWorld not available in this version of ultralytics
    YOLOWORLD_AVAILABLE = False
    YOLOWorld = None
import supervision as sv


class YOLOWorldDetector:
    """
    Wrapper for YOLOWorld object detection model.
    Used to detect objects and extract regions of interest for visual servoing.
    """
    
    def __init__(self, model_size='s', confidence_threshold=0.3, device='cuda'):
        """
        Initialize YOLOWorld detector.
        
        Args:
            model_size: Model size ('s', 'm', 'l', 'x') - small, medium, large, extra-large
            confidence_threshold: Minimum confidence for detections
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_size = model_size
        
        # Check if YOLOWorld is available
        if not YOLOWORLD_AVAILABLE:
            raise ImportError(
                "YOLOWorld is not available in the current ultralytics version. "
                "Please run: python3 setup_yoloworld.py to install the correct version."
            )
            
        # Initialize YOLOWorld model
        try:
            # YOLOWorld v1 models: yolov8s-world, yolov8m-world, yolov8l-world, yolov8x-world
            model_name = f'yolov8{model_size}-world.pt'
            logger.info(f"Loading YOLOWorld model: {model_name}")
            
            self.model = YOLOWorld(model_name)
            self.model.to(self.device)
            
            logger.info(f"Successfully loaded YOLOWorld-{model_size.upper()} on {self.device}")
            logger.info(f"Model info: {model_size.upper()} = {'Small (fast)' if model_size == 's' else 'Medium (balanced)' if model_size == 'm' else 'Large (accurate)' if model_size == 'l' else 'Extra-large (most accurate)'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLOWorld: {e}")
            logger.error("The model will be downloaded on first use.")
            raise
            
        # Cache for goal image detection
        self.cached_goal_roi = None
        self.cached_goal_keyword = None
        
    def set_classes(self, classes):
        """
        Set the classes to detect.
        
        Args:
            classes: List of class names to detect (e.g., ['mug', 'laptop'])
        """
        try:
            self.model.set_classes(classes)
        except Exception as e:
            logger.error(f"Failed to set YOLOWorld classes: {e}")
            
    def detect(self, image, keyword=None, padding_ratio=0.1):
        """
        Detect objects in image and return bounding box.
        
        Args:
            image: PIL Image or numpy array
            keyword: Specific object keyword to detect (if None, uses pre-set classes)
            padding_ratio: Padding to add around detected box (0.1 = 10%)
            
        Returns:
            dict with:
                - 'bbox': [x1, y1, x2, y2] or None if no detection
                - 'confidence': Detection confidence
                - 'class_name': Detected class name
                - 'padded_bbox': Bbox with padding applied
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Set specific class if keyword provided
        if keyword:
            self.set_classes([keyword])
            
        try:
            # Run detection with verbose output suppressed
            results = self.model(image_np, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the most confident detection
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                
                # Extract bounding box
                bbox = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = confidences[best_idx]
                
                # Get class name
                class_id = int(boxes.cls[best_idx])
                class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)
                
                # Apply padding
                padded_bbox = self._apply_padding(bbox, image_np.shape, padding_ratio)
                
                return {
                    'bbox': bbox.tolist(),
                    'confidence': float(confidence),
                    'class_name': class_name,
                    'padded_bbox': padded_bbox
                }
            else:
                return {
                    'bbox': None,
                    'confidence': 0.0,
                    'class_name': None,
                    'padded_bbox': None
                }
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'bbox': None,
                'confidence': 0.0,
                'class_name': None,
                'padded_bbox': None
            }
            
    def detect_and_cache_goal(self, goal_image, keyword, padding_ratio=0.1):
        """
        Detect object in goal image and cache the result.
        
        Args:
            goal_image: PIL Image of the goal
            keyword: Object keyword to detect
            padding_ratio: Padding ratio
            
        Returns:
            Detection result dict
        """
        result = self.detect(goal_image, keyword, padding_ratio)
        
        if result['bbox'] is not None:
            self.cached_goal_roi = result['padded_bbox']
            self.cached_goal_keyword = keyword
        else:
            logger.warning(f"No detection in goal image for keyword '{keyword}'")
            
        return result
        
    def get_cached_goal_roi(self):
        """Get cached goal image ROI."""
        return self.cached_goal_roi
        
    def _apply_padding(self, bbox, image_shape, padding_ratio):
        """
        Apply padding to bounding box while respecting image boundaries.
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width, channels)
            padding_ratio: Padding as ratio of bbox size
            
        Returns:
            Padded bbox [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Calculate padding
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio
        
        # Apply padding with boundary checks
        img_height, img_width = image_shape[:2]

        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        # Debug: Log detection details once
        if not hasattr(self, '_logged_padding'):
            logger.info(f"[DEBUG YOLO] Image shape: {img_width}x{img_height}")
            logger.info(f"[DEBUG YOLO] Original bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            logger.info(f"[DEBUG YOLO] Padding: x={pad_x:.1f}, y={pad_y:.1f}")
            logger.info(f"[DEBUG YOLO] Padded bbox: [{x1_padded:.1f}, {y1_padded:.1f}, {x2_padded:.1f}, {y2_padded:.1f}]")
            self._logged_padding = True

        return [x1_padded, y1_padded, x2_padded, y2_padded]
        
    def crop_roi(self, image, bbox):
        """
        Crop region of interest from image.

        Args:
            image: PIL Image or numpy array
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Cropped PIL Image
        """
        if bbox is None:
            return image

        x1, y1, x2, y2 = [int(x) for x in bbox]

        # DEBUG: Log what we're cropping
        if isinstance(image, Image.Image):
            logger.debug(f"[CROP ROI] Image size: {image.size}, Crop bbox: [{x1}, {y1}, {x2}, {y2}]")
            logger.debug(f"[CROP ROI] ROI dimensions: {x2-x1}x{y2-y1}")

        if isinstance(image, Image.Image):
            return image.crop((x1, y1, x2, y2))
        else:
            # Convert numpy to PIL, crop, and return
            pil_image = Image.fromarray(image)
            return pil_image.crop((x1, y1, x2, y2))