#!/usr/bin/env python3
"""
LangSAM + LightTrack detector for ROI-based visual servoing.
Replaces YOLOWorld with a two-phase approach:
1. Initial detection: LangSAM with text prompt
2. Tracking: LightTrack for subsequent frames

This provides more robust detection in cluttered scenes by:
- Using semantic text-based detection (better for specific objects)
- Fast tracking between detections (~30fps vs ~1-2fps for detection)
"""

import numpy as np
from PIL import Image
import torch
import cv2
import sys
import os
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (e.g., library print statements)."""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

import logging

logger = logging.getLogger(__name__)

# LangSAM import
try:
    from lang_sam import LangSAM
    HAS_LANGSAM = True
except ImportError:
    HAS_LANGSAM = False
    LangSAM = None

# LightTrack import - check multiple paths
HAS_LIGHTTRACK = False
LightTrack = None

try:
    # Try system-installed LightTrack first
    from tracking.lighttrack import LightTrack
    HAS_LIGHTTRACK = True
except ImportError:
    try:
        # Try adding common LightTrack paths
        lighttrack_paths = [
            os.path.expanduser('~/LightTrack'),
            '/root/LightTrack',
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'LightTrack')
        ]
        for path in lighttrack_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        from tracking.lighttrack import LightTrack
        HAS_LIGHTTRACK = True
    except ImportError as e:
        print(f"[WARN] LightTrack not available: {e}")


def _log_info(msg):
    """Log info message."""
    logger.info(msg)


def _log_warn(msg):
    """Log warning message."""
    logger.warning(msg)


def _log_debug(msg):
    """Log debug message."""
    logger.debug(msg)


class LangSAMLightTrackDetector:
    """
    Combined detector using LangSAM for initial detection and LightTrack for tracking.
    Replaces YOLOWorld for more robust ROI detection in cluttered scenes.

    State Machine:
    - DETECT: Using LangSAM (initial + rotation phase)
    - TRACK: Using LightTrack (after rotation alignment)
    - CACHE: Using cached bbox (when tracking fails)

    Interface matches YOLOWorldDetector for seamless integration.
    """

    def __init__(self, lighttrack_weights='/root/LightTrack/snapshot/LightTrackM/LightTrackM.pth',
                 lighttrack_arch='LightTrackM_Subnet',
                 confidence_threshold=0.3,
                 cache_timeout=10,
                 device='cuda'):
        """
        Initialize LangSAM + LightTrack detector.

        Args:
            lighttrack_weights: Path to LightTrack model weights
            lighttrack_arch: LightTrack architecture (e.g., 'LightTrackM_Subnet')
            confidence_threshold: Minimum confidence for LangSAM detections
            cache_timeout: Number of iterations to use cached bbox before re-detection
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.lighttrack_weights = lighttrack_weights
        self.lighttrack_arch = lighttrack_arch
        self.cache_timeout = cache_timeout

        # Validate dependencies
        if not HAS_LANGSAM:
            raise ImportError(
                "LangSAM is not available. Please install it with: pip install lang-sam"
            )

        if not HAS_LIGHTTRACK:
            _log_warn("LightTrack not available. Tracking will be disabled (detection-only mode).")

        # Initialize LangSAM with explicit local cache paths to avoid network access.
        # Without these, SAM2 uses torch.hub.load_state_dict_from_url() and
        # GroundingDINO uses HuggingFace from_pretrained() — both may hang
        # waiting for network when a background ROS2 executor is running.
        import time as _time
        print("  [LangSAM] Resolving local cache paths...")
        sam_ckpt = os.path.expanduser(
            '~/.cache/torch/hub/checkpoints/sam2.1_hiera_small.pt')
        gdino_snapshot = os.path.expanduser(
            '~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/'
            'snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb')

        langsam_kwargs = {}
        if os.path.isfile(sam_ckpt):
            langsam_kwargs['sam_ckpt_path'] = sam_ckpt
            print(f"  [LangSAM] SAM2 checkpoint: {sam_ckpt} (LOCAL)")
        else:
            print(f"  [LangSAM] SAM2 checkpoint NOT cached — will download")
        if os.path.isdir(gdino_snapshot):
            langsam_kwargs['gdino_model_ckpt_path'] = gdino_snapshot
            langsam_kwargs['gdino_processor_ckpt_path'] = gdino_snapshot
            print(f"  [LangSAM] GroundingDINO: local path (LOCAL)")
        else:
            print(f"  [LangSAM] GroundingDINO NOT cached — will download")

        _t0 = _time.time()
        print(f"  [LangSAM] Calling LangSAM(**kwargs) with keys: {list(langsam_kwargs.keys())}...")
        self.langsam = LangSAM(**langsam_kwargs)
        print(f"  [LangSAM] Done in {_time.time()-_t0:.2f}s, device={self.device}")

        # LightTrack tracker (lazy initialization when start_tracking called)
        self.tracker = None

        # State management
        self.state = 'DETECT'  # DETECT, TRACK, CACHE
        self.is_tracking = False
        self.cached_bbox = None
        self.cache_counter = 0

        # Current text prompt
        self.text_prompt = None

        # Cache for goal image detection
        self.cached_goal_roi = None
        self.cached_goal_keyword = None

        # Logging state (to avoid spam)
        self._logged_detection = False
        self._logged_tracking = False

    def set_classes(self, classes):
        """
        Set the text prompt for detection.
        Interface compatibility with YOLOWorldDetector.

        Args:
            classes: List of class names (uses first one as text prompt)
        """
        if classes and len(classes) > 0:
            self.text_prompt = classes[0]
            _log_debug(f"Set LangSAM text prompt: '{self.text_prompt}'")

    def detect(self, image, keyword=None, padding_ratio=0.1, spatial_prior=None):
        """
        Detect or track object in image.
        Interface matches YOLOWorldDetector.

        Behavior depends on current state:
        - DETECT: Run LangSAM detection
        - TRACK: Run LightTrack update
        - CACHE: Return cached bbox, increment counter

        Args:
            image: PIL Image or numpy array
            keyword: Text prompt for detection (overrides set_classes)
            padding_ratio: Padding to add around detected box (0.1 = 10%)
            spatial_prior: Optional (cx, cy) tuple. When provided and multiple
                          detections exist, selects the detection closest to this
                          position instead of highest confidence. Used for
                          multi-instance consistency after rotation compensation.

        Returns:
            dict with:
                - 'bbox': [x1, y1, x2, y2] or None if no detection
                - 'confidence': Detection/tracking confidence
                - 'class_name': Text prompt used
                - 'padded_bbox': Bbox with padding applied
        """
        # Update text prompt if keyword provided
        if keyword:
            self.text_prompt = keyword

        # State machine dispatch
        if self.state == 'TRACK' and self.is_tracking and self.tracker is not None:
            return self._track_frame(image, padding_ratio)
        elif self.state == 'CACHE':
            return self._use_cached_bbox(image, padding_ratio)
        else:
            return self._detect_langsam(image, padding_ratio, spatial_prior=spatial_prior)

    def start_tracking(self, frame, bbox):
        """
        Initialize LightTrack tracker with detected bbox.
        Called by controller when rotation_found=True.

        Args:
            frame: numpy array (BGR or RGB image) or PIL Image
            bbox: [x1, y1, x2, y2] bounding box
        """
        if not HAS_LIGHTTRACK:
            _log_warn("LightTrack not available, staying in detection mode")
            return

        # IMPORTANT: LightTrack wrapper expects BGR input (OpenCV convention)
        # It internally converts BGR -> RGB for the neural network
        # PIL Images are RGB, so we must convert to BGR first
        if isinstance(frame, Image.Image):
            # PIL is RGB, convert to BGR for LightTrack wrapper
            frame_rgb = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _log_info("[LightTrack] Converted PIL RGB -> BGR for wrapper")
        elif isinstance(frame, np.ndarray) and len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume RGB numpy array (from PIL), convert to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _log_info("[LightTrack] Converted numpy RGB -> BGR for wrapper")
        else:
            frame_bgr = frame
            _log_warn("[LightTrack] Using frame as-is (unknown format)")

        # Convert bbox from xyxy to xywh format
        bbox_xywh = self._xyxy_to_xywh(bbox)

        # Initialize tracker
        try:
            _log_info("=" * 60)
            _log_info("[LightTrack INIT] Starting initialization...")
            _log_info(f"[LightTrack INIT] Weights path: {self.lighttrack_weights}")
            _log_info(f"[LightTrack INIT] Architecture: {self.lighttrack_arch}")
            _log_info(f"[LightTrack INIT] Frame shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
            _log_info(f"[LightTrack INIT] Frame BGR channels - B:{frame_bgr[:,:,0].mean():.1f}, G:{frame_bgr[:,:,1].mean():.1f}, R:{frame_bgr[:,:,2].mean():.1f}")
            _log_info(f"[LightTrack INIT] Input bbox xyxy: {bbox}")
            _log_info(f"[LightTrack INIT] Converted to xywh: {bbox_xywh}")

            # Create tracker - this should print weight loading status
            self.tracker = LightTrack(weights=self.lighttrack_weights, arch=self.lighttrack_arch)

            # Verify tracker was created
            if self.tracker is None:
                raise RuntimeError("LightTrack tracker creation returned None")

            # Initialize with frame and bbox
            self.tracker.init(frame_bgr, bbox_xywh)

            self.is_tracking = True
            self.state = 'TRACK'
            self.cached_bbox = list(bbox)  # Cache the initial bbox

            _log_info(f"[LightTrack INIT] SUCCESS - Tracker initialized!")
            _log_info("=" * 60)
        except Exception as e:
            _log_warn(f"[LightTrack INIT] FAILED: {e}")
            import traceback
            _log_warn(traceback.format_exc())
            self.state = 'DETECT'

    def detect_and_cache_goal(self, goal_image, keyword, padding_ratio=0.1):
        """
        Detect object in goal image using LangSAM and cache the result.
        Goal image is always detected (not tracked).

        Args:
            goal_image: PIL Image of the goal
            keyword: Text prompt for detection
            padding_ratio: Padding ratio

        Returns:
            Detection result dict
        """
        self.text_prompt = keyword
        result = self._detect_langsam(goal_image, padding_ratio)

        if result['bbox'] is not None:
            self.cached_goal_roi = result['padded_bbox']
            self.cached_goal_keyword = keyword
            _log_info(f"Cached goal ROI for '{keyword}': {result['padded_bbox']}")
        else:
            _log_warn(f"No detection in goal image for keyword '{keyword}'")

        return result

    def get_cached_goal_roi(self):
        """Get cached goal image ROI."""
        return self.cached_goal_roi

    def reset_tracking(self):
        """
        Reset tracker state for new servoing run.
        Called between samples in experiment loop.
        """
        self.state = 'DETECT'
        self.is_tracking = False
        self.tracker = None
        self.cached_bbox = None
        self.cache_counter = 0
        self._logged_detection = False
        self._logged_tracking = False
        _log_debug("LangSAM+LightTrack detector reset")

    def _detect_langsam(self, image, padding_ratio, spatial_prior=None):
        """
        Internal: Run LangSAM detection on image.

        Args:
            image: PIL Image or numpy array
            padding_ratio: Padding to add around bbox
            spatial_prior: Optional (cx, cy) tuple for multi-instance selection.
                          When provided and multiple detections exist, selects
                          the detection closest to this position instead of
                          highest confidence.

        Returns:
            Detection result dict
        """
        if self.text_prompt is None:
            _log_warn("No text prompt set for LangSAM detection")
            return self._empty_result()

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Assume BGR, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            _log_warn(f"Unsupported image type: {type(image)}")
            return self._empty_result()

        # Get numpy array for padding calculation
        image_np = np.array(image_pil)

        try:
            # Run LangSAM prediction (suppress library prints)
            if not self._logged_detection:
                _log_info(f"Running LangSAM detection with prompt: '{self.text_prompt}'")
                self._logged_detection = True

            with suppress_stdout():
                results = self.langsam.predict([image_pil], [self.text_prompt])

            if results and len(results) > 0:
                result_dict = results[0]

                # Check for masks
                if 'masks' in result_dict and result_dict['masks'] is not None:
                    masks = result_dict['masks']
                    boxes = result_dict.get('boxes', None)
                    scores = result_dict.get('scores', None)

                    # Handle empty results
                    if isinstance(masks, np.ndarray) and masks.size > 0:
                        if boxes is not None and len(boxes) > 0:
                            # Select best detection
                            if spatial_prior is not None and len(boxes) > 1:
                                # Multi-instance: select closest to predicted position
                                best_idx = self._select_by_spatial_prior(boxes, scores, spatial_prior)
                            elif scores is not None and len(scores) > 0:
                                best_idx = np.argmax(scores)
                            else:
                                best_idx = 0

                            bbox = boxes[best_idx]
                            confidence = float(scores[best_idx]) if scores is not None and len(scores) > best_idx else 1.0

                            # Convert to list
                            bbox = [float(x) for x in bbox]

                            # Apply padding
                            padded_bbox = self._apply_padding(bbox, image_np.shape, padding_ratio)

                            # Cache for tracking initialization
                            self.cached_bbox = bbox

                            return {
                                'bbox': bbox,
                                'confidence': float(confidence),
                                'class_name': self.text_prompt,
                                'padded_bbox': padded_bbox
                            }

            _log_debug(f"No detection for prompt '{self.text_prompt}'")
            return self._empty_result()

        except Exception as e:
            _log_warn(f"LangSAM detection failed: {e}")
            return self._empty_result()

    def _select_by_spatial_prior(self, boxes, scores, spatial_prior, confidence_weight=0.3):
        """
        Select detection closest to predicted position, with confidence tie-breaking.

        Uses a combined score: distance_score * (1 - w) + confidence_score * w
        This ensures the closest detection wins unless it has very low confidence.

        Args:
            boxes: Array of [x1, y1, x2, y2] bounding boxes
            scores: Array of confidence scores
            spatial_prior: (cx, cy) predicted center position
            confidence_weight: Weight for confidence in combined score (0-1)

        Returns:
            Index of best detection
        """
        prior_cx, prior_cy = spatial_prior

        # Calculate center of each detection
        centers = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes])

        # Euclidean distance from prior to each center
        distances = np.sqrt((centers[:, 0] - prior_cx)**2 + (centers[:, 1] - prior_cy)**2)

        # Normalize distance to [0, 1] where 0=farthest, 1=closest
        max_dist = distances.max() if distances.max() > 0 else 1.0
        dist_scores = 1.0 - (distances / max_dist)

        # Normalize confidence scores to [0, 1]
        if scores is not None and len(scores) > 0:
            conf_scores = np.array([float(s) for s in scores])
            max_conf = conf_scores.max() if conf_scores.max() > 0 else 1.0
            conf_scores = conf_scores / max_conf
        else:
            conf_scores = np.ones(len(boxes))

        # Combined score: proximity dominates (70%) with confidence tie-break (30%)
        combined = dist_scores * (1 - confidence_weight) + conf_scores * confidence_weight
        best_idx = int(np.argmax(combined))

        _log_info(f"[SPATIAL PRIOR] {len(boxes)} detections, selected idx={best_idx} "
                  f"(dist={distances[best_idx]:.1f}px, "
                  f"conf={float(scores[best_idx]) if scores is not None and len(scores) > best_idx else 'N/A'})")

        return best_idx

    def _track_frame(self, image, padding_ratio):
        """
        Internal: Run LightTrack on current frame.

        Args:
            image: PIL Image or numpy array
            padding_ratio: Padding to add around bbox

        Returns:
            Tracking result dict
        """
        if image is None:
            _log_warn("[LightTrack] Received None image, falling back to cache")
            self.state = 'CACHE'
            self.cache_counter = 0
            return self._use_cached_bbox(image, padding_ratio)

        # IMPORTANT: LightTrack wrapper expects BGR input (OpenCV convention)
        # Convert to BGR numpy array for the wrapper
        if isinstance(image, Image.Image):
            # PIL is RGB, convert to BGR for LightTrack wrapper
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            # Assume RGB numpy array, convert to BGR
            image_np = image
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_np = image
            image_bgr = image

        if not self._logged_tracking:
            _log_info("[LightTrack] Tracking mode ACTIVE - using LightTrack for ROI updates")
            self._logged_tracking = True

        try:
            success, bbox_xywh = self.tracker.update(image_bgr)

            if success and bbox_xywh is not None:
                # Convert xywh to xyxy
                bbox_xyxy = self._xywh_to_xyxy(bbox_xywh)

                # DEBUG: Check for drift by comparing to cached bbox
                if self.cached_bbox is not None:
                    dx = bbox_xyxy[0] - self.cached_bbox[0]
                    dy = bbox_xyxy[1] - self.cached_bbox[1]
                    # Only log if significant movement (>20px) - might indicate drift
                    if abs(dx) > 20 or abs(dy) > 20:
                        _log_debug(f"[LightTrack] Large movement: dx={dx:.0f}, dy={dy:.0f}")

                self.cached_bbox = list(bbox_xyxy)

                # Apply padding
                padded_bbox = self._apply_padding(bbox_xyxy, image_np.shape, padding_ratio)

                return {
                    'bbox': bbox_xyxy,
                    'confidence': 0.9,  # LightTrack doesn't return confidence
                    'class_name': self.text_prompt,
                    'padded_bbox': padded_bbox
                }
            else:
                # Tracking lost - immediately re-detect with LangSAM
                _log_warn("[LightTrack] Lost target (success=False), re-detecting with LangSAM...")
                result = self._detect_langsam(image, padding_ratio)
                if result['bbox'] is not None:
                    # Re-initialize tracker with fresh detection
                    if isinstance(image, Image.Image):
                        frame = np.array(image)
                    else:
                        frame = image
                    self.start_tracking(frame, result['bbox'])
                    _log_info("[LightTrack] Re-initialized tracker after re-detection")
                else:
                    # LangSAM also failed - fall back to cache
                    _log_warn("[LightTrack] LangSAM re-detection also failed, using cached bbox")
                    self.state = 'CACHE'
                    self.cache_counter = 0
                    return self._use_cached_bbox(image, padding_ratio)
                return result

        except Exception as e:
            _log_warn(f"[LightTrack] Update exception: {e}, re-detecting with LangSAM...")
            result = self._detect_langsam(image, padding_ratio)
            if result['bbox'] is not None:
                if isinstance(image, Image.Image):
                    frame = np.array(image)
                else:
                    frame = image
                self.start_tracking(frame, result['bbox'])
                _log_info("[LightTrack] Re-initialized tracker after exception recovery")
            else:
                self.state = 'CACHE'
                self.cache_counter = 0
                return self._use_cached_bbox(image, padding_ratio)
            return result

    def _use_cached_bbox(self, image, padding_ratio):
        """
        Internal: Return cached bbox, trigger re-detection if timeout.

        Args:
            image: PIL Image or numpy array
            padding_ratio: Padding to add around bbox

        Returns:
            Cached or re-detected result dict
        """
        self.cache_counter += 1

        if self.cache_counter >= self.cache_timeout:
            # Re-detect with LangSAM
            _log_info(f"Cache timeout ({self.cache_timeout} iterations), re-detecting with LangSAM")
            result = self._detect_langsam(image, padding_ratio)

            if result['bbox'] is not None:
                # Reinitialize tracking if available
                if HAS_LIGHTTRACK:
                    if isinstance(image, Image.Image):
                        frame = np.array(image)
                    else:
                        frame = image
                    self.start_tracking(frame, result['bbox'])
                else:
                    self.state = 'DETECT'

            return result

        # Return cached bbox
        if self.cached_bbox is not None:
            if image is None:
                _log_warn("[CACHE] Image is None, returning cached bbox without padding")
                return {
                    'bbox': self.cached_bbox,
                    'confidence': 0.5,
                    'class_name': self.text_prompt,
                    'padded_bbox': self.cached_bbox  # Use bbox as padded_bbox
                }
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            padded_bbox = self._apply_padding(self.cached_bbox, image_np.shape, padding_ratio)

            return {
                'bbox': self.cached_bbox,
                'confidence': 0.5,  # Lower confidence for cached bbox
                'class_name': self.text_prompt,
                'padded_bbox': padded_bbox
            }

        return self._empty_result()

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

        return [x1_padded, y1_padded, x2_padded, y2_padded]

    def crop_roi(self, image, bbox):
        """
        Crop region of interest from image.
        Interface matches YOLOWorldDetector.

        Args:
            image: PIL Image or numpy array
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Cropped PIL Image
        """
        if bbox is None:
            return image

        x1, y1, x2, y2 = [int(x) for x in bbox]

        if isinstance(image, Image.Image):
            return image.crop((x1, y1, x2, y2))
        else:
            # Convert numpy to PIL, crop, and return
            pil_image = Image.fromarray(image)
            return pil_image.crop((x1, y1, x2, y2))

    def _empty_result(self):
        """Return empty detection result."""
        return {
            'bbox': None,
            'confidence': 0.0,
            'class_name': None,
            'padded_bbox': None
        }

    @staticmethod
    def _xyxy_to_xywh(bbox):
        """Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]."""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        w = x2 - x1
        h = y2 - y1
        return (x1, y1, w, h)

    @staticmethod
    def _xywh_to_xyxy(bbox):
        """Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]."""
        x, y, w, h = [int(x) for x in bbox]
        return [x, y, x + w, y + h]


# Availability check for imports
LANGSAM_LIGHTTRACK_AVAILABLE = HAS_LANGSAM
