"""
LightTrack Wrapper - Provides simple interface for ObjectTracking4VS compatibility.

This wrapper provides a simple LightTrack class with:
- __init__(weights, arch) - Initialize with model weights and architecture
- init(frame, bbox) - Initialize tracker with first frame and bounding box (xywh)
- update(frame) - Track object in new frame, returns (success, bbox_xywh)

The official LightTrack repo has a more complex API in lib/tracker/lighttrack.py.
This wrapper simplifies it for visual servoing applications.
"""

import os
import sys
import cv2
import numpy as np
import torch

# Add lib to path for imports from the LightTrack repo
lighttrack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_path = os.path.join(lighttrack_root, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Import from LightTrack lib
from lib.models import models as model_zoo
from lib.tracker.lighttrack import Lighttrack as LighttrackCore
from lib.utils.utils import load_pretrain


class SiamInfo:
    """Configuration object for tracker initialization."""
    def __init__(self, stride=16):
        self.stride = stride
        self.dataset = 'VOT2019'  # Default dataset for config loading


# Default weights path - mounted volume in Docker container (override with LIGHTTRACK_WEIGHTS)
DEFAULT_WEIGHTS_PATH = os.environ.get('LIGHTTRACK_WEIGHTS', '/root/vision_ws/src/models/LightTrack/snapshot/LightTrackM/LightTrackM.pth')


class LightTrack:
    """
    Simple wrapper for LightTrack tracker.

    Provides OpenCV-like interface:
    - init(frame, bbox) where bbox is (x, y, w, h)
    - update(frame) returns (success, bbox) where bbox is (x, y, w, h)
    """

    def __init__(self, weights=None, arch='LightTrackM_Subnet'):
        """
        Initialize LightTrack tracker.

        Args:
            weights: Path to model weights (.pth file). If None, uses default mounted path.
            arch: Model architecture name (default: 'LightTrackM_Subnet')
        """
        self.arch = arch
        # Use default path if no weights specified
        if weights is None:
            self.weights = DEFAULT_WEIGHTS_PATH
        else:
            self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Will be initialized on first init() call
        self.model = None
        self.tracker = None
        self.state = None
        self.is_initialized = False

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the neural network model."""
        print("=" * 70)
        print("[LightTrack] Initializing tracker...")
        print(f"[LightTrack] Weights path: {self.weights}")
        print(f"[LightTrack] Architecture: {self.arch}")
        print(f"[LightTrack] Device: {self.device}")

        # Create model info
        self.siam_info = SiamInfo(stride=16)

        # Get path name for model (used in architecture loading)
        # LightTrackM_Subnet expects a path_name parameter
        path_name = 'back_04502514044521042540+cls_211000022+reg_100000111_ops_32'

        # Track if weights were loaded successfully
        self.weights_loaded = False

        try:
            # Load architecture
            print(f"[LightTrack] Loading architecture: {self.arch}")
            self.model = model_zoo.__dict__[self.arch](path_name, stride=self.siam_info.stride)

            # Check if weights file exists FIRST
            if self.weights:
                if os.path.exists(self.weights):
                    print(f"[LightTrack] Found weights file: {self.weights}")
                    self.model = load_pretrain(self.model, self.weights)
                    self.weights_loaded = True
                    print("=" * 70)
                    print("[LightTrack] ✓ WEIGHTS LOADED SUCCESSFULLY")
                    print("=" * 70)
                else:
                    print("=" * 70)
                    print(f"[LightTrack] ✗ ERROR: Weights file NOT FOUND!")
                    print(f"[LightTrack]   Path: {self.weights}")
                    print("[LightTrack]   Tracking will NOT work correctly!")
                    print("[LightTrack]   Please verify the weights path exists.")
                    print("=" * 70)
            else:
                print("=" * 70)
                print("[LightTrack] ✗ ERROR: No weights path specified!")
                print("[LightTrack]   Tracking will NOT work correctly!")
                print("=" * 70)

            # Set to evaluation mode and move to device
            # IMPORTANT: Force float32 - LightTrack doesn't support BFloat16
            self.model.eval()
            self.model = self.model.float()  # Ensure float32
            if self.device == 'cuda':
                self.model = self.model.cuda()
                print(f"[LightTrack] Model moved to CUDA")

            # Create tracker instance
            self.tracker = LighttrackCore(self.siam_info, even=0)
            print(f"[LightTrack] Tracker core initialized")

        except Exception as e:
            print(f"[LightTrack] ✗ ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def init(self, frame, bbox):
        """
        Initialize tracker with first frame and bounding box.

        Args:
            frame: First frame (numpy array, BGR expected from OpenCV)
            bbox: Bounding box as (x, y, w, h) - top-left corner and size
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        # Ensure frame is numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        print(f"[LightTrack] init() called with frame shape: {frame.shape}")
        print(f"[LightTrack] init() bbox (xywh): {bbox}")

        # Convert BGR to RGB (wrapper expects BGR input from OpenCV)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Check channel means to verify BGR format
            b_mean = frame[:,:,0].mean()
            g_mean = frame[:,:,1].mean()
            r_mean = frame[:,:,2].mean()
            print(f"[LightTrack] Input channels (BGR expected): B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"[LightTrack] Converted BGR -> RGB for neural network")
        else:
            frame_rgb = frame
            print(f"[LightTrack] Frame is not 3-channel, using as-is")

        # Parse bbox (x, y, w, h) -> center position and size
        x, y, w, h = [float(v) for v in bbox]
        target_pos = np.array([x + w/2, y + h/2])  # Center position
        target_sz = np.array([w, h])  # Width, height
        print(f"[LightTrack] Target center: {target_pos}, size: {target_sz}")

        # Initialize tracker with autocast disabled (LightTrack doesn't support BFloat16)
        with torch.amp.autocast('cuda', enabled=False):
            self.state = self.tracker.init(frame_rgb, target_pos, target_sz, self.model)
        self.is_initialized = True

        print(f"[LightTrack] ✓ Tracker initialized successfully!")
        return True

    def update(self, frame):
        """
        Track object in new frame.

        Args:
            frame: New frame (numpy array, BGR or RGB)

        Returns:
            tuple: (success, bbox) where bbox is (x, y, w, h) or None if tracking failed
        """
        if not self.is_initialized:
            return False, None

        if frame is None:
            return False, None

        # Ensure frame is numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        try:
            # Run tracking with autocast disabled (LightTrack doesn't support BFloat16)
            with torch.amp.autocast('cuda', enabled=False):
                self.state = self.tracker.track(self.state, frame_rgb)

            # Extract results
            target_pos = self.state['target_pos']
            target_sz = self.state['target_sz']

            # Convert center + size to (x, y, w, h)
            w, h = target_sz
            cx, cy = target_pos
            x = cx - w/2
            y = cy - h/2

            # Validate bbox
            if w <= 0 or h <= 0:
                return False, None

            bbox = (int(x), int(y), int(w), int(h))
            return True, bbox

        except Exception as e:
            print(f"[LightTrack] Tracking error: {e}")
            return False, None

    def reset(self):
        """Reset tracker state."""
        self.state = None
        self.is_initialized = False
