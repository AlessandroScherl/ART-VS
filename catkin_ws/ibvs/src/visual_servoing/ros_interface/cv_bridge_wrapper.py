"""
Wrapper for cv_bridge to handle compatibility issues
"""

import sys
import numpy as np
import cv2

try:
    # Try normal import first
    from cv_bridge import CvBridge
    cv_bridge_available = True
    print("Using native cv_bridge")
except (ImportError, TypeError) as e:
    print(f"Native cv_bridge failed: {e}")
    print("Using fallback cv_bridge implementation")
    cv_bridge_available = False
    
    # Fallback implementation
    class CvBridge:
        """Minimal CvBridge implementation for basic functionality"""
        
        def __init__(self):
            pass
        
        def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
            """Convert ROS image message to OpenCV image"""
            
            # Get the image data
            dtype = np.uint8
            if img_msg.encoding == "32FC1":
                dtype = np.float32
            elif img_msg.encoding == "16UC1":
                dtype = np.uint16
                
            # Reshape the data
            if img_msg.encoding in ["mono8", "8UC1"]:
                image = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width)
            elif img_msg.encoding in ["bgr8", "rgb8"]:
                image = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, 3)
                if img_msg.encoding == "rgb8" and desired_encoding == "bgr8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif img_msg.encoding == "bgr8" and desired_encoding == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif img_msg.encoding == "rgba8":
                image = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, 4)
            elif img_msg.encoding in ["32FC1", "16UC1"]:
                image = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width)
            else:
                raise NotImplementedError(f"Encoding {img_msg.encoding} not supported in fallback mode")
                
            return image
        
        def cv2_to_imgmsg(self, cv_image, encoding="passthrough"):
            """Convert OpenCV image to ROS image message"""
            from sensor_msgs.msg import Image
            
            img_msg = Image()
            img_msg.height = cv_image.shape[0]
            img_msg.width = cv_image.shape[1]
            
            if len(cv_image.shape) == 2:
                # Grayscale
                if cv_image.dtype == np.uint8:
                    img_msg.encoding = "mono8"
                elif cv_image.dtype == np.uint16:
                    img_msg.encoding = "16UC1"
                elif cv_image.dtype == np.float32:
                    img_msg.encoding = "32FC1"
            elif len(cv_image.shape) == 3:
                # Color
                if cv_image.shape[2] == 3:
                    if encoding == "rgb8":
                        img_msg.encoding = "rgb8"
                    else:
                        img_msg.encoding = "bgr8"
                elif cv_image.shape[2] == 4:
                    img_msg.encoding = "rgba8"
            
            img_msg.data = cv_image.tobytes()
            img_msg.step = len(img_msg.data) // img_msg.height
            
            return img_msg

# Export the CvBridge class
__all__ = ['CvBridge', 'cv_bridge_available']