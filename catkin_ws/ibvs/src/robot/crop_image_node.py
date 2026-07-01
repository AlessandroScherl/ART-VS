#!/usr/bin/env python3
"""
ROS2 node to crop RealSense camera images from 1920x1080 to 1440x1080.
Subscribes to the original RGB and aligned depth images and republishes cropped versions.

Usage (Real Robot):
    python3 crop_image_node.py

This node is required for real robot operation where the RealSense camera
outputs 1920x1080 but the visual servoing system expects 1440x1080.

IMPORTANT: Both RGB and depth must be cropped with the same offsets to ensure
feature point coordinates in the cropped RGB map correctly to depth values.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ImageCropper(Node):
    def __init__(self):
        super().__init__('image_cropper')

        # Parameters
        self.declare_parameter('input_width', 1920)
        self.declare_parameter('input_height', 1080)
        self.declare_parameter('output_width', 1440)
        self.declare_parameter('output_height', 1080)

        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.output_width = self.get_parameter('output_width').value
        self.output_height = self.get_parameter('output_height').value

        # Calculate crop offsets (center crop)
        self.x_offset = (self.input_width - self.output_width) // 2
        self.y_offset = (self.input_height - self.output_height) // 2

        self.get_logger().info(
            f"Cropping from {self.input_width}x{self.input_height} "
            f"to {self.output_width}x{self.output_height}")
        self.get_logger().info(f"Offset: x={self.x_offset}, y={self.y_offset}")

        self.bridge = CvBridge()

        # QoS for image topics (RELIABLE + TRANSIENT_LOCAL, matching realsense2_camera)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # QoS for camera_info topics (RELIABLE + VOLATILE, matching realsense2_camera)
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # RGB Publishers
        self.image_pub = self.create_publisher(Image, '/camera/color/image_cropped', 1)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info_cropped', 1)

        # Depth Publishers (aligned depth at same resolution as color)
        self.depth_pub = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_cropped', 1)
        self.depth_info_pub = self.create_publisher(
            CameraInfo, '/camera/aligned_depth_to_color/camera_info_cropped', 1)

        # RGB Subscribers (realsense2_camera publishes under /camera/camera/ namespace)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self.image_callback, image_qos)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self.info_callback, info_qos)

        # Depth Subscribers (aligned depth is at color resolution: 1920x1080)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                 self.depth_callback, image_qos)
        self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info',
                                 self.depth_info_callback, info_qos)

        self.camera_info = None
        self.depth_received = False
        self.get_logger().info(
            "Subscribing to /camera/camera/color/image_raw -> publishing /camera/color/image_cropped")
        self.get_logger().info(
            "Subscribing to /camera/camera/aligned_depth_to_color/image_raw -> "
            "publishing /camera/aligned_depth_to_color/image_cropped")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Crop (numpy view - no memory copy!)
            cropped = cv_image[
                self.y_offset:self.y_offset + self.output_height,
                self.x_offset:self.x_offset + self.output_width
            ]

            cropped_msg = self.bridge.cv2_to_imgmsg(cropped, encoding=msg.encoding)
            cropped_msg.header = msg.header

            self.image_pub.publish(cropped_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def info_callback(self, msg):
        """Adjust camera info for cropped image (updates principal point)."""
        cropped_info = CameraInfo()
        cropped_info.header = msg.header
        cropped_info.height = self.output_height
        cropped_info.width = self.output_width
        cropped_info.distortion_model = msg.distortion_model
        cropped_info.d = list(msg.d)

        # Adjust intrinsic matrix K (principal point shifts by offset)
        K = list(msg.k)
        K[2] -= self.x_offset  # cx
        K[5] -= self.y_offset  # cy
        cropped_info.k = K

        # Adjust rectification matrix (usually identity, copy as-is)
        cropped_info.r = list(msg.r)

        # Adjust projection matrix P
        P = list(msg.p)
        P[2] -= self.x_offset  # cx
        P[6] -= self.y_offset  # cy
        cropped_info.p = P

        cropped_info.binning_x = msg.binning_x
        cropped_info.binning_y = msg.binning_y
        cropped_info.roi = msg.roi

        self.info_pub.publish(cropped_info)

    def depth_callback(self, msg):
        """Crop depth image with same offsets as RGB to maintain coordinate alignment."""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if not self.depth_received:
                self.get_logger().info(
                    f"First depth image received: {cv_depth.shape[1]}x{cv_depth.shape[0]}")
                self.depth_received = True

            # Crop with same offsets as RGB (center crop)
            cropped = cv_depth[
                self.y_offset:self.y_offset + self.output_height,
                self.x_offset:self.x_offset + self.output_width
            ]

            cropped_msg = self.bridge.cv2_to_imgmsg(cropped, encoding=msg.encoding)
            cropped_msg.header = msg.header

            self.depth_pub.publish(cropped_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def depth_info_callback(self, msg):
        """Adjust depth camera info for cropped image."""
        cropped_info = CameraInfo()
        cropped_info.header = msg.header
        cropped_info.height = self.output_height
        cropped_info.width = self.output_width
        cropped_info.distortion_model = msg.distortion_model
        cropped_info.d = list(msg.d)

        K = list(msg.k)
        K[2] -= self.x_offset  # cx
        K[5] -= self.y_offset  # cy
        cropped_info.k = K

        cropped_info.r = list(msg.r)

        P = list(msg.p)
        P[2] -= self.x_offset  # cx
        P[6] -= self.y_offset  # cy
        cropped_info.p = P

        cropped_info.binning_x = msg.binning_x
        cropped_info.binning_y = msg.binning_y
        cropped_info.roi = msg.roi

        self.depth_info_pub.publish(cropped_info)


def main(args=None):
    rclpy.init(args=args)
    node = ImageCropper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
