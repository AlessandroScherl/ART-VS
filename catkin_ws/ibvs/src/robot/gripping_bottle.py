#!/usr/bin/env python3
"""
ROS2 node to move TCP down for bottle grasping via MoveIt Servo.
Publishes TwistStamped in tool0 frame — Servo handles the transform to base.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer as TF2Buffer, TransformListener as TF2TransformListener
import numpy as np
import time


class TCPMover(Node):
    def __init__(self):
        super().__init__('tcp_mover')

        # Setup TF2 listener (for pose monitoring only)
        self.tf_buffer = TF2Buffer()
        self.tf_listener = TF2TransformListener(self.tf_buffer, self)

        # Publisher to MoveIt Servo
        self.twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self.get_logger().info("TCP Mover initialized (MoveIt Servo, tool0 frame)")

    def get_current_tcp_pose(self):
        """Get current TCP position from TF tree."""
        try:
            from rclpy.time import Time
            trans = self.tf_buffer.lookup_transform('base', 'tool0', Time())
            pos = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]
            return pos
        except Exception as e:
            self.get_logger().error(f"Error looking up transform: {e}")
            return None

    def move_tcp_relative(self, dx=0.0, dy=0.0, dz=0.0, duration=2.0):
        """Move TCP by relative offset in tool0 frame.

        With MoveIt Servo (frame_id='tool0'), velocities are expressed
        directly in the tool frame — no manual transform needed.
        """
        current_pos = self.get_current_tcp_pose()
        if current_pos is not None:
            self.get_logger().info(
                f"Current TCP position: [{current_pos[0]:.3f}, "
                f"{current_pos[1]:.3f}, {current_pos[2]:.3f}]")

        self.get_logger().info(
            f"Moving TCP relative (tool0 frame): "
            f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, duration={duration:.1f}s")

        # Create TwistStamped in tool0 frame
        ts = TwistStamped()
        ts.header.frame_id = 'tool0'
        ts.twist.linear.x = dx / duration
        ts.twist.linear.y = dy / duration
        ts.twist.linear.z = dz / duration

        # Publish at ~100Hz for MoveIt Servo watchdog
        self.get_logger().info("Moving...")
        start_time = time.time()
        while time.time() - start_time < duration:
            ts.header.stamp = self.get_clock().now().to_msg()
            self.twist_pub.publish(ts)
            time.sleep(0.01)  # 100Hz

        # Stop motion
        ts.twist.linear.x = 0.0
        ts.twist.linear.y = 0.0
        ts.twist.linear.z = 0.0
        ts.header.stamp = self.get_clock().now().to_msg()
        self.twist_pub.publish(ts)

        # Wait for motion to settle
        time.sleep(0.5)
        final_pos = self.get_current_tcp_pose()
        if final_pos is not None:
            self.get_logger().info(
                f"Final TCP position: [{final_pos[0]:.3f}, "
                f"{final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        self.get_logger().info("Motion completed")


def main(args=None):
    rclpy.init(args=args)
    mover = TCPMover()
    # Spin to let TF buffer populate (sleep alone doesn't spin the node)
    tf_deadline = time.time() + 2.0
    while time.time() < tf_deadline:
        rclpy.spin_once(mover, timeout_sec=0.1)

    try:
        # Move 8cm down in TCP frame (z-axis of tool0)
        print("\nMoving down 8cm in TCP frame")
        mover.move_tcp_relative(dz=0.08, duration=2.0)
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        # Send zero twist to stop motion
        ts = TwistStamped()
        ts.header.stamp = mover.get_clock().now().to_msg()
        ts.header.frame_id = 'tool0'
        mover.twist_pub.publish(ts)
    except Exception as e:
        print(f"Error occurred: {e}")
        ts = TwistStamped()
        ts.header.stamp = mover.get_clock().now().to_msg()
        ts.header.frame_id = 'tool0'
        mover.twist_pub.publish(ts)
    finally:
        mover.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
