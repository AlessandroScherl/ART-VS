#!/usr/bin/env python

import rospy
import tf
import math
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState


class GazeboBroadcaster:

    def __init__(self):
        self.cmd_vel_sub = rospy.Subscriber('/camera_vel', Twist, self.velCallback, queue_size=1)
        self.robot_set_state = ModelState()
        self.robot_set_state.model_name = 'realsense2_camera'
        self.robot_set_state.reference_frame = 'base_link'

        self.listener = tf.TransformListener()
        
        # Wait for TF to be ready
        rospy.sleep(1.0)

    def transform_camera_vel_to_base(self, camera_twist):
        """
        Transform velocity commands from camera_color_optical_frame to base_link
        using proper velocity kinematics transformation
        """
        try:
            # Look up transform from camera_color_optical_frame to base_link
            # This gives us the pose of base_link in camera frame
            (trans, rot) = self.listener.lookupTransform('/base_link', '/camera_color_optical_frame', rospy.Time(0))
            
            # Convert quaternion to rotation matrix (R_base_cam)
            # This rotates vectors from camera frame to base frame
            rotation_matrix = tf.transformations.quaternion_matrix(rot)[:3, :3]
            
            # Extract velocities from camera frame
            v_cam = np.array([camera_twist.linear.x, camera_twist.linear.y, camera_twist.linear.z])
            w_cam = np.array([camera_twist.angular.x, camera_twist.angular.y, camera_twist.angular.z])
            
            # Transform angular velocity first: w_base = R_base_cam * w_cam
            w_base = rotation_matrix.dot(w_cam)
            
            # Transform linear velocity using velocity kinematics:
            # v_base = R_base_cam * v_cam + w_base × r_base_cam
            # where r_base_cam is position of camera origin relative to base origin
            r_base_cam = np.array(trans)
            v_base = rotation_matrix.dot(v_cam) + np.cross(w_base, r_base_cam)
            
            # Create output Twist message
            base_twist = Twist()
            base_twist.linear.x = v_base[0]
            base_twist.linear.y = v_base[1] 
            base_twist.linear.z = v_base[2]
            base_twist.angular.x = w_base[0]
            base_twist.angular.y = w_base[1]
            base_twist.angular.z = w_base[2]
            
            return base_twist
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn('TF transform failed: %s' % str(e))
            # Fallback to static transformation if TF fails
            return self.static_transform_camera_to_base(camera_twist)

    def static_transform_camera_to_base(self, camera_twist):
        """
        Static transformation using known camera offset and orientation.
        Camera offset from base_link: Y=32.5mm, Z=12.5mm
        Camera rotation: -90° around X, then -90° around Z (from your TF data)
        """
        # Camera position relative to base (in base frame)
        # Using 12.5mm as requested, not the 15mm from TF
        r_base_cam = np.array([0.0, 0.0125, 0.0])  # [x, y, z] in meters
        
        # Rotation matrix from camera_color_optical_frame to base_link
        # Based on your TF: RPY = [-90°, 0°, -90°]
        # This is: Rz(-90) * Ry(0) * Rx(-90)
        R_base_cam = np.array([
            [ 0,  0,  1],  # X_base = Z_cam
            [-1,  0,  0],  # Y_base = -X_cam
            [ 0, -1,  0]   # Z_base = -Y_cam
        ])
        
        # Extract velocities
        v_cam = np.array([camera_twist.linear.x, camera_twist.linear.y, camera_twist.linear.z])
        w_cam = np.array([camera_twist.angular.x, camera_twist.angular.y, camera_twist.angular.z])
        
        # Transform angular velocity
        w_base = R_base_cam.dot(w_cam)
        
        # Transform linear velocity
        v_base = R_base_cam.dot(v_cam) + np.cross(w_base, r_base_cam)
        
        # Create output
        base_twist = Twist()
        base_twist.linear.x = v_base[0]
        base_twist.linear.y = v_base[1]
        base_twist.linear.z = v_base[2]
        base_twist.angular.x = w_base[0]
        base_twist.angular.y = w_base[1]
        base_twist.angular.z = w_base[2]
        
        return base_twist

    def velCallback(self, data):
        # Use proper kinematic transformation instead of simple compensation
        base_twist = self.transform_camera_vel_to_base(data)
        
        # Get current robot state
        try:
            self.model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.robot_state = self.model_state_service('realsense2_camera', 'base_link')
            self.robot_set_state.pose = self.robot_state.pose
        except rospy.ServiceException:
            rospy.loginfo('GetModelState service failed')

        # Set transformed velocity
        self.robot_set_state.twist = base_twist
        
        # Apply to Gazebo
        try:
            self.model_state_set_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            result = self.model_state_set_service(self.robot_set_state)
        except rospy.ServiceException:
            rospy.loginfo('SetModelState service failed')


def main():
    rospy.init_node('frame_controller')
    rate = rospy.Rate(50)  # 50 Hz

    gazebo_broadcaster = GazeboBroadcaster()

    rospy.spin()  # Use spin() instead of manual loop

if __name__ == '__main__':
    main()
