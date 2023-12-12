#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from tf.transformations import euler_from_quaternion

class LocalOccupancyGridConcatenator:
    def __init__(self):
        rospy.init_node('local_occ_grid_concatenator', anonymous=True)
        self.global_occ_grid = None
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.local_occ_grid_sub = rospy.Subscriber('/local_occ_grid', OccupancyGrid, self.local_occ_grid_callback)
        self.global_occ_grid_pub = rospy.Publisher('/global_occ_grid', OccupancyGrid, queue_size=10)

        # Initialize global occupancy grid parameters
        self.resolution = 1 / 80.  # Set your resolution (meters per pixel)
        self.map_size = int(100 / self.resolution) # Set your map size (in pixels)
        self.global_occ_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)

        # Initialize robot pose
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}

    def odom_callback(self, odom_msg):
        # Update robot pose based on odometry
        orientation_q = odom_msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        self.robot_pose['x'] = odom_msg.pose.pose.position.x
        self.robot_pose['y'] = odom_msg.pose.pose.position.y
        self.robot_pose['theta'] = yaw

    def local_occ_grid_callback(self, local_occ_grid_msg):
        # Transform local occupancy grid to global coordinates
        local_occ_grid = np.array(local_occ_grid_msg.data).reshape((local_occ_grid_msg.info.height, local_occ_grid_msg.info.width))
        local_occ_grid_size = local_occ_grid.shape
        local_occ_grid_origin = np.array([local_occ_grid_msg.info.origin.position.x, local_occ_grid_msg.info.origin.position.y])

        # Calculate the global position to update the global_occ_grid
        global_position = self.world_to_map_indices(local_occ_grid_origin, local_occ_grid_size)

        # Update global occupancy grid with the local one
        print(self.global_occ_grid[
            0:480, 0:480
        ].shape)

        self.global_occ_grid[
            abs(global_position[0]):abs(global_position[0]) + local_occ_grid_size[0],
            global_position[1]:global_position[1] + local_occ_grid_size[1]
        ] = local_occ_grid

        # Publish the updated global occupancy grid
        self.publish_global_occ_grid()

    def world_to_map_indices(self, origin, size):
        # Convert world coordinates to map indices
        x_idx = int((self.robot_pose['x'] - origin[0]) / self.resolution)
        y_idx = int((self.robot_pose['y'] - origin[1]) / self.resolution)

        return x_idx, y_idx

    def publish_global_occ_grid(self):
        # Publish the updated global occupancy grid
        global_occ_grid_msg = OccupancyGrid()
        global_occ_grid_msg.header.stamp = rospy.Time.now()
        global_occ_grid_msg.header.frame_id = 'map'
        global_occ_grid_msg.info.resolution = self.resolution
        global_occ_grid_msg.info.width = self.map_size
        global_occ_grid_msg.info.height = self.map_size
        global_occ_grid_msg.info.origin.position.x = -self.map_size * self.resolution / 2.0
        global_occ_grid_msg.info.origin.position.y = -self.map_size * self.resolution / 2.0
        global_occ_grid_msg.data = self.global_occ_grid.flatten().tolist()

        self.global_occ_grid_pub.publish(global_occ_grid_msg)

if __name__ == '__main__':
    try:
        local_occ_grid_concatenator = LocalOccupancyGridConcatenator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
