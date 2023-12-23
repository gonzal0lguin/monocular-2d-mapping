#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
import tf2_ros


class LocalOccupancyGridConcatenator:
    def __init__(self):
        rospy.init_node('local_occ_grid_concatenator', anonymous=True)
        self.global_occ_grid = None
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.local_occ_grid_sub = rospy.Subscriber('/local_occ_grid', OccupancyGrid, self.local_occ_grid_callback)
        self.global_occ_grid_pub = rospy.Publisher('/global_occ_grid', OccupancyGrid, queue_size=10)

        # Initialize global occupancy grid parameters
        self.resolution = 1 / 80.  #  (meters per pixel)
        self.map_size = int(100 / self.resolution) # map size (in pixels)
        self.global_occ_grid = np.ones((self.map_size, self.map_size), dtype=np.int8) * -1

        self.linear_update = 0.05
        self.angular_update = 0.02

        # Initialize robot pose
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0} # TODO: change initialization
        self.last_robot_pose = self.robot_pose.copy()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.loginfo(f"Started mapping node with map size of {self.map_size}x{self.map_size} [m2]")

    def odom_callback(self, odom_msg):
        # Update robot pose based on odometry
        q = odom_msg.pose.pose.orientation
        orientation_list = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.robot_pose['x'] = odom_msg.pose.pose.position.x
        self.robot_pose['y'] = odom_msg.pose.pose.position.y
        self.robot_pose['theta'] = yaw

    def local_occ_grid_callback(self, local_occ_grid_msg):
        # Transform local occupancy grid to global coordinates
        if self._check_for_update():
            local_occ_grid = np.array(local_occ_grid_msg.data).reshape((local_occ_grid_msg.info.height, local_occ_grid_msg.info.width))

            # Calculate the global position to update the global_occ_grid

            global_position = self.world_to_map_indices(0)
            self.update_map(self.global_occ_grid, local_occ_grid, global_position)
            
            # Publish the updated global occupancy grid
            self.last_robot_pose = self.robot_pose.copy()

            rospy.loginfo(f"Updated map at position x={self.robot_pose['x']:.2f}, y={self.robot_pose['y']:.2f}")

            self.publish_global_occ_grid()
        self.publish_transform()
        

    def update_map(self, map, sensor_data, pose):
        Ll, Hl = sensor_data.shape
        xR, yR, yawR = pose
        dx, dy = int(3.25 * 80), Ll // 2

        yawR *= -1

        sensor_data = np.rot90(sensor_data)

        for x_ in range(Ll):
            for y_ in range(Hl):
                if sensor_data[y_, x_] == -1: continue

                R = np.sqrt((Hl - y_ + dy)**2 + (x_ - dx)**2)
                phi = np.arctan2(x_ - dx, Hl - y_ + dy)

                rx = int(R * np.cos(yawR - phi)) - self.map_size // 2
                ry = int(R * np.sin(yawR - phi)) - self.map_size // 2
                
                i = xR + rx
                j = yR - ry

                if sensor_data[y_, x_] == 0:
                    map[j, i] = 0
                else:
                    map[j, i] += int(sensor_data[y_, x_])
        
                # clamp values to range 0, 100 (at this point we ignored unkown space "-1")
                map[j, i] = max(min(np.abs(map[j, i]), 100), 0)


    def world_to_map_indices(self, origin):
        # Convert world coordinates to map indices
        x_idx = int(self.robot_pose['x'] / self.resolution)
        y_idx = int(self.robot_pose['y'] / self.resolution) 

        return x_idx, y_idx, self.robot_pose['theta']


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

    def publish_transform(self):
        try:
            # Create a TransformStamped message
            transform = TransformStamped()

            # Header
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "map"
            transform.child_frame_id = "odom"  # Assuming "odom" and "map" frames are the same
            transform.transform.rotation.w = 1.0

            # Publish the static transform
            self.tf_broadcaster.sendTransform(transform)

        except rospy.ROSInterruptException:
            print('nooo')
            pass

    def _check_for_update(self):
        dx = np.linalg.norm([self.robot_pose['x'] - self.last_robot_pose['x'], self.robot_pose['y'] - self.last_robot_pose['y']])
        dtheta = np.abs(self.robot_pose['theta'] - self.last_robot_pose['theta'])

        if dx >= self.linear_update or dtheta >= self.angular_update:
            return True

        return False


if __name__ == '__main__':
    try:
        local_occ_grid_concatenator = LocalOccupancyGridConcatenator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
