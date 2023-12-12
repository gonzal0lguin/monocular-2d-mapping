#!/usr/bin/env python3

import rospy
import rospkg
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2 as cv
import os


class BirdseyeToOccGrid:
    def __init__(self):

        rospy.init_node('birdseye_to_occgrid', anonymous=True)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/bev_image/gray', Image, self.image_callback)
        self.occ_grid_pub = rospy.Publisher('/local_occ_grid', OccupancyGrid, queue_size=10)

        self.meter_per_pixel = 1. / 80 
        self.x_offset = 3.25
        self.HEIGHT, self.WIDTH = 480, 480

        rospy.loginfo('Initialized occupancy grid node')


    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        # Your image processing code (e.g., ray tracing, thresholding, etc.)
        # Convert the processed image to occupancy grid
        occupancy_grid = self.image_to_occ_grid(cv_image)

        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header = data.header
        occ_grid_msg.header.frame_id = 'base_link'
        occ_grid_msg.info.resolution = self.meter_per_pixel  # Set your resolution (meters per pixel)
        occ_grid_msg.info.width = occupancy_grid.shape[1]
        occ_grid_msg.info.height = occupancy_grid.shape[0]
        occ_grid_msg.info.origin.position.x = 3.25  # Set your origin position
        occ_grid_msg.info.origin.position.y = - self.WIDTH / 2 * self.meter_per_pixel
        occ_grid_msg.data = occupancy_grid.flatten()

        self.occ_grid_pub.publish(occ_grid_msg)

    def image_to_occ_grid(self, image):
        # Extract relevant parameters from the bird's eye view image

        # Convert image to binary mask (1 for drivable area, 0 for occupied)
        binary_mask = (image == 1).astype(np.uint8)
        binary_mask = np.rot90(np.rot90(binary_mask.T))
        # Get image dimensions
        height, width = binary_mask.shape

        # Initialize occupancy grid
        occ_grid = np.zeros((height, width), dtype=np.int8)

        # print(height, width)

        # # Iterate through each pixel in the binary mask
        # for y in range(height):
        #     for x in range(width):
        #         # Skip pixels with no drivable area
        #         if binary_mask[y, x] == 0:
        #             continue

        #         # Calculate distance and angle from the robot to the current pixel
        #         # xr, yr = self.image_to_robot_transform(y, x)
        #         # distance = np.sqrt((xr)**2 + (yr)**2)

        #         # angle = np.arctan2(yr, xr)

        #         # Convert polar coordinates to grid coordinates
        #         grid_x = x#int((distance / self.meter_per_pixel) * np.cos(angle))
        #         grid_y = y#int((distance / self.meter_per_pixel) * np.sin(angle))

        #         # Update occupancy grid, considering unknown areas behind occupied space
        #         if 0 <= grid_x < width and 0 <= grid_y < height:
        #             occ_grid[grid_y, grid_x] = 100  # Occupied
        #             occ_grid[grid_y+1:, grid_x:] = -1  # Unknown

        return 100 * (np.ones_like(binary_mask) - binary_mask)

    def image_to_robot_transform(self, px_i, px_j):
        x = (self.HEIGHT - px_i) * self.meter_per_pixel
        y = (px_j - self.WIDTH / 2)  * self.meter_per_pixel
        # print(x, y)

        return x, y

if __name__ == '__main__':
    try:
        birdseye_to_occgrid = BirdseyeToOccGrid()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
