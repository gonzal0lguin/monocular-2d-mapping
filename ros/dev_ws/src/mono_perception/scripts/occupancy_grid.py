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

        self.ponderator = np.linspace(0.6, 0.8, self.WIDTH)

        self.ponderator = np.rot90(np.rot90(np.tile(self.ponderator, (self.HEIGHT, 1))))
        rospy.loginfo('Initialized occupancy grid node')


    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        # Convert the processed image to occupancy grid
        occupancy_grid = self.image_to_occ_grid(cv_image)
 
        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header = data.header
        occ_grid_msg.header.frame_id = 'base_link'
        occ_grid_msg.info.resolution = self.meter_per_pixel  # (meters per pixel)
        occ_grid_msg.info.width = occupancy_grid.shape[1]
        occ_grid_msg.info.height = occupancy_grid.shape[0]
        occ_grid_msg.info.origin.position.x = 3.25  # Set your origin position
        occ_grid_msg.info.origin.position.y = -self.WIDTH / 2 * self.meter_per_pixel
        occ_grid_msg.data = occupancy_grid.flatten()

        self.occ_grid_pub.publish(occ_grid_msg)

    def image_to_occ_grid(self, image):

        # Convert image to binary mask (1 for drivable area, 0 for occupied)
        binary_mask = (image == 2).astype(np.uint8) + (image == 0).astype(np.uint8) 
        binary_mask = np.rot90(np.rot90(binary_mask.T))
        unkown_mask = np.rot90(np.rot90((image == 0).astype(np.uint8).T))
        # Get image dimensions
        height, width = binary_mask.shape

        # Initialize occupancy grid
        occ_grid = np.zeros((height, width), dtype=np.int8)

        return (100 * (np.ones_like(binary_mask) - binary_mask) * self.ponderator - unkown_mask).astype(np.int8) 

    def image_to_robot_transform(self, px_i, px_j):
        x = (self.HEIGHT - px_i) * self.meter_per_pixel
        y = (px_j - self.WIDTH / 2)  * self.meter_per_pixel

        return x, y


if __name__ == '__main__':
    try:
        birdseye_to_occgrid = BirdseyeToOccGrid()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
