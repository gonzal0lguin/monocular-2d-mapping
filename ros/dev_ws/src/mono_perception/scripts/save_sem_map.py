#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

import numpy as np
import cv2 as cv


class SemMapSaver:
    def __init__(self):

        rospy.init_node('sem_map_saver', anonymous=True)
        
        self.bridge = CvBridge()
        self.save_sub = rospy.Subscriber('/global_occ_grid', OccupancyGrid, self.save_cb)


    def save_cb(self, map):

        rospy.loginfo(f"Received map!")

        sem_map = np.array(map.data).reshape((map.info.height, map.info.width))

        cv.imwrite('sem_map.jpeg', sem_map)
        
        rospy.loginfo(f"Saved map!")
        rospy.signal_shutdown('exiting')

        

if __name__ == '__main__':
    try:
        saver = SemMapSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
