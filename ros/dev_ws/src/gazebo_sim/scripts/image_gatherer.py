#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from cv_bridge import CvBridge
import rospkg

import cv2 as cv
import sys
import os
import numpy as np

class TrayectoryRecorder(object):
    def __init__(self, poses_filename):
        self.poses_saved = []
        self.save_state = False
        self.last_x, self.last_y, self.last_w = 0., 0., 0.
        self.poses_filename = poses_filename

        self.sub = rospy.Subscriber('/ground_truth/state', Odometry, self.gt_callback)
        self.save_sub = rospy.Subscriber('/trayectory_recorder/save_state', Empty, self.write_poses_to_file)

    def save_cb(self, msg):
        self.save_state = not self.save_state
        rospy.loginfo(f"save_state switched to {self.save_state}")

    def gt_callback(self, gt_msg):
        x, y, w = gt_msg.pose.pose.position.x, gt_msg.pose.pose.position.y, gt_msg.pose.pose.orientation.w
        if abs(x - self.last_x) > 1e-1 or abs(y - self.last_y) > 1e-1 or abs(w - self.last_w) > 1e-1:
            msg = Pose(
                position=gt_msg.pose.pose.position,
                orientation=gt_msg.pose.pose.orientation
            )
            self.last_x, self.last_y, self.last_w = x, y, w
            
            self.poses_saved.append(msg)

            rospy.loginfo(f"New pose saved. adding to a total of {len(self.poses_saved)} poses.")
        else:
            pass
    
    def write_poses_to_file(self, msg):
        path = rospkg.RosPack().get_path('gazebo_sim')
        if not os.path.exists(os.path.join(path, 'trayectories')):
            os.makedirs(os.path.join(path, 'trayectories'))

        arr = np.asarray(self.poses_saved)
        np.save(os.path.join(path, f'trayectories/{self.poses_filename}.npy'), arr)
        
        rospy.loginfo(f'saved array with {len(arr)} poses.')





class RandomPoseSetter(object):
    def __init__(self, poses_file, out_files_dir, n_poses=10):
        try:
            self.poses = np.load(
                os.path.join(rospkg.RosPack().get_path('gazebo_sim'), f'trayectories/{poses_file}.npy'),
                allow_pickle=True
                )
            rospy.loginfo(f'Loaded array with {len(self.poses)} poses')

        except:
            raise ValueError(f"File {poses_file} does not exist.")
    
        if not os.path.exists(out_files_dir):
            os.makedirs(out_files_dir)

        self.out_path = out_files_dir
        self.n_poses = n_poses if n_poses < len(self.poses) else len(self.poses)
        
        self.bridge = CvBridge()

        rospy.loginfo('Waiting for camera to be ready...')
        _ = rospy.wait_for_message('/camera/image_raw', Image, timeout=60)
        rospy.loginfo('Camera ready.')

    def save_image(self, img, filename):
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        cv.imwrite(os.path.join(self.out_path, filename), cv_image)

        rospy.loginfo(f"Image saved as {filename}")

    def set_model_pose(self, model_name, pose):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

            # Create ModelState message
            model_state = ModelState()
            model_state.model_name = model_name
            model_state.pose = pose
            model_state.reference_frame = 'world'

            # Call the service to set the model state
            response = set_model_state(model_state)

            return response.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False

    def set_random_pose(self, idx):
        
        # Generate random pose
        pose = self.poses[idx]

        # Set the pose in Gazebo
        success = self.set_model_pose('panther', pose)

        if success:
            print(f"Robot spawned at position: {pose.position.x}, {pose.position.y}")
        else:
            print("Failed to spawn the robot.")

    def run_pose_setter(self):
        for i in range(len(self.poses)):
            self.set_random_pose(i)
            
            rospy.sleep(.1)
            
            img = rospy.wait_for_message('/camera/image_raw', Image)
            self.save_image(img=img, filename=f"img_{i:05d}.png")
            
            rospy.sleep(.1)

if __name__ == '__main__':
    rospy.init_node('path_replicator_node')
    type           = rospy.get_param("type")
    world          = rospy.get_param("world") # raw or segmented 
    poses_filename = rospy.get_param("poses_filename")

    print(type)

    if type == 'tr':
        tr = TrayectoryRecorder(poses_filename=poses_filename)
    
    else:
        print(os.path.join(os.getcwd(), f'images/images_{world}'))
        rps = RandomPoseSetter(poses_file=poses_filename, out_files_dir=os.path.join(rospkg.RosPack().get_path('gazebo_sim'), f'images/images_{world}'))
        rps.run_pose_setter()
    
    rospy.spin()
    