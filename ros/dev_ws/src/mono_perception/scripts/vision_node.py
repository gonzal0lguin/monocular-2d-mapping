#!/usr/bin/env python3

import rospy
from unet import UNet
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image as PILimg
from utils.bev import BEV
import os
import rospkg

import matplotlib.pyplot as plt

# TODO: 
#   - prettify
#   - Add parameters from param server
#   - Add second model
#   - Fix class segmentation. DONE
#   - Use independent paths for model and transform import

def _preprocess_unet(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=PILimg.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img


class VisionNode(object):
    def __init__(self):
        rospy.init_node("image_segmenter_node", anonymous=True)

        # ROS Image subscriber and publisher
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        
        self.segmented_pub = rospy.Publisher("/segmented_image", Image, queue_size=1)
        self.bev_pub_rgb   = rospy.Publisher("/bev_image/rgb",   Image, queue_size=1)
        self.bev_pub       = rospy.Publisher("/bev_image/gray",  Image, queue_size=1)

        transform_pth = os.path.join(rospkg.RosPack().get_path('mono_perception'), "configs/top_down_transform.npy")
        self._bev_transformer = BEV(np.load(transform_pth, allow_pickle=True).item())

        # Torch model initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        rospy.loginfo(f"USING DEVICE {self.device}")
        
        self.model = self.load_segmentation_model()
        # CV Bridge for image conversion
        self.bridge = CvBridge()

        rospy.loginfo('Done initalizing Vision Node')

    def load_segmentation_model(self):
        ckp_path = os.path.join(rospkg.RosPack().get_path('segmentation_models'), 'models/UNet/checkpoints/checkpoint_epoch200_bs16.pth')
        try:
            checkpoint = torch.load(ckp_path)
            checkpoint.pop('mask_values')
            model = UNet(n_channels=3, n_classes=6, bilinear=False)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
        
        except:
            rospy.logerr("Could not load Model!")

        return model
    
    def predict_frame(self, img):
        self.model.eval()

        if not img.is_cuda and next(self.model.parameters()).is_cuda: 
            img  = img.to(self.device) # match both devices

        pred = self.model(torch.unsqueeze(img, dim=0)) # add dataloader dim 1
        pred = torch.argmax(pred, dim=1)

        return pred.detach().cpu().numpy()[0]

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            img_tensor = _preprocess_unet(PILimg.fromarray(cv_image), scale=.5)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
            
            # Perform segmentation
            with torch.no_grad():
                segmented_image = self.predict_frame(img_tensor)

            segmented_image = cv2.resize(segmented_image.astype(np.uint8), (640, 480), interpolation=cv2.INTER_AREA)
            
            # Convert segmented image back to ROS Image
            segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image.astype(np.uint8), encoding="mono8")
            segmented_image_rgb_msg = self.bridge.cv2_to_imgmsg(self.map_classes(segmented_image), encoding="rgb8")
            
            segmented_image += 1 # shift classes to have unkown space after BEV trans

            bev_image = self._bev_transformer.apply_bev(segmented_image.astype(np.uint8))
            bev_image_msg = self.bridge.cv2_to_imgmsg(bev_image, encoding="mono8")
            
            # Publish the segmented image
            self.segmented_pub.publish(segmented_image_msg)
            self.bev_pub_rgb.publish(segmented_image_rgb_msg)
            self.bev_pub.publish(bev_image_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def map_classes(self, img):
        img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img_rgb[img==0, 2] = 255
        img_rgb[img==1, 1] = 255
        img_rgb[img==2, 1] = 255
        img_rgb[img==2, 2] = 255
        img_rgb[img==3, :] = 127
        img_rgb[img==4, 0] = 255
        return img_rgb



if __name__ == "__main__":
    image_segmenter_node = VisionNode()
    rospy.sleep(5)
    rospy.loginfo("Running image segmenter")
    rospy.spin()