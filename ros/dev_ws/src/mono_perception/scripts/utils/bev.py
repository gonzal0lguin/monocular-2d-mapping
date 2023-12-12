import numpy as np
import cv2 as cv

class BEV(object):
    def __init__(self, transform, size=(480, 480)):
        self.M = transform['M']
        self.M_inv = transform['M_inv']
        self.origin = transform['origin']
        self.pixel_per_meter = transform['pixel_per_meter']

        self.WIDTH, self.HEIGHT = size

    def apply_bev(self, img):
        return cv.warpPerspective(img, self.M, (self.WIDTH, self.HEIGHT), flags=cv.INTER_LINEAR)

    def apply_inverse_bev(self, img_bev):
        return cv.warpPerspective(img_bev, self.M_inv, (self.WIDTH, self.HEIGHT))