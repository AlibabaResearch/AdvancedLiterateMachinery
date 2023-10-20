#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py

import sys
import numpy as np
import math
import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class TextRecognition(object):
    """
    Description:
      class definition of TextRecognition module: 
      (1) algorithm interfaces for text recognition

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch module
        if configs['from_modelscope_flag'] is True:
            self.text_recognizer = pipeline(Tasks.ocr_recognition, model = configs['model_path'])  # text recognition model from modelscope
        else:
            self.text_recognizer = None  # (20230811) currently we only support models from modelscope

    def __call__(self, image, detections):
        """
        Description:
          recognize each text instance (assume that text detection has been perfomed in advance)

        Parameters:
          image: the image to be processed (assume that it is a *full* image potentially containing text instances)

        Return:
          result: recognition result
        """

        # initialize
        result = None
        
        # perform text recognition
        if self.text_recognizer is not None:
            # recognize the text instances one by one
            result = []
            for i in range(detections.shape[0]):  # this part can be largely accelerated via parallelization (leave for future work)
                pts = self.order_point(detections[i])
                image_crop = self.crop_image(image, pts)
                rec = self.text_recognizer(image_crop)
                result.append(rec)

        return result

    def recognize_cropped_image(self, cropped_image):
        """
        Description:
          recognize the text instance within the cropped image (assume that text detection and sub image cropping have been perfomed in advance)

        Parameters:
          cropped_image: the *cropped* image to be processed

        Return:
          result: recognition result
        """

        # initialize
        result = None
        
        # perform text recognition
        if self.text_recognizer is not None:
            # recognize the text instance
            result = self.text_recognizer(cropped_image)

        return result

    def order_point(self, coor):

        arr = np.array(coor).reshape([4, 2])
        sum_ = np.sum(arr, 0)
        centroid = sum_ / arr.shape[0]
        theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
        sort_points = arr[np.argsort(theta)]
        sort_points = sort_points.reshape([4, -1])

        if sort_points[0][0] > centroid[0]:
            sort_points = np.concatenate([sort_points[3:], sort_points[:3]])

        sort_points = sort_points.reshape([4, 2]).astype('float32')
    
        return sort_points
    
    def crop_image(self, image, position):
        def distance(x1,y1,x2,y2):
            return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
        
        position = position.tolist()
        for i in range(4):
            for j in range(i+1, 4):
                if(position[i][0] > position[j][0]):
                    tmp = position[j]
                    position[j] = position[i]
                    position[i] = tmp
    
        if position[0][1] > position[1][1]:
            tmp = position[0]
            position[0] = position[1]
            position[1] = tmp

        if position[2][1] > position[3][1]:
            tmp = position[2]
            position[2] = position[3]
            position[3] = tmp

        x1, y1 = position[0][0], position[0][1]
        x2, y2 = position[2][0], position[2][1]
        x3, y3 = position[3][0], position[3][1]
        x4, y4 = position[1][0], position[1][1]

        corners = np.zeros((4,2), np.float32)
        corners[0] = [x1, y1]
        corners[1] = [x2, y2]
        corners[2] = [x4, y4]
        corners[3] = [x3, y3]

        img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
        img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

        corners_trans = np.zeros((4,2), np.float32)
        corners_trans[0] = [0, 0]
        corners_trans[1] = [img_width - 1, 0]
        corners_trans[2] = [0, img_height - 1]
        corners_trans[3] = [img_width - 1, img_height - 1]

        transform = cv2.getPerspectiveTransform(corners, corners_trans)
        dst = cv2.warpPerspective(image, transform, (int(img_width), int(img_height)))
        
        return dst

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.text_recognizer is not None:
            del self.text_recognizer

        return 