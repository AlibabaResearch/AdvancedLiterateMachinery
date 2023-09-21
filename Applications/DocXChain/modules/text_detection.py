#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py

import sys
import numpy as np
import datetime
import time
import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class TextDetection(object):
    """
    Description:
      class definition of TextDetection module: 
      (1) algorithm interfaces for text detection

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch module
        if configs['from_modelscope_flag'] is True:
            self.text_detector = pipeline(Tasks.ocr_detection, model = configs['model_path'])  # text detection model from modelscope
        else:
            self.text_detector = None  # (20230811) currently we only support models from modelscope


    def __call__(self, image):
        """
        Description:
          detect all text instances (those virtually machine-identifiable) from the input image

        Parameters:
          image: the image to be processed, assume that it is a *full* image potentially containing text instances

        Return:
          result: detection result
        """

        # initialize
        result = None
        
        # perform text detection
        if self.text_detector is not None:
            # run the text detector
            det_result = self.text_detector(image)
            det_result = det_result['polygons']
        
            # sort detection result with coord
            det_result_list = det_result.tolist()
            det_result_list = sorted(det_result_list, key=lambda x: 0.01*sum(x[::2])/4+sum(x[1::2])/4)     
        
            result = np.array(det_result_list)

        return result

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.text_detector is not None:
            del self.text_detector

        return 