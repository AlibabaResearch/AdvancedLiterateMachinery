#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py

import sys
import numpy as np

from modules.text_detection import TextDetection
from modules.text_recognition import TextRecognition

class GeneralTextReading(object):
    """
    Description:
      class definition of GeneralTextReading pipeline: 
      (1) algorithm interfaces for general text reading (detection + recognition)
      (2) document layout and structure are not taken into consideration

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch pipiline
        self.text_detection_module = TextDetection(configs['text_detection_configs'])
        self.text_recognition_module = TextRecognition(configs['text_recognition_configs'])

    def __call__(self, image):
        """
        Description:
          detect and recognize all text instances (those virtually machine-identifiable) from the input image

        Parameters:
          image: the image to be processed, assume that it is a *full* image potentially containing text instances

        Return:
          final_result: text reading result
        """

        # initialize
        final_result = []
        det_result = None
        rec_result = None

        # perform text detection and recognition successively
        if image is not None:
            det_result = self.text_detection_module(image)
            rec_result = self.text_recognition_module(image, det_result)    

        # assembling
        for i in range(det_result.shape[0]):
            item = {}
            item['position'] = det_result[i].tolist()
            item['content'] = rec_result[i]['text']
            final_result.append(item)

        return final_result

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.text_detection_module is not None:
            self.text_detection_module.release()
        
        if self.text_recognition_module is not None:
            self.text_recognition_module.release()

        return 

