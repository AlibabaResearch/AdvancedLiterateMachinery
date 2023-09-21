#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import datetime
import time
import cv2

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR + '/../../../DocumentUnderstanding/DocXLayout')
from main import DocXLayoutInfo, DocXLayoutPredictor

class LayoutAnalysis(object):
    """
    Description:
      class definition of LayoutAnalysis module: 
      (1) algorithm interfaces for layout analysis

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch module
        if configs['from_modelscope_flag'] is True:
            self.layout_analyser = None  # (20230912) currently we only support models from Advanced Literate Machinery (https://github.com/AlibabaResearch/AdvancedLiterateMachinery)
        else:
            params = {
                'model_file': configs['model_path'],
                'debug': 0, # 1: save vis results, 0: don't save
            }
    
            self.layout_analyser = DocXLayoutPredictor(params) 


    def __call__(self, image):
        """
        Description:
          detect all layout regions (those virtually machine-identifiable) from the input image

        Parameters:
          image: the image to be processed, assume that it is a *full* image

        Return:
          result: layout analysis result
        """

        # initialize
        result = None
        
        # perform text detection
        if self.layout_analyser is not None:
            # run the layout analyser
            la_result = self.layout_analyser(image)
            result = la_result

        return result

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.layout_analyser is not None:
            del self.layout_analyser

        return 