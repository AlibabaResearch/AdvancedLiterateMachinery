#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://www.modelscope.cn/studios/damo/cv_table-ocr/file/view/master/app.py 

import sys
import numpy as np
import datetime
import time
import cv2

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class TableStructureRecognition(object):
    """
    Description:
      class definition of TableStructureRecognition module:
      (1) algorithm interfaces for table structure recognition
      (2) only tables with *visible borders* are supported currently (20230815)
      (3) only concerned with table structures, while textual or other contents are not considered in this module (20230912)

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch module
        if configs['from_modelscope_flag'] is True:
            self.table_structure_recognizer = pipeline(Tasks.table_recognition, model = configs['model_path'])  # table structure recognition model from modelscope
        else:
            self.table_structure_recognizer = None  # (20230815) currently we only support models from modelscope


    def __call__(self, image):
        """
        Description:
          recognize the structure of the table in the input image

        Parameters:
          image: the image to be processed, assume that it is a *full* image potentially containing text instances

        Return:
          result: table structure recognition result
        """

        # initialize
        result = None
        
        # perform table structure recognition
        if self.table_structure_recognizer is not None:
            # run the table structure recognizer
            tsr_result = self.table_structure_recognizer(image)    
   
            result = tsr_result['polygons']

        return result

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.table_structure_recognizer is not None:
            del self.table_structure_recognizer

        return 