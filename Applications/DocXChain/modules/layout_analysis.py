#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import json

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
            self.layout_analyser = None  # (20230912) currently we only support layout analysis model from Advanced Literate Machinery (https://github.com/AlibabaResearch/AdvancedLiterateMachinery)
        else:
            params = {
                'model_file': configs['model_path'],
                'debug': 0, # 1: save vis results, 0: don't save
            }

            # load map information
            map_info = json.load(open(BASE_DIR + '/../../../DocumentUnderstanding/DocXLayout/map_info.json'))
            category_map = {}
            for cate, idx in map_info["huntie"]["primary_map"].items():
                category_map[idx] = cate

            self.category_map = category_map

            # initialize
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

    def mapping(self, index):
        """
        Description:
          return the category name all the index
        """

        category = self.category_map[index]

        return category

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.layout_analyser is not None:
            del self.layout_analyser

        return