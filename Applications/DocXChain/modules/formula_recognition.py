#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://github.com/RapidAI/RapidLatexOCR

import sys
import numpy as np

from rapid_latex_ocr import LatexOCR

class FormulaRecognition(object):
    """
    Description:
      class definition of FormulaRecognition module: 
      (1) algorithm interfaces for formula recognition (currently only mathematical formulas are supported - 20231122)

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch module
        if configs['from_modelscope_flag'] is True:
            self.formula_recognizer = None # (20231122) currently we only support models from https://github.com/RapidAI/RapidLatexOCR
        else:  # (20231122) currently we only support models from https://github.com/RapidAI/RapidLatexOCR
            image_resizer_path = configs['image_resizer_path']
            encoder_path = configs['encoder_path']
            decoder_path = configs['decoder_path']
            tokenizer_json = configs['tokenizer_json']
            self.formula_recognizer = LatexOCR(image_resizer_path = image_resizer_path, encoder_path = encoder_path, decoder_path = decoder_path, tokenizer_json = tokenizer_json)

    def __call__(self, image):
        """
        Description:
          recognize the formula contained in the given image

        Parameters:
          image: the image to be processed (assume that it is a sub image containing a formula)

        Return:
          result: recognition result
        """

        # initialize
        result = None
        
        # perform formula recognition
        if self.formula_recognizer is not None:
            # recognize the given formula
            result, elapse = self.formula_recognizer(image)

        return result

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.formula_recognizer is not None:
            del self.formula_recognizer

        return 