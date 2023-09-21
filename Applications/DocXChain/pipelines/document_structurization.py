#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://www.modelscope.cn/studios/damo/cv_table-ocr/file/view/master/app.py

import sys
import numpy as np

from modules.text_detection import TextDetection
from modules.text_recognition import TextRecognition
from modules.table_structure_recognition import TableStructureRecognition
from modules.layout_analysis import LayoutAnalysis

class DocumentStructurization(object):
    """
    Description:
      class definition of DocumentStructurization pipeline: 
      (1) algorithm interfaces for document structurization (layout analysis + table structure recognition (if any table) + text content recognition)
      (2) only tables with *visible borders* are supported currently (20230815)

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch pipiline
        self.layout_analysis_module = LayoutAnalysis(configs['layout_analysis_configs'])
        self.text_detection_module = TextDetection(configs['text_detection_configs'])
        self.text_recognition_module = TextRecognition(configs['text_recognition_configs'])

    def __call__(self, image):
        """
        Description:
          perform table parsing (table structure recognition + content recognition)

        Parameters:
          image: the image to be processed, assume that it is a *full* image potentially containing a table

        Return:
          final_result: final document structurization result
        """

        # initialize
        final_result = None

        # perform layout analysis, text detection and recognition successively, then combine the results to make the final output
        if image is not None:
            la_result = self.layout_analysis_module(image)
            det_result = self.text_detection_module(image)
            rec_result = self.text_recognition_module(image, det_result)

            #print (la_result)
            final_result = self._assemble(la_result, det_result, rec_result)

        return final_result

    def _assemble(self, la_result, det_result, rec_result):
        """
        Description:
          perform assembling

        Parameters:
          image: the image to be processed, assume that it is a *full* image
          la_result: layout analysis result
          det_result: text detection result
          rec_result: text recognition result

        Return:
          output: final result
        """

        # initialize
        output = []

        # assemble all the intermediate results to make the final output
        for i in range(det_result.shape[0]):
            # fetch
            pts = self.text_recognition_module.order_point(det_result[i])
            rec = rec_result[i]

            find_cell = 0
            p0, p1, p2, p3 = pts
            ctx = (p0[0]+p1[0]+p2[0]+p3[0]) / 4.0
            cty = (p0[1]+p1[1]+p2[1]+p3[1]) / 4.0
            layout_dets = la_result['layout_dets']
            for j in range(0, len(layout_dets)):
                layout_box = [(layout_dets[j]['poly'][0], layout_dets[j]['poly'][1]),\
                              (layout_dets[j]['poly'][2], layout_dets[j]['poly'][3]),\
                              (layout_dets[j]['poly'][4], layout_dets[j]['poly'][5]),\
                              (layout_dets[j]['poly'][6], layout_dets[j]['poly'][7])]
                if self._point_in_box(layout_box, [ctx, cty]):
                    output.append([str(i + 1), layout_dets[j]['category_id'], rec['text'], self._coord2str(pts), self._coord2str(layout_box)])
                    find_cell = 1
                    break
            
            if find_cell == 0:
                output.append([str(i + 1), -1, rec['text'], self._coord2str(pts), ''])   
        
        return output

    def _point_in_box(self, box, point):
        x1,y1 = box[0][0],box[0][1]
        x2,y2 = box[1][0],box[1][1]
        x3,y3 = box[2][0],box[2][1]
        x4,y4 = box[3][0],box[3][1]
        ctx,cty = point[0],point[1]
        a = (x2 - x1)*(cty - y1) - (y2 - y1)*(ctx - x1) 
        b = (x3 - x2)*(cty - y2) - (y3 - y2)*(ctx - x2) 
        c = (x4 - x3)*(cty - y3) - (y4 - y3)*(ctx - x3) 
        d = (x1 - x4)*(cty - y4) - (y1 - y4)*(ctx - x4) 
        if ((a > 0  and  b > 0  and  c > 0  and  d > 0) or (a < 0  and  b < 0  and  c < 0  and  d < 0)):
            return True
        else :
            return False

    def _coord2str(self, box):
        out = []
        for i in range(0,4):
            out.append(str(round(box[i][0],1))+','+str(round(box[i][1],1)))
    
        return ';'.join(out)    

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.layout_analysis_module is not None:
            self.layout_analysis_module.release()
        
        if self.text_detection_module is not None:
            self.text_detection_module.release()
        
        if self.text_recognition_module is not None:
            self.text_recognition_module.release()

        return 

