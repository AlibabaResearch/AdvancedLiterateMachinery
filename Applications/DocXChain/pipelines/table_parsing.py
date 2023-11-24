#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://www.modelscope.cn/studios/damo/cv_table-ocr/file/view/master/app.py

import sys
import numpy as np

from modules.text_detection import TextDetection
from modules.text_recognition import TextRecognition
from modules.table_structure_recognition import TableStructureRecognition

class TableParsing(object):
    """
    Description:
      class definition of TableParsing pipeline: 
      (1) algorithm interfaces for table parsing (table structure recognition + content recognition)
      (2) only tables with *visible borders* are supported currently (20230815)

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch pipiline
        self.table_structure_recognition_module = TableStructureRecognition(configs['table_structure_recognition_configs'])
        self.text_detection_module = TextDetection(configs['text_detection_configs'])
        self.text_recognition_module = TextRecognition(configs['text_recognition_configs'])

    def __call__(self, image):
        """
        Description:
          perform table parsing (table structure recognition + content recognition)

        Parameters:
          image: the image to be processed, assume that it is a *full* image potentially containing a table

        Return:
          final_result: table parsing result
        """

        # initialize
        tsr_result = None
        det_result = None
        final_result = None

        # perform table structure recognition, text detection and recognition successively, then combine the results to make the final output
        if image is not None:
            tsr_result = self.table_structure_recognition_module(image)
            det_result = self.text_detection_module(image)

            final_result = self._recognize_and_assemble(image, tsr_result, det_result)    

        return final_result

    def _recognize_and_assemble(self, image, tsr_result, det_result):
        """
        Description:
          perform content recognition and assembling

        Parameters:
          image: the image to be processed, assume that it is a *full* image
          tsr_result: table structure recognition result
          det_result: text detection result

        Return:
          output: table parsing result
        """

        # initialize
        output = []
        tsr_result = np.array(tsr_result).reshape([len(tsr_result), 4, 2])

        # perform recognition and assembling
        for i in range(det_result.shape[0]):
            # crop sub image and recognize the text content within it
            pts = self.text_recognition_module.order_point(det_result[i])
            cropped_image = self.text_recognition_module.crop_image(image, pts)
            rec = self.text_recognition_module.recognize_cropped_image(cropped_image)

            find_cell = 0
            p0, p1, p2, p3 = pts
            ctx = (p0[0]+p1[0]+p2[0]+p3[0]) / 4.0
            cty = (p0[1]+p1[1]+p2[1]+p3[1]) / 4.0
            for j in range(0, len(tsr_result)):
                if self._point_in_box(tsr_result[j], [ctx, cty]):
                    #output.append([str(i + 1), rec['text'], self._coord2str(pts), self._coord2str(tsr_result[j])])
                    cell_poly = np.array([round(tsr_result[j][0][0]), round(tsr_result[j][0][1]),\
                                          round(tsr_result[j][1][0]), round(tsr_result[j][1][1]),\
                                          round(tsr_result[j][2][0]), round(tsr_result[j][2][1]),\
                                          round(tsr_result[j][3][0]), round(tsr_result[j][3][1])])

                    item = {}
                    item['position'] = det_result[i].tolist()
                    item['content'] = rec['text']
                    item['cell'] = cell_poly.tolist()
                    output.append(item)

                    find_cell = 1
                    break
            
            if find_cell == 0:
                dummy_cell_poly = np.array([-1, -1, -1, -1, -1, -1, -1, -1])

                item = {}
                item['position'] = det_result[i].tolist()
                item['content'] = rec['text']
                item['cell'] = dummy_cell_poly.tolist()
                output.append(item)

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

        if self.table_structure_recognition_module is not None:
            self.table_structure_recognition_module.release()
        
        if self.text_detection_module is not None:
            self.text_detection_module.release()
        
        if self.text_recognition_module is not None:
            self.text_recognition_module.release()

        return 

