#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://www.modelscope.cn/studios/damo/cv_table-ocr/file/view/master/app.py

import sys
import numpy as np

from modules.layout_analysis import LayoutAnalysis
from modules.text_detection import TextDetection
from modules.text_recognition import TextRecognition
from modules.table_structure_recognition import TableStructureRecognition
from modules.formula_recognition import FormulaRecognition

class DocumentStructurization(object):
    """
    Description:
      class definition of DocumentStructurization pipeline: 
      (1) algorithm interfaces for document structurization (layout analysis + text detection + text recognition)
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
        self.formula_recognition_module = FormulaRecognition(configs['formula_recognition_configs'])

    def __call__(self, image):
        """
        Description:
          structurize the given document (layout analysis + content recognition)

        Parameters:
          image: the image to be processed; assume that it is a *full* image

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
            final_result = self._assemble(image, la_result, det_result, rec_result)

        return final_result

    def _assemble(self, image, la_result, det_result, rec_result):
        """
        Description:
          perform assembling (combine the results )

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
        layout_dets = la_result['layout_dets']
        for i in range(0, len(layout_dets)):
            # fetch each layout region
            category_index = layout_dets[i]['category_id']
            category_name = self.layout_analysis_module.mapping(category_index)
            layout_box = [(layout_dets[i]['poly'][0], layout_dets[i]['poly'][1]),\
                          (layout_dets[i]['poly'][2], layout_dets[i]['poly'][3]),\
                          (layout_dets[i]['poly'][4], layout_dets[i]['poly'][5]),\
                          (layout_dets[i]['poly'][6], layout_dets[i]['poly'][7])]

            region_poly = np.array([round(layout_box[0][0]), round(layout_box[0][1]),\
                            round(layout_box[1][0]), round(layout_box[1][1]),\
                            round(layout_box[2][0]), round(layout_box[2][1]),\
                            round(layout_box[3][0]), round(layout_box[3][1])])

            layout_region = {}
            layout_region['category_index'] = category_index
            layout_region['category_name'] = category_name
            layout_region['region_poly'] = region_poly.tolist()
            layout_region['text_list'] = []  # one region may contain multiple text instances

            if layout_region['category_name'] == 'equation':  # special is treatment needed for equations/formulas
                # crop sub image and perform formula recognition
                pts = self.text_recognition_module.order_point(region_poly)
                image_crop = self.text_recognition_module.crop_image(image, pts)
                fr_result = self.formula_recognition_module(image_crop)

                #print ('formua recognition: ', fr_result)

                # record
                item = {}
                item['position'] = region_poly.tolist()
                item['content'] = '$$ ' + fr_result + ' $$'
                layout_region['text_list'].append(item)

            else:
                # match and assign
                for j in range(det_result.shape[0]):
                    # fetch each text instance
                    pts = self.text_recognition_module.order_point(det_result[j])
                    rec = rec_result[j]

                    # check
                    p0, p1, p2, p3 = pts
                    ctx = (p0[0]+p1[0]+p2[0]+p3[0]) / 4.0
                    cty = (p0[1]+p1[1]+p2[1]+p3[1]) / 4.0
                    if self._point_in_box(layout_box, [ctx, cty]):
                        # record if matched
                        item = {}
                        item['position'] = det_result[j].tolist()
                        item['content'] = rec['text']
                        layout_region['text_list'].append(item)

                    else:
                        pass  # (20231010) currently text instances that have not been assigned to any layout region will be discarded 
        
            # record
            output.append(layout_region)

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

