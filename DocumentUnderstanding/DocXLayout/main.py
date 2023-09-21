import os
import cv2
import random

from opts import opts
from huntie_subfield import Huntie_Subfield
from detectors.detector_factory import detector_factory
import ipdb
import numpy as np
import logging

import os.path as osp
import sys
import time
import json

from wrapper import wrap_result

logger = logging.getLogger(__name__)
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

class DocXLayoutInfo:
    def __init__(self, result):
        if result['code'] != 200:
            return
        layout_detection_info = result["layout_dets"]
        subfield_detection_info = result["subfield_dets"]
        

class DocXLayoutPredictor:
    def __init__(self, params):
        model_file = params["model_file"]
        debug = params["debug"]
        
        new_params = {
            'task': 'ctdet_subfield',
            'arch': 'dlav0subfield_34',
            'input_res': 768,
            'num_classes': 13,
            'load_model': model_file,
            'debug': debug, 
        }

        opt = opts().parse(new_params)
        opt = opts().update_dataset_info_and_set_heads(opt, Huntie_Subfield)

        Detector = detector_factory[opt.task]
        detector = Detector(opt)
        self.detector = detector
        self.opt = opt
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.info("Training parameters %s", new_params)
        
    def convert_eval_format(self, all_bboxes, opt):
        layout_detection_items = []
        subfield_detection_items = []
        for cls_ind in all_bboxes:
            for box in all_bboxes[cls_ind]:
                if box[8] < opt.scores_thresh:
                    continue
                pts = np.round(box).tolist()[:8]
                score = box[8]
                category_id = box[9]
                direction_id = box[10]
                secondary_id = box[11]
                detection = {
                    "category_id": int(category_id),
                    # "secondary_id": int(secondary_id),
                    # "direction_id": int(direction_id),
                    "poly": pts,
                    "score": float("{:.2f}".format(score))
                }
                if cls_ind in (12,13):
                    subfield_detection_items.append(detection)
                else:
                    layout_detection_items.append(detection)
        return layout_detection_items, subfield_detection_items
        
    def predict(self, content):
        try:
            logger.info("Running DocXlayout")
            content_json = json.loads(content)
            image_name = content_json['image_name']
            ret = self.detector.run(image_name)
            logger.info("DocXlayout Detector Done")
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            layout_detection_info, subfield_detection_info = self.convert_eval_format(ret['results'], self.opt)
            result = {"code":200,"layout_dets":layout_detection_info,"subfield_dets":subfield_detection_info,"time":time_str,"errMsg":"success"}
            logger.info("DocXlayout Finish")
            return result
        except Exception as e:
            logger.info("DocXlayout Error %s", repr(e))
            return {"code":404,"data":None,"errMsg": repr(e)}

    def __call__(self, image):
        ret = self.detector.run(image)
        layout_detection_info, subfield_detection_info = self.convert_eval_format(ret['results'], self.opt)
        result = {"layout_dets": layout_detection_info, "subfield_dets":subfield_detection_info}
        
        return result
        
        
if __name__ == '__main__':
    params = {
        'model_file': 'model_path/DocX_dla34_230829.pth',
        'debug': 1, # 1 save vis results, 0 don't save
    }
    
    predictor = DocXLayoutPredictor(params)
    # inputs = """{"image_name": "demo_images/layout_DocX_230816_first_phase_1395_10.png"}"""
    # inputs = """{"image_name": "demo_images/layout_DocX_230816_first_phase_1633_4.png"}"""
    inputs = """{"image_name": "demo_images/layout_DocX_230816_paper_chn_art_10821_2.png"}"""
    
    start = time.time()
    result = predictor.predict(inputs)
    end = time.time()
    # print(result)
    print("used time {}".format(end-start))

    map_info = json.load(open('map_info.json'))
    category_map = {}
    for cate, idx in map_info["huntie"]["primary_map"].items():
        category_map[idx] = cate
    DocXLayoutInfo = wrap_result(result, category_map)
    
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(DocXLayoutInfo, f)
    
    # print(json.dumps(DocXLayoutInfo, indent=4, ensure_ascii=False))