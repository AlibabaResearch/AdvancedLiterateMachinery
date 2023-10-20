#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

import numpy as np
import cv2
import datetime
import time
import pytz

from modules.file_loading import load_document
from pipelines.general_text_reading import GeneralTextReading
from pipelines.table_parsing import TableParsing
from pipelines.document_structurization import DocumentStructurization
from utilities.visualization import *

def general_text_reading_example(image):

    # configure
    configs = dict()
    
    text_detection_configs = dict()
    text_detection_configs['from_modelscope_flag'] = True
    text_detection_configs['model_path'] = 'damo/cv_resnet18_ocr-detection-line-level_damo'
    configs['text_detection_configs'] = text_detection_configs

    text_recognition_configs = dict()
    text_recognition_configs['from_modelscope_flag'] = True
    text_recognition_configs['model_path'] = 'damo/cv_convnextTiny_ocr-recognition-general_damo'  # alternatives: 'damo/cv_convnextTiny_ocr-recognition-scene_damo', 'damo/cv_convnextTiny_ocr-recognition-document_damo', 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo' 
    configs['text_recognition_configs'] = text_recognition_configs

    # initialize
    text_reader = GeneralTextReading(configs)

    # run
    final_result = text_reader(image)

    if True:
        print (final_result)

    # visualize
    output_image = general_text_reading_visualization(final_result, image)

    # release
    text_reader.release()

    return final_result, output_image

def table_parsing_example(image):

    # configure
    configs = dict()

    table_structure_recognition_configs = dict()
    table_structure_recognition_configs['from_modelscope_flag'] = True
    table_structure_recognition_configs['model_path'] = 'damo/cv_dla34_table-structure-recognition_cycle-centernet'
    configs['table_structure_recognition_configs'] = table_structure_recognition_configs
    
    text_detection_configs = dict()
    text_detection_configs['from_modelscope_flag'] = True
    text_detection_configs['model_path'] = 'damo/cv_resnet18_ocr-detection-line-level_damo'
    configs['text_detection_configs'] = text_detection_configs

    text_recognition_configs = dict()
    text_recognition_configs['from_modelscope_flag'] = True
    text_recognition_configs['model_path'] = 'damo/cv_convnextTiny_ocr-recognition-general_damo'  # alternatives: 'damo/cv_convnextTiny_ocr-recognition-scene_damo', 'damo/cv_convnextTiny_ocr-recognition-document_damo', 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo' 
    configs['text_recognition_configs'] = text_recognition_configs

    # initialize
    table_parser = TableParsing(configs)

    # run
    final_result = table_parser(image)

    if True:
        print (final_result)

    # visualize
    output_image = table_parsing_visualization(final_result, image)

    # release
    table_parser.release()

    return final_result, output_image


def document_structurization_example(image):

    # configure
    configs = dict()
    
    layout_analysis_configs = dict()
    layout_analysis_configs['from_modelscope_flag'] = False
    layout_analysis_configs['model_path'] = '/home/DocXLayout_231012.pth'  # note that: currently the layout analysis model is NOT from modelscope
    configs['layout_analysis_configs'] = layout_analysis_configs
    
    text_detection_configs = dict()
    text_detection_configs['from_modelscope_flag'] = True
    text_detection_configs['model_path'] = 'damo/cv_resnet18_ocr-detection-line-level_damo'
    configs['text_detection_configs'] = text_detection_configs

    text_recognition_configs = dict()
    text_recognition_configs['from_modelscope_flag'] = True
    text_recognition_configs['model_path'] = 'damo/cv_convnextTiny_ocr-recognition-document_damo'  # alternatives: 'damo/cv_convnextTiny_ocr-recognition-scene_damo', 'damo/cv_convnextTiny_ocr-recognition-general_damo', 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo' 
    configs['text_recognition_configs'] = text_recognition_configs

    # initialize
    document_structurizer = DocumentStructurization(configs)

    # run
    final_result = document_structurizer(image)

    if True:
        print (final_result)

    # visualize
    output_image = document_structurization_visualization(final_result, image)

    # release
    document_structurizer.release()

    return final_result, output_image

# main routine
def main():
    """
    Description:
      a demo to showcase the pipelines
    """

    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices = ['general_text_reading', 'table_parsing', 'document_structurization'], help = "specify the task to be performed", type = str)
    parser.add_argument("document_path", help = "specify the path of the document (supported formats: JPG, PNG, and PDF) to be processed", type = str)
    parser.add_argument("output_path", help = "specify the path of the image with visulization", type = str)
    args = parser.parse_args()

    # start
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    print ("Started!")

    # load document
    image = load_document(args.document_path)
    
    # process
    output_image = None
    if image is not None:
        if args.task == 'general_text_reading':
            final_result, output_image = general_text_reading_example(image)
        elif args.task == 'table_parsing':
            final_result, output_image = table_parsing_example(image)
        else: # args.task == 'document_structurization'
            final_result, output_image = document_structurization_example(image)
    else:
        print ("Failed to load the document file!")

    # output
    if output_image is not None:
        cv2.imwrite(args.output_path, output_image)

    # finish
    now = datetime.datetime.now(tz)
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    print ("Finished!")

    return

if __name__ == "__main__":
    # execute only if run as a script
    main()

