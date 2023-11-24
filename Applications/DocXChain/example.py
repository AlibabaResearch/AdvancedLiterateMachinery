#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

import numpy as np
import cv2
import datetime
import time
import pytz
import json

from modules.file_loading import load_document
from modules.formula_recognition import FormulaRecognition
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

def formula_recognition_example(image):

    # configure
    configs = dict()

    formula_recognition_configs = dict()
    formula_recognition_configs['from_modelscope_flag'] = False
    formula_recognition_configs['image_resizer_path'] = '/home/LaTeX-OCR_image_resizer.onnx'
    formula_recognition_configs['encoder_path'] = '/home/LaTeX-OCR_encoder.onnx'
    formula_recognition_configs['decoder_path'] = '/home/LaTeX-OCR_decoder.onnx'
    formula_recognition_configs['tokenizer_json'] = '/home/LaTeX-OCR_tokenizer.json'
    configs['formula_recognition_configs'] = formula_recognition_configs

    # initialize
    formula_recognizer = FormulaRecognition(configs['formula_recognition_configs'])

    # run
    result = formula_recognizer(image)
    formula_latex = '$$ ' + result + ' $$'
    final_result = {'formula_latex': formula_latex}

    if True:
        print (final_result)

    # release
    formula_recognizer.release()

    return final_result

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

    formula_recognition_configs = dict()
    formula_recognition_configs['from_modelscope_flag'] = False
    formula_recognition_configs['image_resizer_path'] = '/home/LaTeX-OCR_image_resizer.onnx'
    formula_recognition_configs['encoder_path'] = '/home/LaTeX-OCR_encoder.onnx'
    formula_recognition_configs['decoder_path'] = '/home/LaTeX-OCR_decoder.onnx'
    formula_recognition_configs['tokenizer_json'] = '/home/LaTeX-OCR_tokenizer.json'
    configs['formula_recognition_configs'] = formula_recognition_configs

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

def whole_pdf_conversion_example(image_list):

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

    formula_recognition_configs = dict()
    formula_recognition_configs['from_modelscope_flag'] = False
    formula_recognition_configs['image_resizer_path'] = '/home/LaTeX-OCR_image_resizer.onnx'
    formula_recognition_configs['encoder_path'] = '/home/LaTeX-OCR_encoder.onnx'
    formula_recognition_configs['decoder_path'] = '/home/LaTeX-OCR_decoder.onnx'
    formula_recognition_configs['tokenizer_json'] = '/home/LaTeX-OCR_tokenizer.json'
    configs['formula_recognition_configs'] = formula_recognition_configs

    # initialize
    document_structurizer = DocumentStructurization(configs)

    # run
    final_result = []
    page_index = 0
    for image in image_list:
        result = document_structurizer(image)

        page_info = {'page': page_index, 'information': result}
        final_result.append(page_info)

        page_index = page_index + 1

    if True:
        print (final_result)

    # release
    document_structurizer.release()

    return final_result

# main routine
def main():
    """
    Description:
      a demo to showcase the pipelines
    """

    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices = ['general_text_reading', 'table_parsing', 'formula_recognition', 'document_structurization', 'whole_pdf_conversion'], help = "specify the task to be performed", type = str)
    parser.add_argument("document_path", help = "specify the path of the document (supported formats: JPG, PNG, and PDF) to be processed", type = str)
    parser.add_argument("output_path", help = "specify the path of the image with visulization or the json file for storage", type = str)
    args = parser.parse_args()

    # start
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    print ("Started!")

    # load document
    image = None
    image_list = None
    if args.task == 'whole_pdf_conversion':
        name = args.document_path.lower()
        if name.endswith('.pdf'):
            image_list = load_document(args.document_path, whole_flag = True)
        else:
            print ('For the whole PDF conversion task, only PDF files are supported!')
    else:
        image = load_document(args.document_path)
    
    # process
    final_result = None
    output_image = None
    if image is not None or image_list is not None:
        if args.task == 'general_text_reading':
            final_result, output_image = general_text_reading_example(image)
        elif args.task == 'table_parsing':
            final_result, output_image = table_parsing_example(image)
        elif args.task == 'formula_recognition':
            final_result = formula_recognition_example(image)
        elif args.task == 'document_structurization':
            final_result, output_image = document_structurization_example(image)
        else:  # args.task == 'whole_pdf_conversion'
            final_result = whole_pdf_conversion_example(image_list)
    else:
        print ("Failed to load the document file!")

    # dump
    name = args.output_path.lower()
    if name.endswith('.png'):
        if output_image is not None:
            cv2.imwrite(args.output_path, output_image)
    elif name.endswith('.json'):
        if final_result is not None:
            with open(args.output_path, 'w') as json_file:
                json.dump(final_result, json_file, indent = 4) 
    else:
        pass

    # finish
    now = datetime.datetime.now(tz)
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    print ("Finished!")

    return

if __name__ == "__main__":
    # execute only if run as a script
    main()

