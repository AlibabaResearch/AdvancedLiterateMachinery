#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import cv2

import pdfplumber

def load_image(image_path):

    # initialization
    image = None

    # read image (only JPEG and PNG formats are supported currently) (20230815)
    name = image_path.lower()
    if name.endswith('.jpg') or name.endswith('.png'):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    return image

def load_pdf(pdf_path, page_index = 0):

    # initialization
    image = None

    # read PDF file
    name = pdf_path.lower()
    if name.endswith('.pdf'):
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            if page_index >= page_count - 1:
                page_index = page_count - 1

            page = pdf.pages[page_index]  # select the specified page (the first page will be chosen, by default)
            page_image = page.to_image(resolution=150) # convert the page to image by default (20230815)
            image = cv2.cvtColor(np.array(page_image.original), cv2.COLOR_RGB2BGR)

            pdf.close()

    return image

def load_whole_pdf(pdf_path):

    # initialization
    image_list = []

    # read PDF file (load all pages in the PDF file)
    name = pdf_path.lower()
    if name.endswith('.pdf'):
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page_index in range(page_count):  # traverse all pages
                page = pdf.pages[page_index]  # select the current page
                page_image = page.to_image(resolution=150) # convert the page to image by default (20230815)
                image = cv2.cvtColor(np.array(page_image.original), cv2.COLOR_RGB2BGR)

                image_list.append(image)

            pdf.close()

    return image_list

def load_document(document_path, whole_flag = False):

    # initialization
    image = None

    # load file
    name = document_path.lower()
    if name.endswith('.pdf'):
        if whole_flag is True:
            image = load_whole_pdf(document_path)
        else:
            image = load_pdf(document_path)
    else:
        image = load_image(document_path)

    return image