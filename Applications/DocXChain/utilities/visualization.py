#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import cv2

def general_text_reading_visualization(predictions, image, color = (49, 125, 237), thickness = 6):

    # draw quadrangles
    output_image = image.copy()
    for item in predictions:
        quadrangle = item['position']
        pts = np.array([[quadrangle[0], quadrangle[1]], [quadrangle[2], quadrangle[3]],
                        [quadrangle[4], quadrangle[5]], [quadrangle[6], quadrangle[7]]],
                        np.int32)
    
        pts = pts.reshape((-1, 1, 2))

        # draw poly
        isClosed = True
        output_image = cv2.polylines(output_image, [pts],  isClosed, color, thickness)

    return output_image

def table_parsing_visualization(predictions, image, color = (49, 125, 237), thickness = 6):

    # draw quadrangles
    output_image = image.copy()
    for item in predictions:
        # text detection quadrangle
        quadrangle = item['position']
        pts = np.array([[quadrangle[0], quadrangle[1]], [quadrangle[2], quadrangle[3]],
                        [quadrangle[4], quadrangle[5]], [quadrangle[6], quadrangle[7]]],
                        np.int32)
    
        pts = pts.reshape((-1, 1, 2))

        # draw poly
        isClosed = True
        color = (49, 125, 237)
        thickness = 2
        output_image = cv2.polylines(output_image, [pts],  isClosed, color, thickness)

        # table cell polygon 
        quadrangle = item['cell']
    
        if quadrangle[0] < 0 or quadrangle[1] < 0:  # if encounter a dummy cell poly, skip it
            continue

        pts = np.array([[quadrangle[0], quadrangle[1]], [quadrangle[2], quadrangle[3]],
                        [quadrangle[4], quadrangle[5]], [quadrangle[6], quadrangle[7]]],
                        np.int32)
    
        pts = pts.reshape((-1, 1, 2))
 
        # draw poly
        isClosed = True
        color = (0, 225, 0)
        thickness = 4
        output_image = cv2.polylines(output_image, [pts],  isClosed, color, thickness)

    return output_image

def document_structurization_visualization(predictions, image):

    # define color palette
    color_palette = [(0,0,255),\
                     (0,255,0),\
                     (255,64,255),\
                     (255,0,0),\
                     (255,255,0),\
                     (0,255,255),\
                     (255,215,135),\
                     (215,0,95),\
                     (100,0,48),\
                     (0,175,0),\
                     (95,0,95),\
                     (175,95,0),\
                     (95,95,0),\
                     (95,95,255),\
                     (95,175,135),\
                     (215,95,0),\
                     (0,95,215),\
                     (0,0,0),\
                     (0,0,0),\
                     (0,0,0)
                    ]

    # draw quadrangles
    output_image = image.copy()
    for item in predictions:
        # draw text quadrangle
        if True:
            for instance in item['text_list']:
                quadrangle = instance['position']
                pts = np.array([[quadrangle[0], quadrangle[1]], [quadrangle[2], quadrangle[3]],
                                [quadrangle[4], quadrangle[5]], [quadrangle[6], quadrangle[7]]],
                                np.int32)
            
                pts = pts.reshape((-1, 1, 2))

                # draw poly
                isClosed = True
                color = (49, 125, 237)
                thickness = 2
                output_image = cv2.polylines(output_image, [pts],  isClosed, color, thickness)
       

       # layout region quadrangle
        quadrangle = item['region_poly']
        pts = np.array([[quadrangle[0], quadrangle[1]], [quadrangle[2], quadrangle[3]],
                        [quadrangle[4], quadrangle[5]], [quadrangle[6], quadrangle[7]]],
                        np.int32)
    
        pts = pts.reshape((-1, 1, 2))

        # draw layout poly
        isClosed = True
        index = item['category_index']
        color = color_palette[index]
        thickness = 6
        output_image = cv2.polylines(output_image, [pts],  isClosed, color, thickness)

        # draw layout region category name
        name = item['category_name']
        fontScale = 1
        color = color_palette[index]
        thickness = 2
        output_image = cv2.putText(output_image, name, (quadrangle[0] - 5, quadrangle[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale,  color, thickness)


    return output_image