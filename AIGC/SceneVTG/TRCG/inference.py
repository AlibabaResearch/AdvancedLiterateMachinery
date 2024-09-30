import argparse
import torch
import math
import sys

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
import requests
from io import BytesIO
from transformers import TextStreamer
import random
import os
import numpy as np
import json
import copy
from tqdm.auto import tqdm
import re
import pickle
import codecs
import cv2
from scipy.interpolate import interp1d
from bezier_utils import cpts_to_bezier_cpts_cubic, cpts_to_bezier_cpts_cubic_edge

def generate_bezier_cubic(control_points, t):
    P = np.array(control_points).reshape(-1, 2)
    M = np.array(
        [
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ]
    )
    T = np.array([[t**3, t**2, t, 1]])
    B = np.matmul(np.matmul(T, M), P)
    return B[0,0], B[0,1]

def bezier_to_polygon(bz_points):
    ts = np.linspace(0,1,20).tolist()
    up_pts = [generate_bezier_cubic(bz_points[:8], t) for t in ts]
    down_pts = [generate_bezier_cubic(bz_points[8:], t) for t in ts]
    bound_pts = up_pts + down_pts
    bound_pts = [(int(pt[0]), int(pt[1])) for pt in bound_pts]
    return bound_pts

def metrics_iou(bb1, bb2):
    poly1 = bezier_to_polygon(bb1)
    poly2 = bezier_to_polygon(bb2)
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    poly1 = poly1.buffer(0.01)
    poly2 = poly2.buffer(0.01)
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return intersection_area / union_area

def overlay_indices(box):
    n = len(box)
    over_indices_list = []
    over_indices = []
    if n > 1:
        for i in range(n):
            bb1 = box[i]
            for j in range(i + 1, n):
                bb2 = box[j]
                ove = metrics_iou(bb1, bb2)
                if ove > 0.01:
                    over_indices.append((i, j))
    return over_indices


def draw_bezier_layout(img, bz_points):
    color = (0, 0, 255)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
    ts = np.linspace(0,1,20).tolist()
    up_pts = [generate_bezier_cubic(bz_points[:8], t) for t in ts]
    down_pts = [generate_bezier_cubic(bz_points[8:], t) for t in ts]
    bound_pts = up_pts + down_pts
    bound_pts = [(int(pt[0]), int(pt[1])) for pt in bound_pts]
    cv2.polylines(img, [np.array(bound_pts).reshape((-1,1,2))], True, color)
    
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    return image

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def bezier_to_polygon(bz_points):
    ts = np.linspace(0,1,20).tolist()
    up_pts = [generate_bezier_cubic(bz_points[:8], t) for t in ts]
    down_pts = [generate_bezier_cubic(bz_points[8:], t) for t in ts]
    bound_pts = up_pts + down_pts
    bound_pts = [(int(pt[0]), int(pt[1])) for pt in bound_pts]
    return bound_pts

def inference(args, image_file, questions, tokenizer, model, image_processor, context_len):
    # try:
    model = model.cuda()
    conv_mode = args.conv_mode
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in args.model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    try:
        image = load_image(image_file)
    except:
        return [], []
    image = image.resize((512, 512))
    w, h = image.size
    if args.visualize:
        pred_image = copy.deepcopy(image)
        pred_image_file = os.path.join(args.visual_folder, image_file.split("/")[-1])

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    
    bezier_list, text_list = [], []
    
    for index, qs in enumerate(questions):
        inp = f"{roles[0]}: " + qs
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        pd = outputs.replace('</s>', '')[:-1]
        
        if index == 1:
            try:
                pd_dicts = eval(pd)
            except:
                return [], []
            for pd_dict in pd_dicts:
                bezier_poly = pd_dict["layout"]
                beizer_box = [float(x) for x in bezier_poly]
                bezier_list.append(beizer_box)
                text_list.append(pd_dict["text"])

    # post process: skip overlay
    ove_indices = overlay_indices(bezier_list)
    skip_over_indices = []
    if len(ove_indices) > 0:
        for indices in ove_indices:
            if indices[0] not in skip_over_indices and indices[1] not in skip_over_indices:
                skip_over_indices.append(indices[random.choice([0, 1])])

    text_boxes, contents = [], []

    if not args.with_word:
        for i, (box, text) in enumerate(zip(bezier_list, text_list)):
            if i in skip_over_indices:
                continue
            if args.visualize:
                pred_image = draw_bezier_layout(pred_image, [int(x) for x in box])
            text_boxes.append([[int(x) for x in box], [int(x) for x in box]])
            contents.append([text, text])
    

    else:
        for box, text in zip(bezier_list, text_list):
            texts = text.split(" ")
            if len(texts) > 1:
                sub_box_list, sub_text_list = [], []
                sub_box_list.append(box)
                sub_text_list.append(text)
                ts = np.linspace(0,1,100).tolist()
                up_pts = [generate_bezier_cubic(box[:8], t) for t in ts]
                down_pts = [generate_bezier_cubic(box[8:], t) for t in ts]
                box = up_pts + down_pts
                box = [(int(pt[0]), int(pt[1])) for pt in box]
                
                up_x = np.array([x[0] for x in up_pts])
                up_y = np.array([x[1] for x in up_pts])

                curve_length = np.sum(np.sqrt(np.diff(up_x)**2 + np.diff(up_y)**2))

                interval_length = curve_length / len(text)
                split_length = []
                for te in texts:
                    split_length.append(interval_length*len(te))
                    split_length.append(interval_length)
                split_length = split_length[:-1]
                accumulated_length = 0
                up_points, up_point = [], [[up_x[0], up_y[0]]]
                count = 0
                for i in range(len(ts)-1):
                    dx = up_x[i+1] - up_x[i]
                    dy = up_y[i+1] - up_y[i]
                    segment_length = np.sqrt(dx**2 + dy**2)
                    accumulated_length += segment_length

                    up_point.append([up_x[i+1], up_y[i+1]])
                    if i == len(ts)-2:
                        continue
                    
                    if accumulated_length >= split_length[count]:
                        up_points.append(up_point)
                        up_point = []
                        accumulated_length = 0
                        count += 1
                up_points.append(up_point)
                down_x = np.array([x[0] for x in down_pts])
                down_y = np.array([x[1] for x in down_pts])

                curve_length = np.sum(np.sqrt(np.diff(down_x)**2 + np.diff(down_y)**2))

                interval_length = curve_length / len(text)
                split_length = []
                for te in texts:
                    split_length.append(interval_length*len(te))
                    split_length.append(interval_length)
                split_length = split_length[:-1]
                accumulated_length = 0
                down_x = down_x[::-1]
                down_y = down_y[::-1]
                down_points, down_point = [], [[down_x[0], down_y[0]]]
                count = 0
                for i in range(len(ts)-1):
                    dx = down_x[i+1] - down_x[i]
                    dy = down_y[i+1] - down_y[i]
                    segment_length = np.sqrt(dx**2 + dy**2)
                    accumulated_length += segment_length

                    down_point.append([down_x[i+1], down_y[i+1]])
                    if i == len(ts)-2:
                        continue
                    
                    if accumulated_length >= split_length[count]:
                        down_points.append(down_point)
                        down_point = []
                        accumulated_length = 0
                        count += 1
                down_points.append(down_point)
                
                for i, te in enumerate(texts):
                    up_point = up_points[2*i]
                    down_point = down_points[2*i]
                    top_bezier = cpts_to_bezier_cpts_cubic_edge(np.array(up_point))
                    down_bezier = cpts_to_bezier_cpts_cubic_edge(np.array(down_point))
                    bezier = np.concatenate((top_bezier, down_bezier[::-1]), axis=0).reshape(-1)
                    box = [int(x) for x in bezier]
                    sub_box_list.append(box)
                    sub_text_list.append(te)
                    if args.visualize:
                        pred_image = draw_bezier_layout(pred_image, box)
                text_boxes.append(sub_box_list)
                contents.append(sub_text_list)
                    
            else:
                text_boxes.append([[int(x) for x in box], [int(x) for x in box]])
                contents.append([text, text])
                if args.visualize:
                    pred_image = draw_bezier_layout(pred_image, [int(x) for x in box])
    import pdb;pdb.set_trace()
    if args.visualize:
        pred_image.save(pred_image_file)
    
    return text_boxes, contents
    # except:
    #     return [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./ckpts/trcg/")
    parser.add_argument("--model-base", type=str, default="./llava-v1.5-7b")
    parser.add_argument("--test-file", type=str, default="./TRCG_data/scripts/test_file_list.txt")
    parser.add_argument("--model-name", type=str, default="llava-llama-2-lora-vision")
    parser.add_argument("--conv-mode", type=str, default="v2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out-file", type=str, default="results.txt")
    parser.add_argument("--seed", type=int, default=4095)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visual-folder", type=str, default="vis/")
    parser.add_argument("--post-process", type=bool, default=True)
    parser.add_argument("--with_word", action="store_true")

    args = parser.parse_args()

    seed_torch(args.seed)
    disable_torch_init()
    file_lines = open(args.test_file, 'r').readlines()[:5]

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_name)
    pd_dicts = {}
    wtr = open(args.out_file, 'w')
    if args.visualize and not os.path.exists(args.visual_folder):
        os.makedirs(args.visual_folder)
    for line in tqdm(file_lines):
        image_file = line.strip().replace("../", "/mnt/workspace/workgroup/jiayu/dataset/TRCG_data/")
        questions = []
        # step1
        questions.append(f"Given a background image that will be written with text, plan the text and location of the visual text for the image. \
The planned locations are represented by the coordinates of the points, and they should be located in suitable areas of the image for writing text. \
The size of the image is 512*512. Therefore, none of the properties of the positions should exceed 512. \
The planned texts should be related to each other and fit to appear in the location represented by the corresponding point. \
The location and text of the planned visual text needs to be represented in JSON format.")
        # step2
        questions.append(f"Based on the point and text planned above, plan the layout of the visual text for the image. \
Point guide where the layout should be, and the planned layout should be located near the point. \
Layouts are represented in the form of Bezier curve control point coordinates, represents an area on an image suitable for writing visual text. \
Each box consists of eight vertices, starting at the top left corner in counterclockwise order. \
The appropriate layouts should not overlap each other. \
The aspect ratio of the layout boxes need to consider the number of characters in the texts, the more characters, the larger the aspect ratio. \
The layout and text of the planned visual text needs to be represented in JSON format too.")
        text_boxes, contents = inference(args, image_file, questions, tokenizer, model, image_processor, context_len)
        if len(text_boxes) > 0:
            wtr.write('%s\t%s\t%s\n' % (image_file, text_boxes, contents))