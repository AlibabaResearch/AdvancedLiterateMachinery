import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pandas import read_parquet
import pandas as pd
import json
import io
from utils.misc import bezier2bbox, bezier2polygon, sample_bezier_curve
from utils.misc import bezier_fit_quad, insert_mid_points, gen_bezier_ctrl_points
import torch
import os
import random

class OpenImageV5Dataset(Dataset):

    def __init__(self, dataset_name, transforms, args):
        
        self._transforms = transforms

        self.dataset_name = dataset_name
        if 'train' in dataset_name:
            self.split = 'train'
        else:
            self.split = 'val'

        ann_files = [
            './data/text_spotting_datasets/open_image_v5/anns/text_spotting_openimages_v5_train_1.json',
            './data/text_spotting_datasets/open_image_v5/anns/text_spotting_openimages_v5_train_2.json',
            './data/text_spotting_datasets/open_image_v5/anns/text_spotting_openimages_v5_train_5.json',
            './data/text_spotting_datasets/open_image_v5/anns/text_spotting_openimages_v5_train_f.json',
            './data/text_spotting_datasets/open_image_v5/anns/text_spotting_openimages_v5_validation.json',

        ]
        self.data = {}

        for ann_file in ann_files:
            with open(ann_file, 'r') as f:
                anns = json.load(f)
            img_names = anns['images']
            tmp_dict = {}
            for i in range(len(img_names)):
                img_id = img_names[i]['id']
                if img_id not in tmp_dict:
                    tmp_dict[img_id] = {'img_name': img_names[i]['file_name'], 'anns': []}
                else:
                    print('bad img_id')
            for i in range(len(anns['annotations'])):
                img_id = anns['annotations'][i]['image_id']
                if img_id in tmp_dict:
                    tmp_dict[img_id]['anns'].append(anns['annotations'][i])
                else:
                    print('bad ann image id')
            for k,v in tmp_dict.items():
                if v['img_name'] not in self.data:
                    self.data[v['img_name']] = v['anns']
                else:
                    print('bad tmp dict')
        
        self.img_names = list(self.data.keys())
        self.dataset_length = len(self.data)

        self.chars_dict = {}
        for i in range(len(args.chars)):
            self.chars_dict[args.chars[i]] = i
        
        self.root_path = './data/text_spotting_datasets/open_image_v5/data/'
        self.rec_length = args.rec_length
        self.rec_pad_index = len(args.chars) + 1
        self.num_bins = args.num_bins

    def sample_pts(self, bezier_pts):
        center_pts = []
        for bezier_pt in bezier_pts:
            bezier_pt = bezier_pt.numpy().reshape(8, 2)
            mid_pt1 = sample_bezier_curve(bezier_pt[:4], mid_point=True)
            mid_pt2 = sample_bezier_curve(bezier_pt[4:], mid_point=True)
            center_pt = (mid_pt1 + mid_pt2) / 2
            center_pts.append(center_pt)
        return np.vstack(center_pts)
        
    def __len__(self):
        return self.dataset_length
    
    def parse_pt_poly_rec(self, ann):
        bbox = ann['bbox']
        x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        x2 = x1 + w 
        y2 = y1 
        x3 = x2 
        y3 = y1 + h 
        x4 = x1 
        y4 = y3 

        coords = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
        bezier_pt = gen_bezier_ctrl_points(coords)

        text = []
        for char in ann['attributes']['transcription']:
            if char in self.chars_dict:
                text.append(self.chars_dict[char])
            else:
                # for unknown char
                text.append(len(self.chars_dict))
        
        if len(text) <= self.rec_length:
            text = text + [self.rec_pad_index]*(self.rec_length - len(text))
        else:
            text = text[:self.rec_length]
        
        return bezier_pt, text

    def __getitem__(self, index):

        img_name = self.img_names[index]        
        img_anns = self.data[img_name]

        img_path = self.root_path + '/'.join(img_name.split('/')[-2:])
        image = Image.open(img_path).convert('RGB')
        image_w, image_h = image.size

        bezier_pts = []
        recs = []

        for ann in img_anns:
            if ann['attributes']['legible']:
                bezier_pt, rec = self.parse_pt_poly_rec(ann)
                bezier_pts.append(bezier_pt)
                recs.append(rec)

        if len(recs) == 0:
            random_index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_index)

        bezier_pts = torch.from_numpy(np.array(bezier_pts)).float().reshape(-1, 16)
        recs = torch.tensor(recs).reshape(-1, self.rec_length)

        target = {}
        target['image_id'] = img_name.split('/')[-1]
        target['file_name'] = img_name.split('/')[-1]

        target['image_folder'] = self.root_path + '/'.join(img_name.split('/')[-2:-1])
        target['dataset_name'] = self.dataset_name

        image_size = torch.tensor([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 

        target['recog'] = recs 
        target['bezier_pts'] = bezier_pts

        bboxes = []
        polygons = []
        for bezier_pt in bezier_pts:
            bezier_pt = bezier_pt.numpy()
            bbox = bezier2bbox(bezier_pt)
            polygon = bezier2polygon(bezier_pt)
            bboxes.append(bbox)
            polygons.append(polygon)

        bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        target['bboxes'] = bboxes

        polygons = torch.from_numpy(np.array(polygons)).float().reshape(-1, 32)
        target['polygons'] = polygons


        image, target = self._transforms(image, target)
        
        if 'train' in self.dataset_name:
            center_pts = self.sample_pts(target['bezier_pts'])
            center_pts = (center_pts*self.num_bins).astype(np.int32)
            center_pts = np.clip(center_pts, 0, self.num_bins - 1)

            # sorted center pts
            sorted_index = np.lexsort((center_pts[:, 0], center_pts[:, 1]))
            target['center_pts'] = torch.from_numpy(center_pts).type(torch.long)

            target['center_pts'] = target['center_pts'][sorted_index]
            target['polygons'] = target['polygons'][sorted_index]
            target['bboxes'] = target['bboxes'][sorted_index]
            target['bezier_pts'] = target['bezier_pts'][sorted_index]
            target['recog'] = target['recog'][sorted_index]

        return image, target




