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


class CordDataset(Dataset):

    def __init__(self, dataset_name, transforms, args):
        
        self._transforms = transforms

        self.dataset_name = dataset_name
        if 'train' in dataset_name:
            self.split = 'train'
        else:
            self.split = 'val'

        if self.split == 'train':
            self.img_path = './data/text_spotting_datasets/cord/cord-v2/data/images/trainval/'
        else:
            self.img_path = './data/text_spotting_datasets/cord/cord-v2/data/images/test/'

        self.img_names = os.listdir(self.img_path)
        self.img_names = [x for x in self.img_names if int(x[:-4]) < 800]
        self.img_names.sort(key=lambda x: int(x[:-4]))
        self.dataset_length = len(self.img_names)

        self.chars_dict = {}
        for i in range(len(args.chars)):
            self.chars_dict[args.chars[i]] = i
        
        self.classes = ['menu.cnt', 'menu.discountprice', 'menu.etc', 'menu.itemsubtotal', 'menu.nm', 'menu.num',
                        'menu.price', 'menu.sub.cnt', 'menu.sub.nm', 'menu.sub.price', 'menu.sub.unitprice', 'menu.unitprice', 
                        'menu.vatyn', 'sub_total.discount_price', 'sub_total.etc', 'sub_total.othersvc_price', 'sub_total.service_price',
                        'sub_total.subtotal_price', 'sub_total.tax_price', 'total.cashprice', 'total.changeprice', 'total.creditcardprice', 
                        'total.emoneyprice', 'total.menuqty_cnt', 'total.menutype_cnt', 'total.total_etc', 'total.total_price', 'void_menu.nm', 'void_menu.price']
        
        self.class2index = {}
        for i in range(len(self.classes)):
            self.class2index[self.classes[i]] = args.padding_index + 1 + i

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
    
    def parse_pt_poly_rec(self, word, category):
        quad = word['quad']
        x1, x2, x3, x4 = int(quad['x1']), int(quad['x2']), int(quad['x3']), int(quad['x4'])
        y1, y2, y3, y4 = int(quad['y1']), int(quad['y2']), int(quad['y3']), int(quad['y4'])

        center_x = int((x1 + x2 + x3 + x4) / 4)
        center_y = int((y1 + y2 + y3 + y4) / 4)

        pt_class = self.class2index[category]
        coords = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])

        bezier_pt = gen_bezier_ctrl_points(coords)

        text = []
        for char in word['text']:
            if char in self.chars_dict:
                text.append(self.chars_dict[char])
            else:
                # for unknown char
                text.append(len(self.chars_dict))
        
        if len(text) <= self.rec_length:
            text = text + [self.rec_pad_index]*(self.rec_length - len(text))
        else:
            text = text[:self.rec_length]
        
        return pt_class, bezier_pt, text

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.img_names[index])
        label_path = img_path.replace('images', 'anns').replace('.png', '.json')

        image = Image.open(img_path).convert('RGB')
        image_w, image_h = image.size

        with open(label_path, 'r') as f:
            gt = json.load(f)

        total_words = gt['valid_line']

        # for words
        pts_class = []
        bezier_pts = []
        recs = []

        # for instances
        instance_classes = []
        instance_bezier_pts = []
        instance_pt_nums = []

        for words in total_words:
            # init instance bbox
            tmp_x1, tmp_y1, tmp_x3, tmp_y3 = float('inf'), float('inf'), 0, 0
            tmp_pts_class = []

            for word in words['words']:
                if word['is_key'] == 1:
                    continue
                pt_class, bezier_pt, rec = self.parse_pt_poly_rec(word, words['category'])
                pts_class.append(pt_class)
                tmp_pts_class.append(pt_class)
                bezier_pts.append(bezier_pt)
                recs.append(rec)
                
                tmp_x1 = min(tmp_x1, int(word['quad']['x1']))
                tmp_y1 = min(tmp_y1, int(word['quad']['y1']))
                tmp_x3 = max(tmp_x3, int(word['quad']['x3']))
                tmp_y3 = max(tmp_y3, int(word['quad']['y3']))
            
            if tmp_x1 == float('inf'):
                continue
            
            tmp_x2 = tmp_x3
            tmp_y2 = tmp_y1
            tmp_x4 = tmp_x1
            tmp_y4 = tmp_y3

            tmp_coords = np.array([[tmp_x1,tmp_y1], [tmp_x2,tmp_y2], [tmp_x3,tmp_y3], [tmp_x4,tmp_y4]])
            tmp_bezier_pt = gen_bezier_ctrl_points(tmp_coords)

            instance_classes.append(self.class2index[words['category']])
            instance_bezier_pts.append(tmp_bezier_pt)
            instance_pt_nums.append(len(tmp_pts_class))

        pts_class = torch.tensor(pts_class).reshape(-1, 1).float()
        bezier_pts = torch.from_numpy(np.array(bezier_pts)).float().reshape(-1, 16)
        recs = torch.tensor(recs).reshape(-1, 25)

        instance_bezier_pts = torch.from_numpy(np.array(instance_bezier_pts)).float().reshape(-1, 16)

        target = {}
        target['image_id'] = self.img_names[index]
        target['file_name'] = self.img_names[index]

        target['image_folder'] = self.img_path
        target['dataset_name'] = self.dataset_name

        image_size = torch.tensor([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 

        target['pts_class'] = pts_class 
        target['recog'] = recs 
        target['bezier_pts'] = bezier_pts

        target['instance_bezier_pts'] = instance_bezier_pts

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

        instance_bboxes = []
        instance_polygons = []
        for instance_bezier_pt in instance_bezier_pts:
            instance_bezier_pt = instance_bezier_pt.numpy()
            instance_bbox = bezier2bbox(instance_bezier_pt)
            instance_polygon = bezier2polygon(instance_bezier_pt)
            instance_bboxes.append(instance_bbox)
            instance_polygons.append(instance_polygon)

        instance_bboxes = torch.tensor(instance_bboxes, dtype=torch.float32).reshape(-1, 4)
        target['instance_bboxes'] = instance_bboxes

        instance_polygons = torch.from_numpy(np.array(instance_polygons)).float().reshape(-1, 32)
        target['instance_polygons'] = instance_polygons

        target['instance_pt_nums'] = torch.tensor(instance_pt_nums)
        target['instance_classes'] = torch.tensor(instance_classes)

        image, target = self._transforms(image, target)
        
        if 'train' in self.dataset_name:
            center_pts = self.sample_pts(target['bezier_pts'])
            center_pts = (center_pts*self.num_bins).astype(np.int32)
            center_pts = np.clip(center_pts, 0, self.num_bins - 1)
            center_pts = torch.from_numpy(center_pts).type(torch.long)
            target['center_pts'] = center_pts

            instance_center_pts = self.sample_pts(target['instance_bezier_pts'])
            instance_center_pts = (instance_center_pts*self.num_bins).astype(np.int32)
            instance_center_pts = np.clip(instance_center_pts, 0, self.num_bins - 1)
            target['instance_center_pts'] = torch.from_numpy(instance_center_pts).type(torch.long)

            # sorted center pts
            sorted_index = np.lexsort((instance_center_pts[:, 0], instance_center_pts[:, 1]))

            pt_index = []
            index = 0
            for i in range(len(target['instance_pt_nums'])):
                pt_index.append([index, index + target['instance_pt_nums'][i].item()])
                index += target['instance_pt_nums'][i].item()

            sorted_instance_pts = []
            for index in sorted_index:
                start_index, end_index = pt_index[index]
                sorted_instance_pts.append(center_pts[start_index:end_index].reshape(-1))
                sorted_instance_pts.append(target['instance_classes'][index].unsqueeze(0))
                
            sorted_instance_pts = torch.cat(sorted_instance_pts)
            target['sorted_instance_pts'] = sorted_instance_pts
        
        return image, target