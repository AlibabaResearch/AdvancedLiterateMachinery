import sys 
sys.path.append('.')

import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from copy import deepcopy 
from utils.misc import bezier2bbox, bezier2polygon


class RandomCrop(object):
    def __init__(self, min_size_ratio, max_size_ratio, prob):
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.prob = prob 

    def __call__(self, image, target):
        if random.random() > self.prob or len(target['bboxes']) == 0:
            return image, target 

        for _ in range(100):
            crop_w = int(image.width * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_h = int(image.height * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_region = transforms.RandomCrop.get_params(image, [crop_h, crop_w])
            cropped_image, cropped_target = self.crop(deepcopy(image), deepcopy(target), crop_region)
            if not cropped_image is None:
                return cropped_image, cropped_target
                
        print('Can not be cropped with texts')
        return image, target
    
    def crop(self, image, target, crop_region):
        bboxes = target['bboxes']
        crop_region, keep_instance = self.adjust_crop_region(bboxes, crop_region)
        
        if crop_region is None:
            return None, None

        cropped_image = F.crop(image, *crop_region)

        rg_ymin, rg_xmin, rg_h, rg_w = crop_region
        target['size'] = torch.tensor([rg_h, rg_w])
        if bboxes.shape[0] > 0:
            target['bboxes'] = target['bboxes'] - torch.tensor([rg_xmin, rg_ymin] * 2)
            target['bezier_pts'] = target['bezier_pts'] - torch.tensor([rg_xmin, rg_ymin] * 8)
            target['polygons'] = target['polygons'] - torch.tensor([rg_xmin, rg_ymin] * 16)
            for k in ['labels', 'area', 'iscrowd', 'recog', 'bboxes', 'bezier_pts', 'polygons', 'pts_class']:
                if k in target:
                    target[k] = target[k][keep_instance]

        return cropped_image, target

    def adjust_crop_region(self, bboxes, crop_region):
        rg_ymin, rg_xmin, rg_h, rg_w = crop_region 
        rg_xmax = rg_xmin + rg_w 
        rg_ymax = rg_ymin + rg_h 

        pre_keep = torch.zeros((bboxes.shape[0], ), dtype=torch.bool)
        while True:
            ov_xmin = torch.clamp(bboxes[:, 0], min=rg_xmin)
            ov_ymin = torch.clamp(bboxes[:, 1], min=rg_ymin)
            ov_xmax = torch.clamp(bboxes[:, 2], max=rg_xmax)
            ov_ymax = torch.clamp(bboxes[:, 3], max=rg_ymax)
            ov_h = ov_ymax - ov_ymin 
            ov_w = ov_xmax - ov_xmin 
            keep = torch.bitwise_and(ov_w > 0, ov_h > 0)

            if (keep == False).all():
                return None, None

            if keep.equal(pre_keep):
                break 

            keep_bboxes = bboxes[keep]
            keep_bboxes_xmin = int(min(keep_bboxes[:, 0]).item())
            keep_bboxes_ymin = int(min(keep_bboxes[:, 1]).item())
            keep_bboxes_xmax = int(max(keep_bboxes[:, 2]).item())
            keep_bboxes_ymax = int(max(keep_bboxes[:, 3]).item())
            rg_xmin = min(rg_xmin, keep_bboxes_xmin)
            rg_ymin = min(rg_ymin, keep_bboxes_ymin)
            rg_xmax = max(rg_xmax, keep_bboxes_xmax)
            rg_ymax = max(rg_ymax, keep_bboxes_ymax)

            pre_keep = keep
        
        crop_region = (rg_ymin, rg_xmin, rg_ymax - rg_ymin, rg_xmax - rg_xmin)
        return crop_region, keep


class RandomCropInst(object):
    def __init__(self, min_size_ratio, max_size_ratio, prob):
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.prob = prob 

    def __call__(self, image, target):
        if random.random() > self.prob or len(target['instance_bboxes']) == 0:
            return image, target 

        for _ in range(100):
            crop_w = int(image.width * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_h = int(image.height * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_region = transforms.RandomCrop.get_params(image, [crop_h, crop_w])
            cropped_image, cropped_target = self.crop(deepcopy(image), deepcopy(target), crop_region)
            if not cropped_image is None:
                return cropped_image, cropped_target
                
        print('Can not be cropped with texts')
        return image, target
    
    def crop(self, image, target, crop_region):
        bboxes = target['instance_bboxes']
        crop_region, keep_instance = self.adjust_crop_region(bboxes, crop_region)
        
        if crop_region is None:
            return None, None

        cropped_image = F.crop(image, *crop_region)

        rg_ymin, rg_xmin, rg_h, rg_w = crop_region
        target['size'] = torch.tensor([rg_h, rg_w])
        if bboxes.shape[0] > 0:
            single_keep_instance = torch.ones(len(target['bboxes'])).bool()
            index = 0
            for i in range(len(keep_instance)):
                if not keep_instance[i]:
                    single_keep_instance[index:index + target['instance_pt_nums'][i]] = False
                    index += target['instance_pt_nums'][i]
                else:
                    index += target['instance_pt_nums'][i]
            
            target['bboxes'] = target['bboxes'] - torch.tensor([rg_xmin, rg_ymin] * 2)
            target['bezier_pts'] = target['bezier_pts'] - torch.tensor([rg_xmin, rg_ymin] * 8)
            target['polygons'] = target['polygons'] - torch.tensor([rg_xmin, rg_ymin] * 16)

            target['instance_bboxes'] = target['instance_bboxes'] - torch.tensor([rg_xmin, rg_ymin] * 2)
            target['instance_bezier_pts'] = target['instance_bezier_pts'] - torch.tensor([rg_xmin, rg_ymin] * 8)
            target['instance_polygons'] = target['instance_polygons'] - torch.tensor([rg_xmin, rg_ymin] * 16)

            target['instance_bboxes'] = target['instance_bboxes'][keep_instance]
            target['instance_bezier_pts'] = target['instance_bezier_pts'][keep_instance]
            target['instance_polygons'] = target['instance_polygons'][keep_instance]

            target['instance_pt_nums'] = target['instance_pt_nums'][keep_instance]
            target['instance_classes'] = target['instance_classes'][keep_instance]

            for k in ['labels', 'area', 'iscrowd', 'recog', 'bboxes', 'bezier_pts', 'polygons', 'pts_class']:
                if k in target:
                    target[k] = target[k][single_keep_instance]

        return cropped_image, target

    def adjust_crop_region(self, bboxes, crop_region):
        rg_ymin, rg_xmin, rg_h, rg_w = crop_region 
        rg_xmax = rg_xmin + rg_w 
        rg_ymax = rg_ymin + rg_h 

        pre_keep = torch.zeros((bboxes.shape[0], ), dtype=torch.bool)
        while True:
            ov_xmin = torch.clamp(bboxes[:, 0], min=rg_xmin)
            ov_ymin = torch.clamp(bboxes[:, 1], min=rg_ymin)
            ov_xmax = torch.clamp(bboxes[:, 2], max=rg_xmax)
            ov_ymax = torch.clamp(bboxes[:, 3], max=rg_ymax)
            ov_h = ov_ymax - ov_ymin 
            ov_w = ov_xmax - ov_xmin 
            keep = torch.bitwise_and(ov_w > 0, ov_h > 0)

            if (keep == False).all():
                return None, None

            if keep.equal(pre_keep):
                break 

            keep_bboxes = bboxes[keep]
            keep_bboxes_xmin = int(min(keep_bboxes[:, 0]).item())
            keep_bboxes_ymin = int(min(keep_bboxes[:, 1]).item())
            keep_bboxes_xmax = int(max(keep_bboxes[:, 2]).item())
            keep_bboxes_ymax = int(max(keep_bboxes[:, 3]).item())
            rg_xmin = min(rg_xmin, keep_bboxes_xmin)
            rg_ymin = min(rg_ymin, keep_bboxes_ymin)
            rg_xmax = max(rg_xmax, keep_bboxes_xmax)
            rg_ymax = max(rg_ymax, keep_bboxes_ymax)

            pre_keep = keep
        
        crop_region = (rg_ymin, rg_xmin, rg_ymax - rg_ymin, rg_xmax - rg_xmin)
        return crop_region, keep


class RandomRotate(object):
    def __init__(self, max_angle, prob):
        self.max_angle = max_angle 
        self.prob = prob 

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target 
        
        angle = random.uniform(-self.max_angle, self.max_angle)
        image_w, image_h = image.size
        rotation_matrix = cv2.getRotationMatrix2D((image_w//2, image_h//2), angle, 1)
        image = image.rotate(angle, expand=True)

        new_w, new_h = image.size 
        target['size'] = torch.tensor([new_h, new_w])
        pad_w = (new_w - image_w) / 2
        pad_h = (new_h - image_h) / 2

        if 'bezier_pts' in target.keys():
            bezier_pts = target['bezier_pts'].numpy()
            bezier_pts = bezier_pts.reshape(-1, 8, 2)
            bezier_pts = self.rotate_points(bezier_pts, rotation_matrix, (pad_w, pad_h))
            bezier_pts = bezier_pts.reshape(-1, 16)
            target['bezier_pts'] = torch.from_numpy(bezier_pts).type(torch.float32)

            bboxes = [bezier2bbox(ele) for ele in bezier_pts]
            target['bboxes'] = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)

            polygons = [bezier2polygon(ele) for ele in bezier_pts]
            target['polygons']  = torch.from_numpy(np.array(polygons)).float().reshape(-1, 32)

        if 'instance_bezier_pts' in target.keys():
            bezier_pts = target['instance_bezier_pts'].numpy()
            bezier_pts = bezier_pts.reshape(-1, 8, 2)
            bezier_pts = self.rotate_points(bezier_pts, rotation_matrix, (pad_w, pad_h))
            bezier_pts = bezier_pts.reshape(-1, 16)
            target['instance_bezier_pts'] = torch.from_numpy(bezier_pts).type(torch.float32)

            bboxes = [bezier2bbox(ele) for ele in bezier_pts]
            target['instance_bboxes'] = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)

            polygons = [bezier2polygon(ele) for ele in bezier_pts]
            target['instance_polygons']  = torch.from_numpy(np.array(polygons)).float().reshape(-1, 32)

        return image, target

    def rotate_points(self, coords, rotation_matrix, paddings):
        coords = np.pad(coords, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
        coords = np.dot(coords, rotation_matrix.transpose())
        coords[:, :, 0] += paddings[0]
        coords[:, :, 1] += paddings[1]
        return coords


class RandomResize(object):
    def __init__(self, min_sizes, max_size):
        self.min_sizes = min_sizes
        self.max_size = max_size 
    
    def __call__(self, image, target):
        min_size = random.choice(self.min_sizes)
        size = self.get_size_with_aspect_ratio(image.size, min_size, self.max_size)
        rescaled_image = F.resize(image, size)

        ratio_width = rescaled_image.size[0] / image.size[0]
        ratio_height = rescaled_image.size[1] / image.size[1]

        target['size'] = torch.tensor(size)
        # target['area'] = target['area'] * (ratio_width * ratio_height)
        if 'bboxes' in target:
            target['bboxes'] = target['bboxes'] * torch.tensor([ratio_width, ratio_height] * 2)
        if 'bezier_pts' in target:
            target['bezier_pts'] = target['bezier_pts'] * torch.tensor([ratio_width, ratio_height] * 8)
        if 'polygons' in target:
            target['polygons'] = target['polygons'] * torch.tensor([ratio_width, ratio_height] * 16)
        if 'center_pts' in target:
            target['center_pts'][:,:2] = target['center_pts'][:,:2] * torch.tensor([ratio_width, ratio_height])

        if 'instance_bezier_pts' in target:
            target['instance_bboxes'] = target['instance_bboxes'] * torch.tensor([ratio_width, ratio_height] * 2)
            target['instance_bezier_pts'] = target['instance_bezier_pts'] * torch.tensor([ratio_width, ratio_height] * 8)
            target['instance_polygons'] = target['instance_polygons'] * torch.tensor([ratio_width, ratio_height] * 16)

        return rescaled_image, target

    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)


class RandomDistortion(object):
    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.prob = prob 
        self.tfm = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target 
        return self.tfm(image), target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __call__(self, image, target):
        if target is None:
            return image, target 
        
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        h, w = image.shape[-2:]

        if 'bboxes' in target:
            target['bboxes'] = target['bboxes'] / torch.tensor([w, h] * 2, dtype=torch.float32)
        if 'bezier_pts' in target:
            target['bezier_pts'] = target['bezier_pts'] / torch.tensor([w, h] * 8, dtype=torch.float32)
        if 'polygons' in target:
            target['polygons'] = target['polygons'] / torch.tensor([w, h] * 16, dtype=torch.float32)
        if 'center_pts' in target:
            target['center_pts'][:,:2] = target['center_pts'][:,:2] / torch.tensor([w, h], dtype=torch.float32)

        if 'instance_bezier_pts' in target:
            target['instance_bboxes'] = target['instance_bboxes'] / torch.tensor([w, h] * 2, dtype=torch.float32)
            target['instance_bezier_pts'] = target['instance_bezier_pts'] / torch.tensor([w, h] * 8, dtype=torch.float32)
            target['instance_polygons'] = target['instance_polygons'] / torch.tensor([w, h] * 16, dtype=torch.float32)

        return image, target 


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string