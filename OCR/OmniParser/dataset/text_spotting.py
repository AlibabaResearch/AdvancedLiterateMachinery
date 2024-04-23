import random
import numpy as np
import torch
import torchvision
from copy import deepcopy
from utils.misc import bezier2bbox, bezier2polygon, sample_bezier_curve
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None


class TextSpottingDataset(torchvision.datasets.CocoDetection):
    def __init__(self, image_folder, anno_file, dataset_name, transforms, args):
        super(TextSpottingDataset, self).__init__(image_folder, anno_file)
        self.dataset_name = dataset_name 
        self.image_folder = image_folder
        self._transforms = transforms 
        self.rec_length = args.rec_length
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
        
    def __getitem__(self, index):
        image, anno = super(TextSpottingDataset, self).__getitem__(index)

        image_w, image_h = image.size
        anno = [ele for ele in anno if 'iscrowd' not in anno or ele['iscrowd'] == 0]
        
        if len(anno) == 0 and 'train' in self.dataset_name:
            print(self.dataset_name, self.coco.loadImgs(self.ids[index])[0]['file_name'])
            random_index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_index)

        target = {}
        target['image_id'] = self.ids[index]
        target['file_name'] = self.coco.loadImgs(self.ids[index])[0]['file_name']
        target['image_folder'] = self.image_folder
        target['dataset_name'] = self.dataset_name

        image_size = torch.tensor([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 

        recog = [ele['rec'] for ele in anno]
        recog = torch.tensor(recog, dtype=torch.long).reshape(-1, self.rec_length)
        target['recog'] = recog 

        bezier_pts = [ele['bezier_pts'] for ele in anno]
        bezier_pts = torch.tensor(bezier_pts, dtype=torch.float32).reshape(-1, 16)
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
    