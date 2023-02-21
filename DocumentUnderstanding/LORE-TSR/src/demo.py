from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from tqdm import tqdm

import cv2
import pycocotools.coco as coco
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  
  if not opt.anno_path == '':

    image_names = []
    image_annos = []
    coco_data = coco.COCO(opt.anno_path)
    images = coco_data.getImgIds()
  
    for i in range(len(images)):
      img_id = images[i]
      if opt.dataset_name == 'WTW':
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']
      elif opt.dataset_name == 'PTN':
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name'].replace('.jpg', '.png')
      else:
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']

      ann_ids = coco_data.getAnnIds(imgIds=[img_id])
      anns = coco_data.loadAnns(ids=ann_ids)

      image_names.append(os.path.join(opt.demo, file_name))
      image_annos.append(anns)
  
  elif os.path.isdir(opt.demo):

    image_names = []
    ls = os.listdir(opt.demo)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opt.demo, file_name))
  else:

    image_names = [opt.demo]

  if not os.path.exists(opt.output_dir + opt.demo_name):
      os.makedirs(opt.output_dir + opt.demo_name +'/center/')
      os.makedirs(opt.output_dir + opt.demo_name +'/corner/')
      os.makedirs(opt.output_dir + opt.demo_name +'/logi/')

  if not os.path.exists(opt.demo_dir):
    os.makedirs(opt.demo_dir)

  for i in tqdm(range(len(image_names))):
      image_name = image_names[i]
      if not opt.wiz_detect:
        image_anno = image_annos[i]
        ret = detector.run(opt, image_name, image_anno)
      else:
        #image_anno = image_annos[i]
        #print(image_name)
      
        ret = detector.run(opt, image_name)


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
