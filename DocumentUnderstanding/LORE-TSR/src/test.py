from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
from PIL import Image
import matplotlib
import torch.nn as nn

#from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    img = image.copy()
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA) 
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'img':img, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def id2name():
  path = opt.save_path + '/json/test.json'
  data = json.load(open(path,'r'))
  Id2Name = {}
  for info in data["images"]:
    ID = info['id']
    file_name = info['file_name']
    Id2Name[ID]=file_name
  return Id2Name

def save_corner(opt,corners,name):
  path = opt.save_path + '/corner/'
  f = open(path+name+'.txt','w')
  for i in range(len(corners)):
      box = corners[i]
      f.write(str(box[0])+','+str(box[1])+';'+str(box[2])+','+str(box[3])+';'+\
              str(box[4])+','+str(box[5])+';'+str(box[6])+','+str(box[7])+';'+\
              str(box[8])+','+str(box[9])+'\n')
  f.close()


def save_corner_json(corners):
  CorNer = []
  corner = {}
  for img_id in corners:
    result = corners[img_id]
    cor_over_thresh = []
    for point in result:
      if point[2] > opt.vis_thresh_corner:
        cor_over_thresh.append([float(point[0]),float(point[1])])
    corner[img_id] = cor_over_thresh
  CorNer.append(corner)
  json.dump(CorNer,open('/home/rujiao.lrj/CenterNet_4point_Mask_4_rotate/data/code/corner.json','w'))

def save_det(opt,result,name):
  path = opt.save_path + '/detect/'
  f = open(path+name+'.txt','w')
  for i in range(len(result)):
      box = result[i]
      if box[8] < opt.scores_thresh:
          continue
      f.write(str(box[0])+','+str(box[1])+';'+str(box[2])+','+str(box[3])+';'+\
              str(box[4])+','+str(box[5])+';'+str(box[6])+','+str(box[7])+';'+\
              str(box[8])+'\n')
  f.close()

def save_img_txt(img):
  shape = list(img.shape)
  f1 = open('/home/rujiao.lrj/CenterNet_4point_Mask_4_rotate_offset/src/img.txt','w')
  f = open('/home/rujiao.lrj/CenterNet_4point_Mask_4_rotate_offset/src/save_map.txt','w')
  f.write('data:\n')
  for i in range(shape[0]):
    for j in range(shape[1]):
      if j==16:
        break
      string = ''
      for k in range(shape[2]):
        if k == 16:
          break
        data = img[i][j][k].item() 
        string = string + str(data)+' '
      f.write(str(string)+'\n')
  f.close()

  for i in range(shape[0]):
    for j in range(shape[1]):
      for k in range(shape[2]):
        data = img[i][j][k].item() 
        f1.write(str(data)+'\n')
  f1.close()

def save_img(img,hm):
  shape1 = img.shape #256,256,3
  shape2 = hm.shape  #1,256,256
  for i in range(256):
    for j in range(256):
      img[i][j][0] = img[i][j][0] * hm[0][i][j]
      img[i][j][1] = img[i][j][1] * hm[0][i][j]
      img[i][j][2] = img[i][j][2] * hm[0][i][j]
  return img

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  corners = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  i = 0 
  ID2NAME = id2name()
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    index = ID2NAME[img_id.item()].rfind('.')
    imgname = ID2NAME[img_id.item()][0:index]
    results[str(img_id.item())] = ret['results']
    ps = ret['4ps'][1]
    st = ret['corner_st_reg']
    save_det(opt,ps,imgname)
    #save_corner(opt,st,imgname)
    #break
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  #save_corner(corners)
  #dataset.run_eval(results, opt.save_dir, opt.scores_thresh)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']
    
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
    
  bar.finish()
  #dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)
