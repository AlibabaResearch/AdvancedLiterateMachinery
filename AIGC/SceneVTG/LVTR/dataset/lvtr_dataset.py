# coding:utf-8
import math
import os
import pdb
import random
import sys

import cv2
import lmdb
import numpy as np
import six
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset, sampler


class DatasetLVTR(Dataset):
    def __init__(
        self,
        roots=None,
        ratio=None,
        img_height=32,
        img_width=128,
        fullimg_size=(512, 512),
        transform=None,
        global_state="Test",
        augmentation=False,
        render_height=32,
        dataset_root='./SceneVTG-Erase/LVTR_data/data',
    ):

        self.dataset_root = dataset_root
        self.global_state = global_state
        self.training = False
        if self.global_state == "Train":
            self.training = True

        self.totalpaths = []
        for root in roots:
            self.totalpaths.extend([l for l in open(root).readlines()])

        if self.training:
            random.shuffle(self.totalpaths)

        self.nSamples = len(self.totalpaths)

        self.font_type = "./dataset/ARIALBD.TTF"

        self.img_height = img_height
        self.img_width = img_width
        self.render_height = render_height
        self.fullimg_size = fullimg_size
        self.augmentation = augmentation
        self.global_cnt = 0

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        if self.training:
            return 12*8*6000
        else:
            return self.nSamples

    def __getitem__(self, index):
        if self.training:
            index = random.randint(0, self.nSamples - 1)
        index = index % self.nSamples
        line = self.totalpaths[index]

        datasetname, sampleindex, boxtotal, ocrtotal = line.strip("\n").split("\t")
        sampleindex = sampleindex[:-4]
        imagepath = '/'.join([self.dataset_root, datasetname, 'images_ori', sampleindex+'.jpg'])
        image_removal_path = '/'.join([self.dataset_root, datasetname, 'images_re', sampleindex+'.jpg'])

        try:
            imagefull = Image.open(imagepath).convert("RGB")
            image_removal = Image.open(image_removal_path).convert("RGB")
        except:
            return self[index + 1]
        
        w_ratio = 1.0
        h_ratio = 1.0            

        if (
            image_removal.size[0] != imagefull.size[0]
            or image_removal.size[1] != imagefull.size[1]
        ):
            w_ratio = imagefull.size[0] / self.fullimg_size[0]
            h_ratio = imagefull.size[1] / self.fullimg_size[1]
            imagefull = imagefull.resize(self.fullimg_size)

        boxtotal = eval(boxtotal)
        ocrtotal = eval(ocrtotal)
        linebox = boxtotal[0]
        ocr_strings = ocrtotal[0]
        points = np.array(linebox).reshape(-1, 2)
        points[:, 0] = points[:, 0] / w_ratio
        points[:, 1] = points[:, 1] / h_ratio
        xmin = min(points[:, 0])
        xmax = max(points[:, 0])
        ymin = min(points[:, 1])
        ymax = max(points[:, 1])

        line_mask_poly = np.zeros(
            (imagefull.size[1], imagefull.size[0]), dtype=np.uint8
        )
        cv2.fillPoly(line_mask_poly, [points], 1)

        word_mask_poly = np.zeros(
            (imagefull.size[1], imagefull.size[0]), dtype=np.uint8
        )
        for boxitem in boxtotal[1:]:
            boxitem = np.array(boxitem).reshape(-1, 2)
            boxitem[:, 0] = boxitem[:, 0] / w_ratio
            boxitem[:, 1] = boxitem[:, 1] / h_ratio
            cv2.fillPoly(word_mask_poly, [boxitem], 1)

        if np.random.rand() < 0.5:
            image_removal_fix = Image.composite(
                imagefull, image_removal, Image.fromarray((1 - line_mask_poly) * 255)
            )
        else:
            image_removal_fix = image_removal
            imagefull = Image.composite(
                image_removal, imagefull, Image.fromarray((1 - line_mask_poly) * 255)
            )

        line_mask_poly = Image.fromarray(line_mask_poly * 255)
        word_mask_poly = Image.fromarray(word_mask_poly * 255)

        image_removal_fix_resize = image_removal_fix.resize(self.fullimg_size)

        patchwidth = xmax - xmin
        patchheight = ymax - ymin
        height_real = patchheight * 1.4
        ystart = max(0, ymin - int((height_real - patchheight) * 0.5))
        yend = min(imagefull.size[1] - 1, ystart + height_real)
        ystart = max(0, yend - height_real)
        height_real = yend - ystart
        width_real = max(
            height_real / self.img_height * self.img_width, 1.2 * patchwidth
        )
        xstart = max(0, (xmin + xmax) // 2 - int(width_real * 0.5))
        xend = min(imagefull.size[0] - 1, xstart + width_real)
        width_real = xend - xstart

        try:
            crop_box = (xstart, ystart, xend, yend)
            image_crop = imagefull.crop(crop_box)
            image_removal_crop = image_removal_fix.crop(crop_box)
            line_mask_poly_crop = line_mask_poly.crop(crop_box)
            word_mask_poly_crop = word_mask_poly.crop(crop_box)
        except:
            return self[index + 1]

        width_resize = self.img_width

        crop_box_resize = (round(xstart), round(ystart), round(xend), round(yend))

        padding_left = (self.img_width - width_resize) // 2
        padding_right = self.img_width - padding_left - width_resize

        image_crop = image_crop.resize((width_resize, self.img_height))
        image_crop_bk = Image.new(
            "RGB", (self.img_width, self.img_height), (255, 255, 255)
        )
        image_crop_bk.paste(image_crop, (padding_left, 0))
        image_crop = image_crop_bk

        image_removal_crop = image_removal_crop.resize((width_resize, self.img_height))
        image_removal_crop_bk = Image.new(
            "RGB", (self.img_width, self.img_height), (255, 255, 255)
        )
        image_removal_crop_bk.paste(image_removal_crop, (padding_left, 0))
        image_removal_crop = image_removal_crop_bk

        line_mask_poly_crop = line_mask_poly_crop.resize(
            (width_resize, self.img_height)
        )
        line_mask_poly_crop_bk = Image.new(
            "L", (self.img_width, self.img_height), (255)
        )
        line_mask_poly_crop_bk.paste(line_mask_poly_crop, (padding_left, 0))
        line_mask_poly_crop = line_mask_poly_crop_bk

        word_mask_poly_crop = word_mask_poly_crop.resize(
            (width_resize, self.img_height)
        )
        word_mask_poly_crop_bk = Image.new(
            "L", (self.img_width, self.img_height), (255)
        )
        word_mask_poly_crop_bk.paste(word_mask_poly_crop, (padding_left, 0))
        word_mask_poly_crop = word_mask_poly_crop_bk

        mask_line = Image.new("L", (width_resize, self.img_height), "black")
        mask_line = ImageOps.pad(
            mask_line, (self.img_width, self.img_height), color=(255)
        )

        font_new = ImageFont.truetype(self.font_type, 24)
        text_width_new, text_height_new = font_new.getsize(ocr_strings)
        image_width = max(text_width_new, self.img_width)
        crop_render = Image.new("RGB", (image_width, self.render_height), "white")
        draw = ImageDraw.Draw(crop_render)
        x = (
            (image_width - text_width_new) // 2
            if text_width_new < self.img_width
            else 0
        )
        y = (crop_render.height - text_height_new) // 2
        draw.text((x, y), ocr_strings, font=font_new, fill="black")
        crop_render = ImageOps.pad(
            crop_render, (self.img_width, self.render_height), color=(255, 255, 255)
        )

        image_crop_withoutbackground = np.array(image_crop)
        image_crop_withoutbackground[np.array(line_mask_poly_crop) == 0] = 0
        image_crop_withoutbackground = Image.fromarray(image_crop_withoutbackground)

        self.global_cnt += 1

        sample = {}
        sample["crop_image"] = transforms.ToTensor()(image_crop)
        sample["crop_image_wobk"] = transforms.ToTensor()(image_crop_withoutbackground)
        sample["crop_render"] = transforms.ToTensor()(crop_render)
        sample["crop_ocrstring"] = ocr_strings
        sample["mask_line"] = transforms.ToTensor()(mask_line)
        sample["crop_image_removal"] = transforms.ToTensor()(image_removal_crop)
        sample["line_poly_mask"] = transforms.ToTensor()(line_mask_poly_crop)
        sample["word_poly_mask"] = transforms.ToTensor()(word_mask_poly_crop)
        sample["image_removal_fix"] = transforms.ToTensor()(image_removal_fix_resize)
        sample["crop_box"] = np.array(crop_box_resize)
        sample["width_resize"] = np.array(width_resize)
        sample["image_full"] = transforms.ToTensor()(
            imagefull.resize(self.fullimg_size)
        )
        sample["imagepath"] = image_removal_path
        return sample

