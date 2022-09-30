import os
import time
import string
import argparse
import re
import PIL
import math

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from matplotlib import colors
import cv2
from torchvision import transforms
import torchvision.utils as vutils

from utils import TokenLabelConverter
from models import Model
from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_model(image_tensors, model, converter, opt):
    image = image_tensors.to(device)
    batch_size = image.shape[0]

    attens, char_preds, bpe_preds, wp_preds = model(image, is_eval=True) # final
    
    # char pred
    _, char_pred_index = char_preds.topk(1, dim=-1, largest=True, sorted=True)
    char_pred_index = char_pred_index.view(-1, converter.batch_max_length)
    length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
    char_preds_str = converter.char_decode(char_pred_index[:, 1:], length_for_pred)
    char_pred_prob = F.softmax(char_preds, dim=2)
    char_pred_max_prob, _ = char_pred_prob.max(dim=2)
    char_preds_max_prob = char_pred_max_prob[:, 1:]
    
    # bpe pred
    _, bpe_preds_index = bpe_preds.topk(1, dim=-1, largest=True, sorted=True)
    bpe_preds_index = bpe_preds_index.view(-1, converter.batch_max_length)
    bpe_preds_str = converter.bpe_decode(bpe_preds_index[:,1:], length_for_pred)
    bpe_preds_prob = F.softmax(bpe_preds, dim=2)
    bpe_preds_max_prob, _ = bpe_preds_prob.max(dim=2)
    bpe_preds_max_prob = bpe_preds_max_prob[:, 1:]
    bpe_preds_index = bpe_preds_index[:, 1:]

    # wp pred
    _, wp_preds_index = wp_preds.topk(1, dim=-1, largest=True, sorted=True)
    wp_preds_index = wp_preds_index.view(-1, converter.batch_max_length)
    wp_preds_str = converter.wp_decode(wp_preds_index[:,1:], length_for_pred)
    wp_preds_prob = F.softmax(wp_preds, dim=2)
    wp_preds_max_prob, _ = wp_preds_prob.max(dim=2)
    wp_preds_max_prob = wp_preds_max_prob[:, 1:]
    wp_preds_index = wp_preds_index[:, 1:]

    # for index in range(image.shape[0]):
    index = 0

    # char
    char_pred = char_preds_str[index]
    char_pred_max_prob = char_preds_max_prob[index]
    char_pred_EOS = char_pred.find('[s]')
    char_pred = char_pred[:char_pred_EOS]  # prune after "end of sentence" token ([s])

    char_pred_max_prob = char_pred_max_prob[:char_pred_EOS+1]
    try:
        char_confidence_score = char_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
    except:
        char_confidence_score = 0.0
    print('char:', char_pred, char_confidence_score)

    # bpe
    bpe_pred = bpe_preds_str[index]
    bpe_pred_max_prob = bpe_preds_max_prob[index]
    bpe_pred_EOS = bpe_pred.find('#')
    bpe_pred = bpe_pred[:bpe_pred_EOS]

    bpe_pred_index = bpe_preds_index[index].cpu().tolist()
    try:
        bpe_pred_EOS_index = bpe_pred_index.index(2)
    except:
        bpe_pred_EOS_index = -1
    bpe_pred_max_prob = bpe_pred_max_prob[:bpe_pred_EOS_index+1]
    try:
        bpe_confidence_score = bpe_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
    except:
        bpe_confidence_score = 0.0
    print('bpe:', bpe_pred, bpe_confidence_score)

    # wp
    wp_pred = wp_preds_str[index]
    wp_pred_max_prob = wp_preds_max_prob[index]
    wp_pred_EOS = wp_pred.find('[SEP]')
    wp_pred = wp_pred[:wp_pred_EOS]

    wp_pred_index = wp_preds_index[index].cpu().tolist()
    try:
        wp_pred_EOS_index = wp_pred_index.index(102)
    except:
        wp_pred_EOS_index = -1
    wp_pred_max_prob = wp_pred_max_prob[:wp_pred_EOS_index+1]
    try:
        wp_confidence_score = wp_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
    except:
        wp_confidence_score = 0.0
    print('wp:', wp_pred, wp_confidence_score)

    # draw atten
    pil = transforms.ToPILImage()
    tensor = transforms.ToTensor()
    size = opt.imgH , opt.imgW
    resize = transforms.Resize(size=size, interpolation=0)
    char_atten = attens[0][index]
    bpe_atten = attens[1][index]
    wp_atten = attens[2][index]
    char_atten = char_atten[:, 1:].view(-1, 8, 32)
    char_atten = char_atten[1:char_pred_EOS+1]
    draw_atten(opt.demo_imgs, char_pred, char_atten, pil, tensor, resize, flag='char')

def load_img(img_path, opt):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    img_arr = np.array(img)
    img_tensor = transforms.ToTensor()(img)
    image_tensor = img_tensor.unsqueeze(0)
    return image_tensor
    
def draw_atten(img_path, pred, attn, pil, tensor, resize, flag=''):
    image = PIL.Image.open(img_path).convert('RGB')
    image = cv2.resize(np.array(image), (128, 32))
    
    image = tensor(image)
    image_np = np.array(pil(image))

    attn_pil = [pil(a) for a in attn[:, None, :, :]]
    attn = [tensor(resize(a)).repeat(3, 1, 1) for a in attn_pil]
    attn_sum = np.array([np.array(a) for a in attn_pil[:len(pred)]]).sum(axis=0)
    blended_sum = tensor(blend_mask(image_np, attn_sum))
    blended = [tensor(blend_mask(image_np, np.array(a))) for a in attn_pil]
    save_image = torch.stack([image] + attn + [blended_sum] + blended)
    save_image = save_image.view(2, -1, *save_image.shape[1:])
    save_image = save_image.permute(1, 0, 2, 3, 4).flatten(0, 1)
    
    gt = os.path.basename(img_path).split('.')[0]
    vutils.save_image(save_image, f'demo_imgs/attens/{gt}_{pred}_{flag}.jpg', nrow=2, normalize=True, scale_each=True)

def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask-mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask,(image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:,:,:3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255 
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1]) 
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1-color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1-alpha, 0)

    return blended_img


def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # load img
    if os.path.isdir(opt.demo_imgs):
        imgs = [os.path.join(opt.demo_imgs, fname) for fname in os.listdir(opt.demo_imgs)]
        imgs = [img for img in imgs if img.endswith('.jpg') or img.endswith('.png')]
    else:
        imgs = [opt.demo_imgs]
    
    for img in imgs:
        opt.demo_imgs = img
        img_tensor = load_img(opt.demo_imgs, opt)
        print('imgs:', img)

        """ evaluation """
        model.eval()
        opt.eval = True
        with torch.no_grad():
            run_model(img_tensor, model, converter, opt)
        print('============================================================================')


if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    opt.saved_model = opt.model_dir
    test(opt)
