from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _flatten_and_gather_feat, _tranpose_and_gather_feat
import torch.nn.functional as F

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class AxisLoss(nn.Module):
  def __init__(self):
    super(AxisLoss, self).__init__()
  
  def forward(self, output, mask, ind, target, logi=None):
    span_type = False
    #computing vanilla axis loss
    if logi is None:
      pred = _tranpose_and_gather_feat(output, ind)
    else:
      pred = logi

    #mask = mask.unsqueeze(2).expand_as(pred).float()
    mask = mask.unsqueeze(2).float()
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    #loss = F.mse_loss(2*pred * mask, 2*target * mask, size_average=False)
    #loss = F.mse_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (4*(mask.sum() + 1e-4))

    #construct span data
    target_width = target[:,:,1] - target[:,:,0]
    target_height = target[:,:,3] - target[:,:,2]
    width = pred[:,:,1] - pred[:,:,0]
    height = pred[:,:,3] - pred[:,:,2]
    target_span = torch.cat((target_width.unsqueeze(2), target_height.unsqueeze(2)), axis = 2)
    span = torch.cat((width.unsqueeze(2), height.unsqueeze(2)), axis = 2)

    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    
    #solving unbalanced type in span data 
    if span_type :
      major_mask = (target_span == 0) * mask
      minor_mask = (target_span != 0) * mask
      
      major_loss = F.l1_loss(span  * major_mask, target_span * major_mask, size_average=False)
      major_loss = major_loss/ (major_mask.sum())

      minor_loss = F.l1_loss(span  * minor_mask, target_span * minor_mask, size_average=False)
      minor_loss = minor_loss/ (minor_mask.sum())
    
    else:  
      span_w = torch.exp(torch.abs(span - target_span)/2)
      loss_span = F.l1_loss(span * mask, target_span * mask, size_average=False)
      #loss_span = F.mse_loss(2*span * mask, 2*target_span * mask, size_average=False)
      #loss_span = F.mse_loss(span * mask, target_span * mask, size_average=False)
      loss_span = loss_span / (2*(mask.sum() + 1e-4))
    
    return loss, loss_span

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class PairLoss(nn.Module):
  def __init__(self):
    super(PairLoss, self).__init__()
  
  def forward(self, output1, ind1, output2, ind2, mask, mask_cro, ctr_cro_ind, target1, target2, hm_ctxy):
 
    pred1 = _tranpose_and_gather_feat(output1, ind1) #bxmx8
    pred2 = _tranpose_and_gather_feat(output2, ind2) #bxnx8
    pred2_tmp = pred2
    target2_tmp = target2
    mask = mask.unsqueeze(2).expand_as(pred1).float()

    b = pred1.size(0)
    m = pred1.size(1)
    n = pred2.size(1)
    pred2 = pred2.view(b,4*n,2)
    ctr_cro_ind = ctr_cro_ind.unsqueeze(2).expand(b,4*m,2)
    pred2 = pred2.gather(1,ctr_cro_ind).view(b,m,8) #bxmx8
    target2 = target2.view(b,4*n,2).gather(1,ctr_cro_ind).view(b,m,8)

    delta = (torch.abs(pred1-target1)+torch.abs(pred2-target2)) / (torch.abs(target1) + 1e-4)
    delta = delta * delta
    delta_mask = (~delta.gt(1.0))*1#1 - delta.gt(1.0)
    delta = delta*(delta_mask.float())+(1-delta_mask).float()
    delta = (-3.14)*delta
    weight = 1 - torch.exp(delta)

    loss1 = F.l1_loss(pred1 * mask * weight, target1 * mask * weight, size_average=False)
    loss2 = F.l1_loss(pred2 * mask * weight, target2 * mask * weight, size_average=False)
    loss1 = loss1 / (mask.sum() + 1e-4)
    loss2 = loss2 / (mask.sum() + 1e-4) 

    mask1 = (target2_tmp==0)
    mask_cro = mask_cro.unsqueeze(2).expand(b,n,8)
    MASK = (mask1==mask_cro).float() 
    loss3 = F.l1_loss(pred2_tmp * MASK, target2_tmp * MASK, size_average=False)
    loss3 = loss3 / (mask.sum() + 1e-4)

    return loss1, 0.5 * loss2 + 0.2 * loss3

class DistLoss(nn.Module):
  def __init__(self):
    super(DistLoss, self).__init__()
  
  def forward(self, h_pair_ind, v_pair_ind, logic, logi=None):
    
    # if logi is None:
    #   pred = _tranpose_and_gather_feat(output, ind)
    # else:
    
    pred = logi
    pred_pair = _make_pair_feat(pred)

    h_pred = _flatten_and_gather_feat(pred_pair, h_pair_ind)
    v_pred = _flatten_and_gather_feat(pred_pair, v_pair_ind)

    logic_pair = _make_pair_feat(logic)
    h_pair = _flatten_and_gather_feat(logic_pair, h_pair_ind)
    v_pair = _flatten_and_gather_feat(logic_pair, v_pair_ind)
    
    h_dist = rank_dist(h_pred, h_pair, h_pair_ind, rtype = 'horizontal').squeeze() 
    v_dist = rank_dist(v_pred, v_pair, v_pair_ind, rtype = 'vertical').squeeze()
  
    loss = torch.max(torch.zeros(h_dist.shape).cuda(), (1-h_dist)*(h_pair_ind!=0)).sum()/((h_pair_ind!=0).sum().to(torch.float32) +1e-5) + \
           torch.max(torch.zeros(v_dist.shape).cuda(), (1-v_dist)*(v_pair_ind!=0)).sum()/((v_pair_ind!=0).sum().to(torch.float32) +1e-5) 
    return loss
    
class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


def _axis_eval(output, mask, ind, target, logi=None, mode = None):
  if logi is None:
    pred = _tranpose_and_gather_feat(output, ind)
  else:
    pred = logi
  
  dev = pred - target
  dev = torch.abs(dev)
  
  one = torch.ones(dev.shape).cuda()
  zero = torch.zeros(dev.shape).cuda()
 
  true_vec = torch.where(dev < 0.5, one, zero)
  #true = torch.sum((true_vec * mask.unsqueeze(2)) == 1)
  true = torch.sum(true_vec.sum(axis=2)*mask == 4)
  total = (mask == 1).sum()
  true = true.to(torch.float32)
  total = total.to(torch.float32)

  acc = true/total

  if acc is None:
    acc = 0

  if mode == 'full':
    pred_int = process_logi(pred)

    pred_pair = _make_pair_feat(pred_int)
    target_pair = _make_pair_feat(target)
    mask_pair = _make_pair_feat(ind)

    #pred_int = pred_int.expand()

    mask_1 = mask_pair[:,:,:,0]
    mask_2 = mask_pair[:,:,:,1]

    at_vec_h = (target_pair[:,:,:,2]<=target_pair[:,:,:,6]) & (target_pair[:,:,:,6]<=target_pair[:,:,:,3]) & (mask_1 != 0) & (mask_2 != 0)
    at_vec_w = (target_pair[:,:,:,0]<=target_pair[:,:,:,4]) & (target_pair[:,:,:,4]<=target_pair[:,:,:,1]) & (mask_1 != 0) & (mask_2 != 0)

    ap_vec_h = (pred_pair[:,:,:,2]<=pred_pair[:,:,:,6]) & (pred_pair[:,:,:,6]<=pred_pair[:,:,:,3]) & (mask_1 != 0) & (mask_2 != 0)
    ap_vec_w = (pred_pair[:,:,:,0]<=pred_pair[:,:,:,4]) & (pred_pair[:,:,:,4]<=pred_pair[:,:,:,1]) & (mask_1 != 0) & (mask_2 != 0)

    tp_h = at_vec_h & ap_vec_h
    tp_w = at_vec_w & ap_vec_w

    ap = torch.sum(ap_vec_h) + torch.sum(ap_vec_w)     
    at = torch.sum(at_vec_h) + torch.sum(at_vec_w)   
    tp = torch.sum(tp_h) + torch.sum(tp_w)
    
    pre = tp/(ap + 1e-4)
    rec = tp/(at + 1e-4)

    return acc , pre, rec
  else:
    return acc

