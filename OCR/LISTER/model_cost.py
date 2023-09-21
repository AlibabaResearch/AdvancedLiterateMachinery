# Copyright (2023) Alibaba Group and its affiliates

""" Measure the Params and FLOPs.
Usage: python model_cost.py -c=config/lister.yml
"""

import torch
import time
from thop import profile

from utils import get_configs, get_model

device = torch.device('cuda')

def count_params(model):
    cnt = 0
    for k, v in model.named_parameters():
        cnt += v.numel()
    print(f'Params: {cnt/1e6:.3f} M')
    return cnt

config_dict = get_configs(is_training=False)
config_dict['num_classes'] = 37
net = get_model(config_dict, device, training=False, blank_id=0)
net = net.cuda()

bs = 1
img_size = (32, 128)
x = torch.rand(bs, 3, *img_size).cuda()
mask = torch.ones(bs, *img_size).cuda()
mask[0, :, 100:] = 0
if config_dict['model_type'] in ['lister', 'rnn']:
    inputs = (x, mask, 12)
else:
    inputs = (x, mask)
flops, params = profile(net, inputs)
# print('flops: ', flops, 'params: ', params)
print('FLOPs = ' + str(flops/1e9) + 'G')
# print('Params = ' + str(params/1e6) + 'M')
count_params(net)

# forward time
mask.fill_(1)
if config_dict['model_type'] in ['lister', 'rnn']:
    inputs = (x, mask, 12)
else:
    inputs = (x, mask)
net.eval()

N = 2000
t0 = time.time()
for _ in range(N):
    with torch.no_grad():
        net(*inputs)
        torch.cuda.synchronize()
dt = time.time() - t0
avg_time = dt / N
print("avg_time:", avg_time*1000, "ms")
