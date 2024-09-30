# Visual Text Generation in the Wild

## Paper
* [Arxiv](https://arxiv.org/abs/2407.14138)

## Text Region and Content Generator (TRCG)

### Install requirements
```
cd TRCG
conda create -n trcg python=3.10 -y
conda activate trcg
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# if trainging
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install xformers==0.0.22

# if inference
pip install shapely
pip install opencv-python
```

### Dataset
Download dataset from [modelscope](https://www.modelscope.cn/datasets/Kpillow/SceneVTG-Erase).

### Pretrain models
1. Download pretrained [LLaVA-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b)
2. Download pretrained [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)
3. Download TRCG models from [modelscope](https://www.modelscope.cn/models/Kpillow/SceneVTG).
ckpts/trcg/* (need to change model path in ```config.json``` and ```adapter_config.json```)

### Training
change the dataset path and pretrained LLaVA-v1.5 to yours in scripts/finetune_flash_attn.sh or scripts/finetune_xformers.sh
```
# with flash-attn
sh scripts/finetune_flash_attn.sh
# with xformers
sh scripts/finetune_xformers.sh
```

### Inference
```Shell
CUDA_VISIBLE_DEVICES=0 python inference.py
```

## Acknowledge
Thanks to [LLaVA](https://github.com/haotian-liu/LLaVA), our code is mainly based on it.

## Citation
If you find this work useful, please cite:

```
@misc{zhu2024visualtextgenerationwild,
      title={Visual Text Generation in the Wild}, 
      author={Yuanzhi Zhu and Jiawei Liu and Feiyu Gao and Wenyu Liu and Xinggang Wang and Peng Wang and Fei Huang and Cong Yao and Zhibo Yang},
      year={2024},
      eprint={2407.14138},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14138}, 
}
```

## *License*

SceneVTG is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
SceneVTG is an algorithm for visual text generation and the code and models herein created by the authors from Alibaba can only be used for research purpose.
Copyright (C) 1999-2022 Alibaba Group Holding Ltd. 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
