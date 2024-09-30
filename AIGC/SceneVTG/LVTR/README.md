# Visual Text Generation in the Wild

## Paper
* [Arxiv](https://arxiv.org/abs/2407.14138)

## Local Visual Text Renderer (LVTR)

### Install requirements
```
conda env create -f environment.yml
conda activate lvtr
```

### Dataset
Download dataset from [modelscope](https://www.modelscope.cn/datasets/Kpillow/SceneVTG-Erase).

### Pretrain models
Download models from [modelscope](https://www.modelscope.cn/models/Kpillow/SceneVTG).
```
ckpts/lvtr.pth  # pretrain model of LVTR
ckpts/recognizer.pth  # pretrain model of recognizer
```

### Training
1. Training recognizer.
change the dataset path to yours in configs/cfgs_recognizer.py
```
sh train_recognizer.sh
```

2. Training LVTR.
change the dataset path to yours in configs/cfgs_lvtr.py
```
sh train_lvtr.sh
```

### Local Image Generation (Inference)
change "testmode" to "True" in configs/cfgs_lvtr.py
```
sh train_lvtr.sh
```

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