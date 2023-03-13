# LORE-TSR

The official PyTorch implementation of LORE-TSR. LORE can perform table structure recognition (TSR) in the end-to-end way by modeling TSR as logical location regression. The model streamlines the TSR pipeline as a key-point based detector-like framework. LORE-TSR exhibits good efficiency and performance in the implemention, which could be useful for TSR models in the future.

## Installation
### Installing requirements

```
conda create --name Lore python=3.7
conda activate Lore
pip install -r requirements.txt
```

### Installing cocoapi
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
pip install Cython
make
python setup.py install --user
```

### Installing DCNv2 from Scratch
If you would like to using the DLA backbone, an environment based on CUDA 10.1 is strongly recommended.

Firstly, CUDA 10.1 (FROM https://developer.nvidia.com/cuda-toolkit-archive) should be installed. Take an example of Linux-x86_64-Ubuntu-18.04:
```
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run

#Setting env variables
export CUDA_HOME='your cuda-10.1 path'
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
### Installing Torch
```
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Installing DCNv2
```
pip install Cython
chmod +x  *.sh
cd src/lib/models/network/DCNv2
./make.sh
```

## Dataset

The labels are supposed to be transformed into COCO format, here the WTW dataset and a subset of PubTabNet dataset are taken as examples.

Download images of WTW dataset from [WTW-Dataset](https://github.com/wangwen-whu/WTW-Dataset). It provide the original dataset along with tools for changing it into COCO format. The example of processed COCO-like label of [WTW]() and a subset of [PubTabNet]() are provided. The directory of dataset are organized as following:

```
data
├── WTW
│   ├── images
│   └── json
│       ├──train.json
│       └──test.json
│
└── PTN
    ├── images
    └── json
        ├── train.json
        └── test.json
    
```

## Pretrained Models 

Available model weights (using dla-34 backbone):

| Model Arc | Image Size | Checkpoint | 
| :---: | :---: | :---: |
| 4+4 | 1024 |[Trained on WTW]|
| 3+3 | 512 | [Trained on PubTabNet]|



The provided model on WTW wired table dataset is of 4-layer base and 4-layer stacking, which has a better trade-off between accuracy and efficiency. We are also working in progress to release a model for wireless table trained on Chinese tables.

## Run demo with pretrained model
1. Download pretrained model (for wired table on WTW or for wireless table on PubTabNet)
2. Add image files to test into `./input_imgs/`
3. Change the parameters such as model architecture and output directory
3. Run:

```
cd src
bash scripts/demo/demo.sh
```

Notice: 
LORE is incorporated with the parsing-and-grouping mechenism similar to Cycle-CenterNet for wired tables. Setting `--wiz_rev` arguments to activate such process at inference stage. It provides accurate detection results on wired tables, but could slow the inference.

## Train
1. Organizing the dataset as mentioned before
2. Changing the parameters such as model architecture, experiment ID and dataset path, etc.
3. Run:

```
cd src
bash scripts/train/train.sh
```
We modified the original model to stabilize converging and make it easier to change backbone, by removing the learning weight in Eq. 2 and gathering the feature of cell centers from a conv-head.

### Test
Taking the PubTabNet as an example:
1. Find the path to `model_best.pth` checkpoint file (usually in `./exp/*/` folder)
2. Run demo on the test dataset:
```
cd src
bash scripts/demo/demo_ptn.sh
```
3. Evaluating the result of model (remeber to change the directory of model results)
```
bash eval.sh
```

## Acknowledgements
This implementation has been based on the repository [CenterNet](https://github.com/xingyizhou/CenterNet) and [DCNv2].

### Paper
* [AAAI 2023]
* [Arxiv](https://arxiv.org/abs/2303.03730)

## Citation
If you find this work useful, please cite:

LORE:
```
@article{Xing_2023_Lore,
  author={Hangdi, Xing and Feiyu Gao and Rujiao Long and Jiajun Bu and Qi Zheng and Liangcheng Li and Cong Yao and Zhi Yu},
  title={LORE: Logical Location Regression Network for Table Structure Recognition},
  journal={arXiv preprint arXiv:2303.03730},
  year={2023}
}
```
Cycle-CenterNet:
```
@InProceedings{Long_2021_ICCV,
	author = {Rujiao, Long and Wen, Wang and Nan, Xue and Feiyu, Gao and Zhibo, Yang and Yongpan, Wang and Gui-Song, Xia},
	title = {Parsing Table Structures in the Wild},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	month = {October},
	year = {2021}
}

```

## *License*

LORE-TSR is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
LORE-TSR is an algorithm for table structure recognition and the code and models herein created by the authors from Alibaba can only be used for research purpose.
Copyright (C) 1999-2023 Alibaba Group Holding Ltd. 

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