# LORE-TSR

The official PyTorch implementation of LORE-TSR. LORE can perform table structure recognition (TSR) in the end-to-end way by modeling TSR as logical location regression. The model streamlines the TSR pipeline as a key-point based detector-like framework. LORE-TSR exhibits good efficiency and performance in the implemention, which could be useful for TSR models in the future.

### Install requirements


```
#installing LORE
conda create --name Lore 
conda activate Lore
pip3 install -r requirements.txt

#installing cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
python setup.py install --user
```

If you would like to using the DLA backbone, please run the following scripts:
```
#installing DCNv2
conda install Cython
chmod +x  *.sh
cd src/lib/models/network/DCNv2
./make.sh
```

### Dataset

The labels are supposed to be transformed into COCO format, here the WTW dataset and a subset of PubTabNet dataset are taken as examples.
Download images of WTW dataset from [WTW-Dataset](https://github.com/wangwen-whu/WTW-Dataset). It provide the original dataset along with tools for changing it into COCO format. The example of processed COCO-like label of WTW and a subset of PubTabNet are provided. The directory of dataset are organized as following:

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

### Pretrained Models 

Available model weights (using dla-34 backbone):


| :---: | :---: | :---: |
| LORE-TSR |[Pretrained on WTW]|[Pretrained on PubTabNet]|

The provided model on WTW wired table dataset is of 4-layer base and 4-layer stacking, which has a better trade-off between accuracy and efficiency. We are also working in progress to release a model for wireless table trained on Chinese tables.

### Run demo with pretrained model
1. Download pretrained model (for wired table on WTW or for wireless table on PubTabNet)
2. Add image files to test into `./input_imgs/`
3. Change the parameters such as model architecture and output directory
3. Run 

```
cd src
bash scripts/demo/demo.sh
```

Notice: 
LORE is incorporated with the parsing-and-grouping mechenism similar to Cycle-CenterNet. Setting `--wiz_rev` arguments to activate such process at inference stage. It provides accurate detection results on wired tables, but could slow the inference.

### Train
1. Organizing the dataset as mentioned before
2. Changing the parameters such as model architecture, experiment ID and dataset path, etc.
3. Run

```
cd src
bash scripts/train/train.sh
```
We modified the original model to stabilize converging and make it easier to change backbone, by removing the learning weight in Eq. 2 and gathering the feature of cell centers from a conv-head.

### Test
Taking the PubTabNet as an example:
1. Find the path to `model_best.pth` checkpoint file (usually in `./exp/*/` folder)
2. Run demo on the test dataset
```
cd src
bash scripts/demo/demo_ptn.sh
```
3. Evaluating the result of model (remeber to change the directory of model results)
```
bash eval.sh
```

## Acknowledgements
This implementation has been based on the repository [CenterNet](https://github.com/xingyizhou/CenterNet).

### Paper
* [AAAI 2023]
* [Arxiv]

## Citation
If you find this work useful, please cite:

<!-- ```
@InProceedings{Long_2021_ICCV,
	author = {Rujiao, Long and Wen, Wang and Nan, Xue and Feiyu, Gao and Zhibo, Yang and Yongpan, Wang and Gui-Song, Xia},
	title = {Parsing Table Structures in the Wild},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	month = {October},
	year = {2021}
}

``` -->

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