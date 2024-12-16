# WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation

Welcome to the official repository of WebRPG (ECCV2024).

## Introduction
We introduce WebRPG, a novel task that focuses on **automating the generation of visual presentations** for web pages based on HTML code. In the absence of a benchmark, we created a new dataset via an **automated pipeline**. Our proposed models, built on **VAE architecture** and **custom HTML embeddings**, efficiently manage numerous web elements and rendering parameters. Comprehensive experiments, including customized quantitative evaluations, demonstrate the effectiveness of WebRPG model in generating web presentations. For more details, please refer to our paper:

- [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73242-3_4)
- [arXiv](https://arxiv.org/pdf/2407.15502)


## Installation

### Environment

All dependencies are listed in `requirements.txt`. Ensure you install a proper version of PyTorch that matches your CUDA version before proceeding.

```bash
conda create -n webrpg python=3.11
conda activate webrpg
pip install -r requirements.txt
```

### Model Checkpoints

Download the necessary model checkpoints from [ModelScope](https://modelscope.cn/models/iic/WebRPG). After downloading, place them in the root directory of the project, maintaining the directory structure as provided in ModelScope.

### Additional Requirement

This repository also requires the code from MarkupLM. To set it up:

Download the [MarkupLM](https://github.com/microsoft/unilm/tree/master/markuplm/markuplmft/models/markuplm) folder and place it in the root directory of this project.

Additionally, download the pre-trained MarkupLM-Large weights from [Huggingface](https://huggingface.co/microsoft/markuplm-large) and place them in `/Path/To/MarkupLM/Large`.

## Prepare data

1. **Download the Dataset**: Obtain the WebRPG dataset from [ModelScope](https://modelscope.cn/datasets/iic/WebRPG_Dataset) and extract it to `/Path/To/RawData`.
2. **Preprocess the Data**: Follow the steps below to preprocess the dataset.

```bash
# Step 1

python create_webrpg_data_1.py \
    --output_dir /Path/To/Processed/Step1/Train \
    --root_dir /Path/To/RawData \
    --file_json /Path/To/RawData/train.json \
    --start 0 \
    --end 37

python create_webrpg_data_1.py \
    --output_dir /Path/To/Processed/Step1/Test \
    --root_dir /Path/To/RawData \
    --file_json /Path/To/RawData/test.json \
    --start 0 \
    --end 10

# Step 2

python create_webrpg_data_2.py \
    --output_dir /Path/To/Processed/Step2/Train \
    --input_dir /Path/To/Processed/Step1/Train \
    --markuplm_model_name_or_path /Path/To/MarkupLM/Large \
    --start 0 \
    --end 37

python create_webrpg_data_2.py \
    --output_dir /Path/To/Processed/Step2/Test \
    --input_dir /Path/To/Processed/Step1/Test \
    --markuplm_model_name_or_path /Path/To/MarkupLM/Large \
    --start 0 \
    --end 10

# Step 3

python create_webrpg_data_split.py \
    --output_dir /Path/To/Processed/Split/Train \
    --input_dir /Path/To/Processed/Step2/Train \
    --start 0 \
    --end 37

python create_webrpg_data_split.py \
    --output_dir /Path/To/Processed/Split/Test \
    --input_dir /Path/To/Processed/Step2/Test \
    --start 0 \
    --end 10
```


## Training

Before training, generate the training data cache file:

```bash
python get_cache_file.py \
  --cache_file /Path/To/Cache/train.txt \
  --target_folder /Path/To/Processed/Split/Train
```

Once the cache file is ready, start training. Note that this repository currently includes the implementation of `WebRPG-AR` due to its superior performance:

```bash
python trainer_ar.py \
  --output_dir /Path/To/Output \
  --cache_path /Path/To/Cache/train.txt \
  --vae_pretrained_path ./pretrained_vae/pretrained_vae.pt \
  --pretrained_markuplm_path /Path/To/MarkupLM/Large \
  --logging_step 10 \
  --lr_scheduler_type "constant" \
  --learning_rate 1.2e-4 \
  --do_train \
  --max_steps 1000000 \
  --save_steps 5000 \
  --per_device_train_batch_size 75 \
  --dataloader_num_workers 8 \
  --label_names "param" \
  --overwrite_output_dir
```

## Evaluation

Generate the evaluation data cache file:

```bash
python get_cache_file.py \
  --cache_file /Path/To/Cache/test.txt \
  --target_folder /Path/To/Processed/Split/Test
```

Run the evaluation using all available metrics, including Fréchet Inception Distance (FID), Element-wise Intersection over Union (Ele. IoU), and Style Consistency Score (Sc Score):

```bash
sh run_tests.sh \
--eval_output_dir=/Path/To/Eval/Output \
--checkpoint_dir=/Path/To/Output
```


## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{shao2025webrpg,
  title={WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation},
  author={Shao, Zirui and Gao, Feiyu and Xing, Hangdi and Zhu, Zepeng and Yu, Zhi and Bu, Jiajun and Zheng, Qi and Yao, Cong},
  booktitle={European Conference on Computer Vision},
  pages={56--74},
  year={2025},
  organization={Springer}
}
```

### Contact Information

For any questions related to WebRPG, please feel free to contact the authors of the paper.

## *License*

WebRPG is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
This code created by the authors from Alibaba can only be used for research purposes.
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