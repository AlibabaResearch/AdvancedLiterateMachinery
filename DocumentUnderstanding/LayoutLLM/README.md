# LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding

This is the repo for the LayoutLLM (CVPR 2024), which is an LLM/MLLM based method for document understanding. The repo contains:

- The Layout Instruction tuning data
    - Layout-aware pre-training data
        - Document Dense Caption data
        - Code for the Text and Layout Reconstruction data generation
    - Layout-aware supervised fine-tuning (SFT) data (with the **LayoutCoT**)

- The evaluation data
    - QA for VIE

## Data

**Description:** The layout instruction tuning data (with images) of LayoutLLM.

**Data Access**: [Download Dataset](https://modelscope.cn/datasets/iic/Layout-Instruction-Data/summary)


## Paper
* [CVPR 2024]
* [Arxiv](https://arxiv.org/abs/2404.05225)

## Citation

If you find this repository useful, please consider citing our work:
```
@inproceedings{luo2024layoutllm,
    title={LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding},
    author={Chuwei Luo and Yufan Shen and Zhaoqing Zhu and Qi Zheng and Zhi Yu and Cong Yao},
    year={2024},
    booktitle = {CVPR},
}
```

## License

The training data of the LayoutLLM is released under the terms of the [Apache License, Version 2.0](LICENSE.md).

```
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