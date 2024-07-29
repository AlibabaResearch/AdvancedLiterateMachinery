# Visual Text Generation in the Wild

The official PyTorch implementation of SceneVTG (ECCV 2024).

The authors propose a visual text generator (termed SceneVTG), which can produce high-quality text images in the wild. Following a two-stage paradigm, SceneVTG leverages a Multimodal Large Language Model to recommend reasonable text regions and contents across multiple scales and levels, which are used by a conditional diffusion model as conditions to generate text images. To train SceneVTG, the authors also contribute a new dataset SceneVTG-Erase, which contains 110K scene text images and their text-erased backgrounds with detailed OCR annotations. Extensive experiments verified the fidelity, reasonability, and utility of our proposed SceneVTG, and the authors plan to publicly release both the SceneVTG model and the SceneVTG-Erase dataset to facilitate further research and application in advanced visual text generation tasks. <br>


### Paper
* [Arxiv](https://arxiv.org/abs/2407.14138)



### Dataset

Download dataset from [modelscope](https://www.modelscope.cn/datasets/Kpillow/SceneVTG-Erase).



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