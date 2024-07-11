# Platypus: A Generalized Specialist Model for Reading Text in Various Forms

The official PyTorch implementation of Platypus (ECCV 2024).

Platypus introduces a novel approach to text reading from images, addressing limitations of both specialist and generalist models. Built on a simple vision encoder and a transformer-based autoregressive decoder, Platypus leverages a single unified architecture to effectively recognize text in various forms, maintaining high accuracy and efficiency. The authors also introduce a newly curated dataset named "Worms," which combines and partially re-labels previous datasets to support the model's development and evaluation. Experiments on standard benchmarks demonstrate Platypus's superior performance over existing specialist and generalist approaches, and the authors plan to publicly release both the Platypus model and the Worms dataset to facilitate further research and application in advanced text reading tasks. <br>


### Paper
* [Arxiv]



### Dataset

Download lmdb dataset from [modelscope](https://modelscope.cn/datasets/yuekun/Worms).



## *License*

Platypus is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
Platypus is an algorithm for text reading and the code and models herein created by the authors from Alibaba can only be used for research purpose.
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