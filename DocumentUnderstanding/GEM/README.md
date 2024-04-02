# GEM: Gestalt Enhanced Markup Language Model for Web Understanding via Render Tree

This repository hosts the pre-trained weights and fine-tuning source code for the paper "[GEM: Gestalt Enhanced Markup Language Model for Web Understanding via Render Tree](https://aclanthology.org/2023.emnlp-main.375/)," which has been accepted as a main conference paper at EMNLP 2023.

## Pre-trained Models

We pre-trained GEM on approximately 2 million training samples from 100k renderable web pages. The pre-training is done on 8 Nvidia-V100 GPUs for 300K steps. The pre-trained weights of GEM are available for download [here](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/tag/v1.7.0-gem-model-release).

## Fine-tuning

### The Basic Codes

Please note that the fine-tuning code for our method is mostly adopted from [MarkupLM](https://arxiv.org/abs/2110.08518). To respect the copyright of MarkupLM, we upload only the files that are modified to align with our work. Therefore, if you plan to run the code, you should follow [this link](https://github.com/microsoft/unilm/tree/master/markuplm) to the original MarkupLM repository, where you can build the virtual environment and prepare the data for fine-tuning.


### The Modified Files
Please replace the `examples/fine_tuning/run_swde/run.py` and `examples/fine_tuning/run_websrc/run.py` files in the original repository with the ones provided in our repository.

### Run Script
#### SWDE
Example using the 'nbaplayer' vertical:

```bash
cd ./examples/fine_tuning/run_swde
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
        --root_dir /Path/To/Processed_SWDE \
        --vertical nbaplayer \
        --n_seed 5 \
        --n_pages 2000 \
        --prev_nodes_into_account 8 \
        --model_name_or_path /Path/To/GEM \
        --output_dir /Your/Output/Path \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 48 \
        --per_gpu_eval_batch_size 32 \
        --num_train_epochs 10 \
        --learning_rate 2e-5 \
        --save_steps 1000000 \
        --overwrite_output_dir
```

#### WebSRC

```bash
cd ./examples/fine_tuning/run_websrc
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_opens.py \
        --train_file /Path/To/WebSRC/websrc1.0_train_.json \
        --predict_file /Path/To/WebSRC/websrc1.0_dev_.json \
        --root_dir /Path/To/WebSRC \
        --model_name_or_path /Path/To/GEM \
        --output_dir /Your/Output/Path \
        --do_train \
        --do_eval \
        --evaluate_on_saving \
        --per_gpu_train_batch_size 8 \
        --num_train_epochs 10 \
        --overwrite_output_dir \
        --learning_rate 1e-5
```

## Citation
If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{shao-etal-2023-gem,
    title = "{GEM}: Gestalt Enhanced Markup Language Model for Web Understanding via Render Tree",
    author = "Shao, Zirui and Gao, Feiyu and Qi, Zhongda and Xing, Hangdi and Bu, Jiajun and Yu, Zhi and Zheng, Qi and Liu, Xiaozhong",
    editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.375",
    doi = "10.18653/v1/2023.emnlp-main.375",
    pages = "6132--6145",
}
```

