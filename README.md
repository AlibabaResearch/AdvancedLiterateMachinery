# Advanced Literate Machinery

## Introduction

The ultimate goal of our research is to build a system that has high-level intelligence, i.e., possessing the abilities to read, think and create, so advanced that it could even surpass human intelligence one day in the future. We name this kind of systems **Advanced Literate Machinery (ALM)**.

This project is maintained by the **读光 OCR Team** (读光-Du Guang means “*Reading The Light*”) in the [Language Technology Lab, Alibaba DAMO Academy](https://damo.alibaba.com/labs/language-technology). 

![Logo](./resources/DuGuang.png)

Visit our [读光-Du Guang Portal](https://duguang.aliyun.com/) to experience online demos for OCR and Document Understanding.

## Recent Updates

**2023.4 Release**
  - [**GeoLayoutLM**](./DocumentUnderstanding/GeoLayoutLM/) (*GeoLayoutLM: Geometric Pre-training for Visual Information Extraction,* CVPR 2023. [paper](https://arxiv.org/abs/2304.10759)): We propose a multi-modal framework, named GeoLayoutLM, for Visual Information Extraction (VIE). In contrast to previous methods for document pre-training, which usually learns geometric representation in an implicit way, GeoLayoutLM **explicitly** models the geometric relations.

**2023.2 Release**
  - [**LORE-TSR**](./DocumentUnderstanding/LORE-TSR/) (*LORE: Logical Location Regression Network for Table Structure Recognition,* AAAI 2022. [paper](https://arxiv.org/abs/2303.03730)): We model Table Structure Recognition (TSR) as a logical location regression problem and propose a new algorithm called LORE, standing for LOgical location REgression network, which for the first time combines logical location regression together with spatial location regression of table cells.

**2022.9 Release**
  - [**MGP-STR**](./OCR/MGP-STR/) (*Multi-Granularity Prediction for Scene Text Recognition,* ECCV 2022. [paper](https://arxiv.org/abs/2209.03592)): Based on [ViT](https://arxiv.org/abs/2010.11929) and a tailored Adaptive Addressing and Aggregation module, we explore an implicit way for incorporating linguistic knowledge by introducing subword representations to facilitate **multi-granularity** prediction and fusion in scene text recognition.
  - [**LevOCR**](./OCR/LevOCR/) (*Levenshtein OCR,* ECCV 2022. [paper](https://arxiv.org/abs/2209.03594)): Inspired by [Levenshtein Transformer](https://arxiv.org/abs/1905.11006), we cast the problem of scene text recognition as an iterative sequence refinement process, which allows for **parallel decoding, dynamic length change and good interpretability**.
