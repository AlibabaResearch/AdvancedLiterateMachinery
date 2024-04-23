# Advanced Literate Machinery

## Introduction

The ultimate goal of our research is to build a system that has high-level intelligence, i.e., possessing the abilities to ***read, think and create***, so advanced that it could even surpass human intelligence one day in the future. We name this kind of systems **Advanced Literate Machinery (ALM)**.

To start with, we currently focus on teaching machines to ***read*** from images and documents. In years to come, we will explore the possibilities of endowing machines with the intellectual capabilities of ***thinking and creating***, catching up with and surpassing [GPT-4](https://openai.com/research/gpt-4) and [GPT-4V](https://openai.com/research/gpt-4v-system-card).

This project is maintained by the **读光 OCR Team** (读光-Du Guang means “*Reading The Light*”) in the [Tongyi Lab, Alibaba Group](https://tongyi.aliyun.com/).

![Logo](./resources/DuGuang.png)

Visit our [读光-Du Guang Portal](https://duguang.aliyun.com/) and [DocMaster](https://www.modelscope.cn/studios/damo/DocMaster/summary) to experience online demos for OCR and Document Understanding.

## Recent Updates

**2024.4 Release**
  - [**OmniParser**](./OCR/OmniParser/) (*OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition,* CVPR 2024. [paper](https://arxiv.org/abs/2403.19128)): We propose a universal model for parsing visually-situated text across diverse scenarios, called OmniParser, which can simultaneously handle three typical visually-situated text parsing tasks: text spotting, key information extraction, and table recognition. In OmniParser, all tasks share the **unified encoder-decoder architecture**, the unified objective: **point-conditioned text generation**, and the unified input & output representation: **prompt & structured sequences**.

**2024.3 Release**
  - [**GEM**](./DocumentUnderstanding/GEM/) (*GEM: Gestalt Enhanced Markup Language Model for Web Understanding via Render Tree,* EMNLP 2023. [paper](https://aclanthology.org/2023.emnlp-main.375.pdf)): Web pages serve as crucial carriers for humans to acquire and perceive information. Inspired by the Gestalt psychological theory, we propose an innovative Gestalt Enhanced Markup Language Model (GEM for short) for **hosting heterogeneous visual information from render trees of web pages**, leading to excellent performances on tasks such as web question answering and web information extraction.

**2023.9 Release**
  - [**DocXChain**](./Applications/DocXChain/) (*DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond,* arXiv 2023. [report](https://arxiv.org/abs/2310.12430)): To **promote the level of digitization and structurization for documents**, we develop and release an open-source toolchain, called DocXChain, for precise and detailed document parsing. Currently, basic capabilities, including text detection, text recognition, table structure recognition, and layout analysis, are provided. Also, typical pipelines, i.e., general text reading, table parsing, and document structurization, are built to support more complicated applications related to documents. Most of the algorithmic models are from [ModelScope](https://github.com/modelscope/modelscope). Formula recognition (using models from [RapidLatexOCR](https://github.com/RapidAI/RapidLatexOCR)) and whole PDF conversion (PDF to JSON format) are now supported.
  - [**LISTER**](./OCR/LISTER/) (*LISTER: Neighbor Decoding for Length-Insensitive Scene Text Recognition,* ICCV 2023. [paper](https://arxiv.org/abs/2308.12774v1)): We propose a method called Length-Insensitive Scene TExt Recognizer (LISTER), which remedies the limitation regarding the **robustness to various text lengths**. Specifically, a Neighbor Decoder is proposed to obtain accurate character attention maps with the assistance of a novel neighbor matrix regardless of the text lengths. Besides, a Feature Enhancement Module is devised to model the long-range dependency with low computation cost, which is able to perform iterations with the neighbor decoder to enhance the feature map progressively..
  - [**VGT**](./DocumentUnderstanding/VGT/) (*Vision Grid Transformer for Document Layout Analysis,* ICCV 2023. [paper](https://arxiv.org/abs/2308.14978)): To **fully leverage multi-modal information and exploit pre-training techniques to learn better representation** for document layout analysis (DLA), we present VGT, a two-stream Vision Grid Transformer, in which Grid Transformer (GiT) is proposed and pre-trained for 2D token-level and segment-level semantic understanding. In addition, a new benchmark for assessing document layout analysis algorithms, called [D^4LA](https://modelscope.cn/datasets/damo/D4LA/summary), is curated and released.
  - [**VLPT-STD**](./OCR/VLPT-STD/) (*Vision-Language Pre-Training for Boosting Scene Text Detectors,* CVPR 2022. [paper](https://arxiv.org/abs/2204.13867)): We adapt **vision-language joint learning for scene text detection**, a task that intrinsically involves cross-modal interaction between the two modalities: vision and language. The pre-trained model is able to produce more informative representations with richer semantics, which could readily benefit existing scene text detectors (such as EAST and DB) in the down-stream text detection task.

**2023.6 Release**
  - [**LiteWeightOCR**](./OCR/LiteWeightOCR/) (*Building A Mobile Text Recognizer via Truncated SVD-based Knowledge Distillation-Guided NAS,* BMVC 2023. [paper](https://papers.bmvc2023.org/0375.pdf)): To make OCR models **deployable on mobile devices while keeping high accuracy**, we propose a light-weight text recognizer that integrates Truncated Singular Value Decomposition (TSVD)-based Knowledge Distillation (KD) into the Neural Architecture Search (NAS) process.

**2023.4 Release**
  - [**GeoLayoutLM**](./DocumentUnderstanding/GeoLayoutLM/) (*GeoLayoutLM: Geometric Pre-training for Visual Information Extraction,* CVPR 2023. [paper](https://arxiv.org/abs/2304.10759)): We propose a multi-modal framework, named GeoLayoutLM, for Visual Information Extraction (VIE). In contrast to previous methods for document pre-training, which usually learn geometric representation in an implicit way, GeoLayoutLM **explicitly models the geometric relations of entities in documents**.

**2023.2 Release**
  - [**LORE-TSR**](./DocumentUnderstanding/LORE-TSR/) (*LORE: Logical Location Regression Network for Table Structure Recognition,* AAAI 2022. [paper](https://arxiv.org/abs/2303.03730)): We model Table Structure Recognition (TSR) as a logical location regression problem and propose a new algorithm called LORE, standing for LOgical location REgression network, which for the first time **combines logical location regression together with spatial location regression** of table cells.

**2022.9 Release**
  - [**MGP-STR**](./OCR/MGP-STR/) (*Multi-Granularity Prediction for Scene Text Recognition,* ECCV 2022. [paper](https://arxiv.org/abs/2209.03592)): Based on [ViT](https://arxiv.org/abs/2010.11929) and a tailored Adaptive Addressing and Aggregation module, we explore an implicit way for incorporating linguistic knowledge by introducing subword representations to facilitate **multi-granularity** prediction and fusion in scene text recognition.
  - [**LevOCR**](./OCR/LevOCR/) (*Levenshtein OCR,* ECCV 2022. [paper](https://arxiv.org/abs/2209.03594)): Inspired by [Levenshtein Transformer](https://arxiv.org/abs/1905.11006), we cast the problem of scene text recognition as an iterative sequence refinement process, which allows for **parallel decoding, dynamic length change and good interpretability**.
