# 使用说明
我们已经支持在 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 中使用 CC-OCR 数据集。在我们的 PR 正式合入官方库前，你可以使用以下代码进行运行环境准备：
```shell
git clone https://github.com/wulipc/VLMEvalKit.git
cd VLMEvalKit
git checkout CC-OCR  # our code is in the CC-OCR branch.
pip install -e .
````
后续准备工作请参考 VLMEvalKit [官方文档](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/Quickstart.md)；
我们将及时公布最新 PR 进度，请留意。


## 运行脚本
请准备好运行环境后，在 `VLMEvalKit` 的根目录下运行一下脚本，会批量完成推理和评测任务；`VLMEvalKit` 中评测代码与本仓库完全一致。
```shell
MODEL_NAME="QwenVLMax"
OUTPUT_DIR="/your/path/to/output_dir"

SUB_OUTPUT_DIR=${OUTPUT_DIR}/multi_scene_ocr
python run.py --data CCOCR_MultiSceneOcr_Cord CCOCR_MultiSceneOcr_Funsd CCOCR_MultiSceneOcr_Iam CCOCR_MultiSceneOcr_ZhDoc CCOCR_MultiSceneOcr_ZhHandwriting CCOCR_MultiSceneOcr_Hieragent CCOCR_MultiSceneOcr_Ic15 CCOCR_MultiSceneOcr_Inversetext CCOCR_MultiSceneOcr_Totaltext CCOCR_MultiSceneOcr_ZhScene CCOCR_MultiSceneOcr_UgcLaion CCOCR_MultiSceneOcr_ZhDense CCOCR_MultiSceneOcr_ZhVertical --model ${MODEL_NAME} --work-dir ${SUB_OUTPUT_DIR} --verbose
python vlmeval/dataset/utils/ccocr_evaluator/common.py ${SUB_OUTPUT_DIR}

SUB_OUTPUT_DIR=${OUTPUT_DIR}/multi_lan_ocr
python run.py --data CCOCR_MultiLanOcr_Arabic CCOCR_MultiLanOcr_French CCOCR_MultiLanOcr_German CCOCR_MultiLanOcr_Italian CCOCR_MultiLanOcr_Japanese CCOCR_MultiLanOcr_Korean CCOCR_MultiLanOcr_Portuguese CCOCR_MultiLanOcr_Russian CCOCR_MultiLanOcr_Spanish CCOCR_MultiLanOcr_Vietnamese --model ${MODEL_NAME} --work-dir ${SUB_OUTPUT_DIR} --verbose
python vlmeval/dataset/utils/ccocr_evaluator/common.py ${SUB_OUTPUT_DIR}

SUB_OUTPUT_DIR=${OUTPUT_DIR}/doc_parsing
python run.py --data CCOCR_DocParsing_DocPhotoChn CCOCR_DocParsing_DocPhotoEng CCOCR_DocParsing_DocScanChn CCOCR_DocParsing_DocScanEng CCOCR_DocParsing_TablePhotoChn CCOCR_DocParsing_TablePhotoEng CCOCR_DocParsing_TableScanChn CCOCR_DocParsing_TableScanEng CCOCR_DocParsing_MolecularHandwriting CCOCR_DocParsing_FormulaHandwriting --model ${MODEL_NAME} --work-dir ${SUB_OUTPUT_DIR} --verbose
python vlmeval/dataset/utils/ccocr_evaluator/common.py ${SUB_OUTPUT_DIR}

SUB_OUTPUT_DIR=${OUTPUT_DIR}/kie
python run.py --data CCOCR_Kie_Sroie2019Word CCOCR_Kie_Cord CCOCR_Kie_EphoieScut CCOCR_Kie_Poie CCOCR_Kie_ColdSibr CCOCR_Kie_ColdCell --model ${MODEL_NAME} --work-dir ${SUB_OUTPUT_DIR} --verbose
python vlmeval/dataset/utils/ccocr_evaluator/common.py ${SUB_OUTPUT_DIR}
```

## 示例输出
* 评测结果会保存在 `${SUB_OUTPUT_DIR}/summary.md` 中，以 `KIE`子集为例，输出内容如下：

| exp_name(f1_score) |   COLD_CELL |   COLD_SIBR |   CORD |   EPHOIE_SCUT |   POIE |   sroie2019_word |   summary |
|:-------------------|------------:|------------:|-------:|--------------:|-------:|-----------------:|----------:|
| QwenVLMax          |       81.01 |       72.46 |  69.33 |          71.2 |  60.85 |            76.37 |     71.87 |
