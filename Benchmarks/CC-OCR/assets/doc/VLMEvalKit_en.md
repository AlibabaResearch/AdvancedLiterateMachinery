# Installation
We now support the use of the CC-OCR dataset in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). 
Before our pull request is officially merged, you can prepare the environment as following instruction:
```shell
git clone https://github.com/wulipc/VLMEvalKit.git
cd VLMEvalKit
git checkout CC-OCR  # our code is in the CC-OCR branch.
pip install -e .
````
For subsequent preparation work, please refer to the official documentation of VLMEvalKit [here](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/Quickstart.md).
We will promptly announce the latest progress of the pull request, so please stay tuned.

## Running Scripts

Once the environment is ready, execute the following script from the root directory of VLMEvalKit 
to perform inference and evaluation tasks in batch. The evaluation code in VLMEvalKit is exactly 
the same as that in this repository.

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

## Example Output
The evaluation results will be saved in `${SUB_OUTPUT_DIR}/summary.md`. For example, for the KIE subset,
the output is as follows:

| exp_name(f1_score) |   COLD_CELL |   COLD_SIBR |   CORD |   EPHOIE_SCUT |   POIE |   sroie2019_word |   summary |
|:-------------------|------------:|------------:|-------:|--------------:|-------:|-----------------:|----------:|
| QwenVLMax          |       81.01 |       72.46 |  69.33 |          71.2 |  60.85 |            76.37 |     71.87 |
