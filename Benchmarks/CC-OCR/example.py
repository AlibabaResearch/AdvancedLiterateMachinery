import os
import sys
import json
import time
from tqdm import tqdm
from functools import partial
from retry import retry

import dashscope
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor
from urllib.parse import urlparse, unquote

# local import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.main import evaluate_and_summary


def run_tasks_parallel(local_file_list, worker_func, max_workers=32, with_tqdm=False, multi_process=True, desc=None):
    """
    """
    uploaded_list = []
    PoolExecutor = ProcessPoolExecutor if multi_process else ThreadPoolExecutor
    with PoolExecutor(max_workers=max_workers) as executor:
        if with_tqdm:
            for remote_path in tqdm(executor.map(worker_func, local_file_list), total=len(local_file_list), desc=desc):
                uploaded_list.append(remote_path)
        else:
            for remote_path in executor.map(worker_func, local_file_list):
                uploaded_list.append(remote_path)
    return uploaded_list


def get_file_list_in_dir(base_dir, is_recursive=False):
    """
    """
    files = []
    with os.scandir(base_dir) as it:
        for entry in tqdm(it):
            if is_recursive:
                if entry.is_file():
                    files.append(entry.path)
                else:
                    files.extend(get_file_list_in_dir(entry.path, is_recursive=is_recursive))
            else:
                files.append(entry.path)
    return files


def qwen_vl_func(image_path, question, is_max=True, max_tokens=2000):
    """
    """
    if os.path.exists(image_path):
        image_path = "file://" + os.path.abspath(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": question}
            ]
        }
    ]

    if max_tokens > 2000:
        print("--> warning: max_tokens of qwen_vl too large, got: {}, use 2000 as default.".format(max_tokens))

    api_key = os.environ.get('DASHBOARD_API_KEY', None)
    if api_key is None:
        raise Exception("DASHBOARD_API_KEY not set: please export DASHBOARD_API_KEY in your environment variable.")

    model_name = 'qwen-vl-max' if is_max else 'qwen-vl-plus'
    response = dashscope.MultiModalConversation.call(
        model=model_name,
        max_tokens=min(max_tokens, 2000),
        api_key=api_key,
        messages=messages)

    response_text = json.loads(str(response))
    if response_text.get("status_code") != 200:
        raise ValueError(response_text)
    return response_text


@retry(tries=3, delay=5)
def chat_func_api(image_path, question, func_name="default", max_tokens=8192):
    """
    """
    if func_name == "qwen_vl_plus":
        return qwen_vl_func(image_path, question, is_max=False, max_tokens=min(max_tokens, 2000))
    elif func_name == "qwen_vl_max":
        return qwen_vl_func(image_path, question, is_max=True, max_tokens=min(max_tokens, 2000))
    # TODO: add your own func here
    # pick_response_text func for your model also should be implemented in evaluation/evaluator/common.py:9
    else:
        raise ValueError("unsupported api name")


def run_single_inference(tuple_input, func_name="qwen_vl_max"):
    """
    """
    image_path, question, output_result_path = tuple_input
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)

    try:
        rsp_json = chat_func_api(image_path, question, func_name=func_name)
        if rsp_json is None:
            return None

        res = {
            "image": image_path,
            "question": question,
            "model_name": func_name,
            "response": rsp_json,
            "time": time.time()
        }
        with open(output_result_path, "w") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("--> error:  skip: {}, Q: {}, msg: {}".format(image_path, question, e))
        return None
    return output_result_path


def process_batch_data(dataset_base_dir, output_dir, dataset_name, func_name="qwen_vl_max", max_workers=1, is_resume=True):
    """
    """
    sample_list = []
    qa_jsonl_path = os.path.join(dataset_base_dir, "qa.jsonl")
    with open(qa_jsonl_path, "r") as f:
        for line in f:
            json_data = json.loads(line.strip())
            image_url = json_data["url"]
            question = json_data["prompt"]

            file_name = os.path.basename(unquote(urlparse(image_url).path)) + ".json"
            output_result_path = os.path.join(output_dir, file_name)
            if is_resume and os.path.exists(output_result_path):
                continue

            image_local_path = os.path.join(dataset_base_dir, image_url)
            sample_list.append((image_local_path, question, output_result_path))

    inference_func = partial(run_single_inference, func_name=func_name)
    print("--> info: load: {} from: {}".format(len(sample_list), qa_jsonl_path))
    res_list = run_tasks_parallel(sample_list, inference_func, max_workers=max_workers, with_tqdm=True,
                                  multi_process=True, desc=dataset_name)
    return res_list


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python {} mode_name index_path output_dir".format(__file__))
        exit(-1)
    else:
        print('--> info: {}'.format(sys.argv))
        func_name, index_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    valid_list = ["qwen_vl_max",  "qwen_vl_plus"]
    if func_name not in valid_list:
        raise ValueError("unsupported api name: {}, valid_list: {}".format(func_name, valid_list))

    is_resume = True
    max_workers = 4
    print("--> info: max_workers: {}, is_resume: {}".format(max_workers, is_resume))
    base_dir = os.path.dirname(os.path.dirname(index_path))

    with open(index_path, "r") as f:
        data_info_list = json.load(f)

    data_info_valid = {}
    for data_info in data_info_list:
        data_name = data_info["dataset"]
        if data_info.get("release", True):
            data_info_valid[data_name] = data_info

    for idx, (data_name, data_info) in enumerate(data_info_valid.items()):
        print("--> [{}/{}] info: processing: {}".format(idx, len(data_info_valid), data_name))
        sub_output_dir = os.path.join(output_dir, func_name, data_name)
        os.makedirs(sub_output_dir, exist_ok=True)

        dataset_base_dir = os.path.abspath(os.path.join(base_dir, data_info["base_dir"]))
        res_list = process_batch_data(dataset_base_dir, sub_output_dir, dataset_name=data_name,
                                      func_name=func_name, max_workers=max_workers, is_resume=is_resume)
        success_list = [1 if x is not None else 0 for x in res_list]
        print("--> [{}]: success ratio: {}, total: {}".format(data_name,
                sum(success_list) / (len(success_list) + 1e-6), len(success_list)))

    exp_dir_path = os.path.join(output_dir, func_name)
    summary_path = evaluate_and_summary(index_path, exp_dir_path)
    print("--> info: summary saved at : {}".format(summary_path))
    print("happy coding.")