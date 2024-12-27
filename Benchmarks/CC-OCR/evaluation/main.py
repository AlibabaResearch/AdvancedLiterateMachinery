import json
import os
import sys
import time

# local import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluator import evaluator_map_info, summary


def evaluate_and_summary(index_path, exp_dir_path):
    """
    """
    with open(index_path, "r") as f:
        data_list = json.load(f)

    all_evaluation_info = {}
    res_path = os.path.join(exp_dir_path, "status.json")
    keeper_base = os.path.abspath(os.path.join(os.path.dirname(index_path), ".."))
    for data_info in data_list:
        data_name = data_info["dataset"]
        group_name = data_info["group"]
        is_release = data_info.get("release", True)
        if not is_release:
            # print("--> skip: not release: {}".format(data_name))
            continue

        data_base_dir = os.path.join(keeper_base, data_info["base_dir"])
        kie_gt_file_path = os.path.join(data_base_dir, "label.json")
        pdt_res_dir_path = os.path.join(exp_dir_path, data_name)
        if not os.path.exists(pdt_res_dir_path):
            continue

        with open(kie_gt_file_path, "r") as f:
            gt_info = json.load(f)

        eval_func = evaluator_map_info.get(group_name, None)
        if eval_func is None:
            raise ValueError("error: evaluator not defined for: {}".format(group_name))

        meta_info, eval_info = eval_func(pdt_res_dir_path, gt_info, **data_info)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        all_evaluation_info[data_name] = {
            "config": data_info, "meta": meta_info,
            "evaluation": eval_info, "time": formatted_time
        }

    print("--> exp evaluation results save at: {}".format(os.path.abspath(res_path)))
    with open(res_path, "w") as f:
        json.dump(all_evaluation_info, f, ensure_ascii=False, indent=4)

    # summary all exp in the parent dir
    exp_dir_base = os.path.dirname(os.path.abspath(exp_dir_path))
    summary_path = summary(index_path, exp_dir_base)
    return summary_path


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python {} index_path exp_dir_path".format(__file__))
        exit(-1)
    else:
        print('--> info: {}'.format(sys.argv))
        index_path, exp_dir_path = sys.argv[1], sys.argv[2]

    with open(index_path, "r") as f:
        data_list = json.load(f)

    summary_path = evaluate_and_summary(index_path, exp_dir_path)
    print("--> info: summary saved at : {}".format(summary_path))
    print("happy coding.")
