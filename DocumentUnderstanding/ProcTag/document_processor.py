import re
from json_helper import JsonHelper
from geometry_utils import box_contains, find_closest_box
from openai_integration import call_GPT
import torch
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer, AutoModel
from space_layout import DocSpaceLayout
import numpy as np

class DocumentProcessor:

    def ocr2local_prompt(self, ocr_data):
        if not ocr_data:
            return ""
        texts = [x["text"] for x in ocr_data]
        text_boxes = [x["box"] for x in ocr_data]
        doc_space_layout = DocSpaceLayout()
        doc_str, space_line_texts = doc_space_layout.space_layout(texts, text_boxes)
        return doc_str

    def generate_DocLayPrompt(self, data):
        for idx in range(len(data)):
            vgt_list = data[idx]["layout_info"]
            ocr_data = JsonHelper.load_json(data[idx]["ocr_path"])
            for item in vgt_list:
                current_box = item["box"]
                contain_ocr_block = [ocr_block for ocr_block in ocr_data if box_contains(current_box, ocr_block["box"])]
                item["contain_ocr_block"] = contain_ocr_block
                for ocr in contain_ocr_block:
                    ocr["class"] = item["class"]
            nocontain_ocr_block = [item for item in ocr_data if "class" not in item]
            for item in nocontain_ocr_block:
                layout_idx = find_closest_box(item["box"], [vgt["box"] for vgt in vgt_list])
                vgt_list[layout_idx].setdefault("contain_ocr_block", []).append(item)
            doclay_prompt = self._build_local_prompt(vgt_list)
            data[idx]["DocLayPrompt"] = doclay_prompt
        return data
    
    def _build_local_prompt(self, vgt_list):
        doclay_prompt = ''
        rename_dict = {
            "DocTitle": "Title",
            "ParaText": "Paragraph",
            "ListText": "List",
            "OtherText": "Text"
        }
        for item in vgt_list:
            if "contain_ocr_block" not in item:
                continue
            contain_ocr_block = item["contain_ocr_block"]
            for ocr_item in contain_ocr_block:
                ocr_item["text"] += " "
            ocr_text = self.ocr2local_prompt(contain_ocr_block)
            layout_class = rename_dict.get(item["class"], item["class"])
            doclay_prompt += f"<{layout_class}>\n{ocr_text}\n</{layout_class}>\n"
        return doclay_prompt

    def generate_proctags(self, data, prompt_template):
        for item in data:
            message_content = prompt_template.format(DocLayPrompt=item["DocLayPrompt"],
                                                     Question=item["conversations"][0]["value"])
            result = call_GPT(message_content)
            item["result"] = result
        JsonHelper.save_json("./temp.json", data)
        return data

    def parse_proctags(self, data):
        for item in data:
            result = item.get("result")
            if not result:
                print("error: no result")
                continue
            text = result.replace(": ", ":").replace(" -> ", "->").replace('\\"', '"')
            pattern = r'>S\d+:(\w+)\(([^)]+)\)->(\w+);'
            matches = re.findall(pattern, text)
            steps = [{'function': match[0], 'input': [x.strip() for x in match[1].split(',')], 'output': [match[2]]} for match in matches]
            item["steps"] = steps
        return data

    def cluster_and_tag_procedures(self, data, model_name, tokenizer_name, clustering_params):
        func_arr = []
        for item in data:
            if 'steps' not in item or not item['steps']:
                continue
            func_arr.extend([step['function'] for step in item['steps']])
        unique_funcs = list(set(func_arr))
        embeddings = self._get_embeddings(unique_funcs, model_name, tokenizer_name)
        clustering = DBSCAN(**clustering_params).fit(embeddings)
        cluster_labels = clustering.labels_
        clustered_tags = {}
        for func, label in zip(unique_funcs, cluster_labels):
            clustered_tags.setdefault(label, []).append(func)
        
        representative_tags = {}
        for label, funcs in clustered_tags.items():
            representative_tag = funcs[0]
            for func in funcs:
                representative_tags[func] = representative_tag
        
        for item in data:
            if 'steps' not in item:
                continue
            tags = []
            for step in item['steps']:
                step['tag'] = representative_tags[step['function']]
                tags.append({
                    "tag": step["function"]
                })
            item["tags"] = tags
        
        return data

    def _get_embeddings(self, sentences, model_name, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Aggregate the last hidden state embeddings (mean pooling)
            embed = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()
            embeddings.append(embed)
        
        return np.array(embeddings)


    def complexity_first_diverse_sampling(self, data, N):
        D = data.copy()
        if N > len(D):
            raise(ValueError("N must be smaller than the size of D"))
        Ds = []
        D.sort(key=lambda x: len(x['tags']), reverse=True)
        while len(Ds) < N:
            TBs = set()
            for q in D.copy():
                qtags = [tag["tag"] for tag in q['tags']]
                Tq = set(qtags)
                if len(TBs.union(Tq)) > len(TBs):
                    Ds.append(q)
                    TBs.update(Tq)
                    D.remove(q)
                    if len(Ds) == N:
                        break
        return Ds