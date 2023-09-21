import random
import torch
import io
import pyarrow as pa
import os
from PIL import Image

from torchvision import transforms

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


def get_img_transforms(size=512):

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomRotation(20, resample=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )


class SynthTextDataset(torch.utils.data.Dataset):
    def __init__(self, _config, split):
        super().__init__()

        self.transforms = get_img_transforms(_config["image_size"])
        self.text_column_name = _config["text_column_name"]
        self.max_text_len = _config["max_text_len"]
        self.data_dir = _config["data_dir"]
        self.split = split
        
        self.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )

        self.table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{self.data_dir}/synthtext_{self.split}.arrow", "r")
            ).read_all()
        self.all_texts = self.table[self.text_column_name].to_pandas().tolist()

    @property
    def corpus(self):
        return [text for text in self.all_texts]

    def __len__(self):
        return len(self.all_texts)

    def get_image(self, index, image_key="image"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        image_tensor = self.transforms(image)
        return {
            "image": image_tensor,
            "index": index,
        }

    def get_text(self, index):
        text = self.all_texts[index][0]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "index": index,
        }

    def __getitem__(self, index):
        ret = dict()
        ret.update(self.get_image(index))
        ret.update(self.get_text(index))
        return ret
    
    def collate(self, batch):
        batch_size = len(batch)
        keys = batch[0].keys()

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]

        for img_key in img_keys:
            dict_batch[img_key] = torch.stack(dict_batch[img_key])

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]

            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)      

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_masks = torch.zeros_like(mlm_ids)

                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_masks[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_masks

        return dict_batch
