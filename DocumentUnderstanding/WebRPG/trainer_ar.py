import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, List, Optional, Tuple, Union
import json

import numpy as np
import transformers

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    BartModel,
    BartConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import torch
from torch.utils.data import Dataset

from baselines_model.vae import VAE
from markuplm import MarkupLMConfig, MarkupLMModel
from baselines_model.BART_vae_web_rendering import *

from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)
pretrained_vae = None

@dataclass
class ModelArguments:
    """
    Arguments of model.
    """
    channels: int = field(
        default=13, metadata={"help": "The number of CSS properties."}
    )
    num_elements: int = field(
        default=128, metadata={"help": "The number of web elements."}
    )
    param_dim: int = field(
        default=128, metadata={"help": "The dim of rendering parameter features."}
    )
    text_dim: int = field(
        default=1024, metadata={"help": "The dim of text features."}
    )
    chrlen_dim: int = field(
        default=128, metadata={"help": "The dim fo character length feature."}
    )
    xpath_dim: int = field(
        default=1024, metadata={"help": "The dim of xpath feature."}
    )
    latent_dim: int = field(
        default=128, metadata={"help": "The hidden dim of transformer."}
    )
    max_chrlen: int = field(
        default=512, metadata={"help": "The max length of element char."}
    )
    depth: int = field(
        default=12, metadata={"help": "The depth of transformer."}
    )
    mask_token: int = field(
        default=585, metadata={"help": "The id of [MASK]."}
    )
    pad_token: int = field(
        default=1992, metadata={"help": "The id of [PAD]."}
    )

    pretrained_markuplm_path: str = field(
        default="./markuplm-large", metadata={"help": "The path of pretrained markuplm."}
    )

    # VAE
    vae_input_dim: int = field(
        default=1993, metadata={"help": "The dim of VAE input."}
    )
    vae_is_onehot: bool = field(
        default=True, metadata={"help": "Does VAE use one hot encoding."}
    )
    vae_is_dimension_reduction: bool = field(
        default=True, metadata={"help": "Does VAE reduce dimension."}
    )
    vae_layer_num: int = field(
        default=5, metadata={"help": "The layer number of vae."}
    )
    vae_start_hidden_dim: int = field(
        default=128, metadata={"help": "The start hidden dim of vae."}
    )
    vae_pretrained_path: str = field(
        default=None, metadata={"help": "The path of pretrained vae."}
    )
    vae_is_layernorm: bool = field(
        default=True, metadata={"help": "If use layernorm."}
    )
    vae_loss_weight : float = field(
        default=1.0, metadata={"help": "The weight of vae loss."}
    )
    kld_weight : float = field(
        default=1e-6, metadata={"help": "The weight of kld loss."}
    )
   
    max_render_range: int = field(
        default=1920, metadata={"help": "The maximum rendering range."}
    )

    def to_json_string(self):
        return json.dumps(self.__dict__)


@dataclass
class MyTrainingArguments(TrainingArguments):
    cache_path: str = field(default=None, metadata={"help": "The path of training cache file."})
    cache_eval_path: str = field(default=None, metadata={"help": "The path of eval cache file."})

    save_safetensors : bool = field(default=False, metadata={"help": "Use safetensors saving and loading for state dicts."})

    # related to eval
    checkpoint_dir: str = field(default=None, metadata={"help":"The directory of checkpoint."})
    max_eval_samples: int = field(default=None, metadata={"help":"The max eval samples."})
    eval_output_dir: str = field(default=None, metadata={"help":"The output directory of evaluation."})


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.other_losses = self.OtherLosses()

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            for key in ["vae_loss", "recons_loss", "kld_loss", "bart_loss"]:
                logs[key] = round(getattr(self.other_losses, key) / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self.other_losses.reset()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
            
        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    class OtherLosses:
        def __init__(self):
            self.vae_loss = 0.0
            self.recons_loss = 0.0
            self.kld_loss = 0.0
            self.bart_loss = 0.0
        def reset(self):
            self.vae_loss = 0.0
            self.recons_loss = 0.0
            self.kld_loss = 0.0
            self.bart_loss = 0.0


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.n_gpu > 1:
            for key in ["vae_loss", "recons_loss", "kld_loss", "bart_loss"]:
                outputs[key] = outputs[key].mean()
        for key in ["vae_loss", "recons_loss", "kld_loss", "bart_loss"]:
            self.other_losses.__dict__[key] += outputs[key].item()

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
            

class training_dataset(Dataset):
    def __init__(self, file_lis, is_eval=False):
        self.file_lis = file_lis
        self.is_eval = is_eval
    def __len__(self):
        return len(self.file_lis)

    def __getitem__(self, idx):
        file_dir = self.file_lis[idx]

        new_data = {
            "param": torch.from_numpy(torch.load(os.path.join(file_dir, "meta_data_seq.pt")).astype(np.int64)),
            "all_xpath_tags_seq": torch.from_numpy(torch.load(os.path.join(file_dir, "all_xpath_tags_seq.pt")).astype(np.int64)),
            "all_xpath_subs_seq": torch.from_numpy(torch.load(os.path.join(file_dir, "all_xpath_subs_seq.pt")).astype(np.int64)),
            "element_mask": torch.from_numpy(torch.load(os.path.join(file_dir, "element_mask.pt")).astype(np.int64)),
            "chrlen": torch.from_numpy(torch.load(os.path.join(file_dir, "element_text_len.pt")).astype(np.int64)),
            "global_text": torch.from_numpy(torch.load(os.path.join(file_dir, "all_pooled_embeddings.pt")).astype(np.float32)),
            "element_text": torch.from_numpy(torch.load(os.path.join(file_dir, "all_sequence_embeddings.pt")).astype(np.float32)),
        }

        if self.is_eval:
            new_data["mask_ratio"] = np.array([0.0])
            new_data["file_id"] = torch.from_numpy(torch.load(os.path.join(file_dir, "file_id.pt")).astype(np.int64))
            new_data["offset"] = torch.from_numpy(torch.load(os.path.join(file_dir, "offset.pt")).astype(np.int64))
            new_data["unique_tids"] = torch.from_numpy(torch.load(os.path.join(file_dir, "unique_tids.pt")).astype(np.int64))

        new_data["global_text"] = torch.from_numpy(torch.load(os.path.join(file_dir, "all_pooled_embeddings.pt")).astype(np.float32))
        new_data["element_text"] = torch.from_numpy(torch.load(os.path.join(file_dir, "all_sequence_embeddings.pt")).astype(np.float32))

        return new_data


def main():
    # Argument
    parser = HfArgumentParser((ModelArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info("loading pretrained markuplm ...")
    markup_config = MarkupLMConfig.from_pretrained(model_args.pretrained_markuplm_path)
    markuplm = MarkupLMModel(markup_config)
    model_CKPT = torch.load(os.path.join(model_args.pretrained_markuplm_path,"pytorch_model.bin"), map_location='cpu')
    markuplm.load_state_dict(model_CKPT,False)
    xpath_layer = copy.deepcopy(markuplm.embeddings.xpath_embeddings)
    logger.info("Loading completed.")

    logger.info("loading pretrained vae ...")
    vae_layer_num = model_args.vae_layer_num
    vae_start_hidden_dim = model_args.vae_start_hidden_dim
    vae_hidden_dims = []
    for i in range(vae_layer_num):
        vae_hidden_dims.append(vae_start_hidden_dim*(2 ** i))
    pretrained_vae = VAE(input_dim = model_args.vae_input_dim,latent_dim = model_args.param_dim,parameters_len=model_args.channels,hidden_dims=vae_hidden_dims)
    if model_args.vae_pretrained_path != None:
        model_CKPT = torch.load(model_args.vae_pretrained_path, map_location='cpu')
        pretrained_vae.load_state_dict(model_CKPT['model'],False)
    logger.info("Loading completed.")
    
    # init bart
    bart_config = BartConfig(
        max_position_embeddings=64*13+1,
        encoder_layers=model_args.depth//2,
        decoder_layers=model_args.depth//2,
        d_model=model_args.latent_dim
        )
    bart = BartModel(bart_config)

    # init bartbackbone
    logger.info("init bartbackbone ...")
    backbone = BartVAEBackbone(
        in_dim=model_args.param_dim,
        out_dim=model_args.param_dim,
        embed_dim=model_args.latent_dim,
        num_element_tokens = model_args.num_elements,
        global_text_dim = model_args.text_dim,
        element_text_dim = model_args.text_dim,
        chrlen_dim = model_args.chrlen_dim,
        xpath_dim = model_args.xpath_dim,
        bart=bart)
    
    logger.info("inti bart_web_model ...")
    bart_web_model = BartVAEWebModel(
        model=backbone,
        xpath_layer=xpath_layer,
        max_chrlen=model_args.max_chrlen,
        chrlen_dim=model_args.chrlen_dim,
        mask_dim=model_args.param_dim,
        vae=pretrained_vae,
        kld_weight=model_args.kld_weight,
        vae_loss_weight=model_args.vae_loss_weight,
        config=model_args,
        max_render_range=model_args.max_render_range
    )

    # Training
    if training_args.do_train:
        # DataLoader
        cache_file = training_args.cache_path
        # train_dataset
        with open(cache_file, 'r') as f:
            data_lis = [line.strip() for line in f.readlines()]
        random.shuffle(data_lis)
        train_data = training_dataset(data_lis)

        # Initialize our Trainer
        trainer = MyTrainer(
            model=bart_web_model,
            args=training_args,
            train_dataset=train_data
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        ignore_keys_for_eval = ["bart_loss","vae_loss","recons_loss","kld_loss", "last_hidden_state","encoder_last_hidden_state"]
        train_result = trainer.train(resume_from_checkpoint=checkpoint,ignore_keys_for_eval=ignore_keys_for_eval) 
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics["train_samples"] = len(train_data)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
      # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Create directory
        if not os.path.exists(training_args.eval_output_dir):
            os.mkdir(training_args.eval_output_dir)
            os.mkdir(os.path.join(training_args.eval_output_dir,"result_dict"))

        # Save the checkpoint path
        json_file_path = os.path.join(training_args.eval_output_dir,"checkpoint_dir.json")
        checkpoint_dir_data = {"checkpoint_dir": training_args.checkpoint_dir}
        with open(json_file_path, 'w') as f:
            json.dump(checkpoint_dir_data, f, ensure_ascii=False, indent=4)

        # eval data
        with open(training_args.cache_eval_path, 'r') as f:
            data_lis = [line.strip() for line in f.readlines()]
        random.shuffle(data_lis)
        eval_dataset = training_dataset(data_lis, is_eval=True)
        max_val_samples = training_args.max_eval_samples if training_args.max_eval_samples is not None else len(eval_dataset)
        val_sampler = SubsetRandomSampler(random.sample(range(len(eval_dataset)), max_val_samples))
        eval_dataloader = DataLoader(eval_dataset,batch_size=training_args.per_device_eval_batch_size,num_workers=training_args.dataloader_num_workers,sampler=val_sampler)

        if training_args.checkpoint_dir is not None:
            last_checkpoint = training_args.checkpoint_dir 
        logger.info(f"Loading checkpoint from {last_checkpoint}")
        model_CKPT = torch.load(os.path.join(last_checkpoint,"pytorch_model.bin"), map_location='cpu')
        bart_web_model.load_state_dict(model_CKPT, strict=False)
        bart_web_model.to("cuda")
        bart_web_model.eval()

        for batch in tqdm(eval_dataloader):
            for k,v in batch.items():
                batch[k] = batch[k].to("cuda")

            with torch.no_grad():
                output = bart_web_model(param=batch["param"],global_text=batch["global_text"],element_text=batch["element_text"],chrlen=batch["chrlen"],element_mask=batch["element_mask"],all_xpath_subs_seq=batch["all_xpath_subs_seq"],all_xpath_tags_seq=batch["all_xpath_tags_seq"],mask_ratio=0)

            element_len = batch["element_mask"].shape[1]
            param = output["last_hidden_state"].reshape(batch["param"].shape[0],element_len,-1)
            pred = output["pred"]
            pred = torch.argmax(pred,dim=-1)
            pred = pred.reshape(batch["param"].shape)

            def to_numpy(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        to_numpy(v)
                    elif isinstance(v, torch.Tensor):
                        d[k] = v.cpu().numpy()
                return d
            batch = to_numpy(batch)

            for i in range(batch["file_id"].shape[0]):
                result_dict = {
                    "file_id":batch["file_id"][i][0],
                    "offset":batch["offset"][i][0],
                    "gt":batch["param"][i],
                    "pred":pred[i].cpu().numpy(),
                    "pred_emd":param[i].cpu().numpy(),
                    "unique_tids":batch["unique_tids"][i],
                    "element_mask":batch["element_mask"][i],
                    "all_xpath_tags_seq":batch["all_xpath_tags_seq"][i],
                    "all_xpath_subs_seq":batch["all_xpath_subs_seq"][i],
                    "chrlen":batch["chrlen"][i]}

                torch.save(result_dict, os.path.join(training_args.eval_output_dir,"result_dict",f'result_{result_dict["file_id"]}_{result_dict["offset"]}.pt'))


if __name__ == "__main__":
    main()