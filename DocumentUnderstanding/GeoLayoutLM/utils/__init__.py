import os
import datetime

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


def get_config(default_conf_file="./configs/default.yaml"):
    cfg = OmegaConf.load(default_conf_file)

    cfg_cli = _get_config_from_cli()
    if "config" in cfg_cli:
        cfg_cli_config = OmegaConf.load(cfg_cli.config)
        cfg = OmegaConf.merge(cfg, cfg_cli_config)
        del cfg_cli["config"]

    cfg = OmegaConf.merge(cfg, cfg_cli)

    _update_config(cfg)

    return cfg


def _get_config_from_cli():
    cfg_cli = OmegaConf.from_cli()
    cli_keys = list(cfg_cli.keys())
    for cli_key in cli_keys:
        if "--" in cli_key:
            cfg_cli[cli_key.replace("--", "")] = cfg_cli[cli_key]
            del cfg_cli[cli_key]

    return cfg_cli


def _update_config(cfg):
    # if os.path.exists(cfg.workspace):
    #     cfg.workspace = cfg.workspace.rstrip('/') + '_' + datetime.datetime.now().strftime('%m%d%H%M')
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
    cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")

    if cfg.dataset == "funsd":
        cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, "funsd_geo")
        cfg.model.n_classes = 7

    elif cfg.dataset == "cord":
        cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, "cord_geo")
        cfg.model.n_classes = 2 * 22 + 1

    # set per-gpu batch size
    num_devices = torch.cuda.device_count()
    for mode in ["train", "val"]:
        new_batch_size = cfg[mode].batch_size // num_devices
        cfg[mode].batch_size = new_batch_size


def get_callbacks(cfg):
    callbacks = []
    cb1 = CustomModelCheckpoint(
        dirpath=cfg.save_weight_dir, filename='{epoch}-{f1_labeling:.4f}', monitor="f1_labeling", save_top_k=1, mode='max',
         save_last=False, every_n_epochs=1, save_on_train_epoch_end=False
    )
    cb1.CHECKPOINT_NAME_LAST = "{epoch}-last"
    cb1.FILE_EXTENSION = ".pt"
    callbacks.append(cb1)

    cb2 = CustomModelCheckpoint(
        dirpath=cfg.save_weight_dir, filename='{epoch}-{f1_linking:.4f}', monitor="f1_linking", save_top_k=1, mode='max',
         save_last=False, every_n_epochs=1, save_on_train_epoch_end=False
    )
    cb2.CHECKPOINT_NAME_LAST = "{epoch}-last"
    cb2.FILE_EXTENSION = ".pt"
    callbacks.append(cb2)

    return callbacks


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        trainer.fit_loop.global_step -= 1
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._save_on_train_epoch_end
            and self._every_n_epochs > 0
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        ):
            self.save_checkpoint(trainer)
        trainer.fit_loop.global_step += 1


def get_plugins(cfg):
    plugins = []

    if cfg.train.strategy.type == "ddp":
        plugins.append(DDPPlugin())

    return plugins


def get_loggers(cfg):
    loggers = []

    loggers.append(
        TensorBoardLogger(
            cfg.tensorboard_dir, name="", version="", default_hp_metric=False
        )
    )

    return loggers


def cfg_to_hparams(cfg, hparam_dict, parent_str=""):
    for key, val in cfg.items():
        if isinstance(val, DictConfig):
            hparam_dict = cfg_to_hparams(val, hparam_dict, parent_str + key + "__")
        else:
            hparam_dict[parent_str + key] = str(val)
    return hparam_dict


def get_specific_pl_logger(pl_loggers, logger_type):
    for pl_logger in pl_loggers:
        if isinstance(pl_logger, logger_type):
            return pl_logger
    return None


def get_class_names(dataset_root_path):
    class_names_file = os.path.join(dataset_root_path, "class_names.txt")
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    return class_names


def get_label_map(dataset_root_path):
    label_map_file = os.path.join(dataset_root_path, "labels.txt")
    label_map = {}
    lines = open(label_map_file, "r", encoding="utf-8").readlines()
    for line_idx, line in enumerate(lines):
        label_map[line_idx] = line.strip()
    return label_map
