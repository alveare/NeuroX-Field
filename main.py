import os
import time
from typing import Any

from src.utils.patch_bugs import *
from lightning import Trainer, LightningModule
from src.lightning_data import DataModule
from src.lightning_model import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback

import logging
logger = logging.getLogger("lightning.pytorch")

WORKDIR_ROOT = os.environ.get("WORKDIR_ROOT", "workdirs")


class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class ReWriteRootDirCli(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config_trainer = self._get(self.config, "trainer", default={})
        # Override exp name from environment variable if set
        exp_name = os.environ.get("EXP_NAME")
        if exp_name:
            tags_config = self._get(self.config, "tags", default={})
            tags_config["exp"] = exp_name
            # Also update default_root_dir if EXP_NAME is set
            config_trainer["default_root_dir"] = f"./{WORKDIR_ROOT}/{exp_name}"
            
            # Update logger settings if logger exists
            logger_config = self._get(self.config, "trainer.logger", default=None)
            if logger_config is not None:
                init_args = logger_config.get("init_args", {})
                if "save_dir" in init_args:
                    init_args["save_dir"] = f"./{WORKDIR_ROOT}/{exp_name}"
                if "name" in init_args:
                    init_args["name"] = exp_name

        # predict path & logger check
        if self.subcommand == "predict":
            config_trainer.logger = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), WORKDIR_ROOT)

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)
        is_resume = self._get(self.config_init, "ckpt_path", default=None)
        current_rank = os.environ.get("GLOBAL_RANK", os.environ.get("SLURM_PROCID", "0"))
        print(f"current_rank: {current_rank}")
        if current_rank == "0" and os.path.exists(default_root_dir) and "debug" not in default_root_dir:
            if os.listdir(default_root_dir) and self.subcommand != "predict" and not is_resume:
                raise FileExistsError(f"{default_root_dir} already exists")
        kwargs["enable_progress_bar"] = (current_rank == "0")
        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if current_rank == "0":
            os.makedirs(default_root_dir, exist_ok=True)
        return trainer

    def instantiate_classes(self) -> None:
        super().instantiate_classes()

if __name__ == "__main__":

    cli = ReWriteRootDirCli(LightningModel, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})