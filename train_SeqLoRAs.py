import os
from typing import List
from argparse import ArgumentParser
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

from src.config import get_seq_lora_config, add_task_to_seq_lora_config
from src.data import get_datamodule
from src.models import SequentialLoRAModel

"""
# TODO: save init weights r1, r2, r3
# TODO: train for 1 epoch and save weights r1, r2, r3
# TODO: load these weights back to train for 2nd epoch and save them
# TODO: Check if all these weights are changing and remaining consistent
# TODO: Load best model at the end of training and save lora weights, cls weights
# TODO: Save hyperparameters at the end of training
# TODO: Run the experiment for "sst2, mnli, cola"
"""


def main(exp_name: str, tasks: List[str]):
    config = get_seq_lora_config(exp_name, "qv")

    # Set the seed
    seed_everything(config.seed)

    # Set the float32 matmul precision
    torch.set_float32_matmul_precision("high")

    # Create the experiment directory
    os.makedirs(f"experiments/{exp_name}/checkpoints", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/logs", exist_ok=True)
    os.makedirs("state_dicts/seq-loras", exist_ok=True)

    logger.info(f"Running experiment: {exp_name}")

    model = None
    data_modules = {}
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"experiments/{exp_name}/checkpoints",
            save_top_k=1,
            monitor="val/acc",
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/loss",
            patience=config.training.early_stopping_patience,
            mode="min",
        ),
    ]

    for i, task in enumerate(tasks):
        # Adding task to Config
        logger.info(f"Adding task - `{task}` to config dict")
        if model is None:
            config = add_task_to_seq_lora_config(config, task)
        else:
            model.config = add_task_to_seq_lora_config(model.config, task)
            config = model.config

        # Data module
        logger.info(f"Preparing data module - `{task}`")
        data_module_cls = get_datamodule(task)
        data_modules[task] = data_module_cls(
            model_name=config.model.model_name,
            batch_size=config.training.batch_size,
            num_workers=config.data[i].num_workers,
        )
        data_modules[task].prepare_data()
        data_modules[task].setup()

        # Model
        if model is None:
            logger.info("Creating model")
            model = SequentialLoRAModel(config)
        else:
            model.add_task(task)
            model.set_curr_task(task)

        # Loggers
        loggers = [
            CSVLogger(f"experiments/{exp_name}/logs/"),
            WandbLogger(
                project="LoRA-Ensembling",
                name=f"exp_name-{task}",
                log_model=True,
                save_dir="experiments",
            ),
        ]

        # Trainer
        logger.info("Creating trainer")
        trainer = Trainer(
            max_epochs=config.training.max_epochs,
            accelerator=config.training.accelerator,
            precision=config.training.precision,
            deterministic=True,
            val_check_interval=config.training.val_check_interval,
            callbacks=callbacks,
            logger=loggers,
        )

        # Save the init model weights
        model.save_lora_stack(
            f"state_dicts/seq-loras/loras_r{model.config.lora.r}_{'-'.join(tasks)}_INIT.pt"
        )
        model.save_classifier(
            f"state_dicts/seq-loras/cls_r{model.config.lora.r}_{'-'.join(tasks)}_INIT.pt"
        )

        # Train the model
        logger.info("Training model")
        trainer.fit(model, datamodule=data_modules[task])

        # Save the model
        model.save_lora_stack(
            f"state_dicts/seq-loras/loras_r{model.config.lora.r}_{'-'.join(tasks)}.pt"
        )
        model.save_classifier(
            f"state_dicts/seq-loras/cls_r{model.config.lora.r}_{'-'.join(tasks)}.pt"
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--exp_name", type=str, required=True)
    arg_parser.add_argument("--tasks", type=str, required=True)
    args = arg_parser.parse_args()

    main(args.exp_name, args.tasks.split("-"))
