import os
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

from src.config import get_config
from src.data import SequenceClassificationDataModule
from src.models import PeftModelForSequenceClassification


def main(exp_name, task):
    # Get the configuration for GLUE-SST2
    config = get_config(exp_name, task)

    # Create the experiment directory
    os.makedirs(f"experiments/{exp_name}/checkpoints", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/logs", exist_ok=True)
    os.makedirs("state_dicts/", exist_ok=True)

    logger.info(f"Running experiment: {exp_name}")

    # Set the seed
    seed_everything(config.seed)

    # Set the float32 matmul precision
    torch.set_float32_matmul_precision("high")

    # Data module
    logger.info("Preparing data module")
    data_module = SequenceClassificationDataModule(
        model_name=config.model.model_name,
        dataset_name=tuple(config.data.dataset),
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
    )
    data_module.prepare_data()
    data_module.setup()

    # Model
    logger.info("Creating model")
    model = PeftModelForSequenceClassification(config)

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

    loggers = [
        CSVLogger(f"experiments/{exp_name}/logs/"),
        WandbLogger(
            project="LoRA-Ensembling",
            name=exp_name,
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
        val_check_interval=config.training.val_check_interval,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    logger.info("Training model")
    trainer.fit(model, datamodule=data_module)

    logger.info("Training complete. Saving final model state dict")
    model.save_best_model_state_dict(callbacks[0].best_model_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--exp_name", type=str, required=True)
    arg_parser.add_argument("--task", type=int, required=True)
    args = arg_parser.parse_args()

    main(args.exp_name, args.task)
