import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from src.config import get_config
from src.data import SequenceClassificationDataModule
from src.models import PeftModelForSequenceClassification

from argparse import ArgumentParser


def main(exp_name, exp_num):
    # Set the seed
    seed_everything(42)

    # Get the configuration for GLUE-SST2
    config = get_config(exp_num)

    # Data module
    data_module = SequenceClassificationDataModule(
        model_name=config.model.model_name,
        dataset_name=tuple(config.data.dataset),
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
    )
    data_module.prepare_data()
    data_module.setup()

    # Model
    model = PeftModelForSequenceClassification(config)

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f"cache/{exp_name}/checkpoints",
            save_top_k=1,
            monitor="val/acc",
            mode="max",
        ),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        precision=config.training.precision,
        progress_bar_refresh_rate=1,
        devices=0,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--exp_name", type=str, required=True)
    arg_parser.add_argument("--exp_num", type=int, required=True)
    args = arg_parser.parse_args()

    main(args.exp_name, args.exp_num)
