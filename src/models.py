import os
import shutil
from copy import deepcopy
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torchmetrics.classification import Accuracy

from loguru import logger


class PeftModelForSequenceClassification(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.accuracy = Accuracy(task="multiclass", num_classes=config.data.num_labels)
        self.accuracy.to(config.training.device)
        self.automatic_optimization = False

        logger.debug(f"Loading model {config.model.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=config.data.num_labels,
            cache_dir=config.model.cache_dir,
        )
        self.lora_conf = LoraConfig(
            r=config.lora.r,
            target_modules=config.lora.target_modules,
            modules_to_save=config.lora.modules_to_save,
        )

        logger.debug("Creating LoRA model...")
        self.model = get_peft_model(self.model, self.lora_conf)

        logger.debug("Seeding model parameters...")
        self.seed_model_params()

    def seed_model_params(self):
        """
        Seed the model parameters.
        """
        if os.path.exists(self.config.model.init_lora_weights):
            logger.debug(
                f"Initial weights already exist at `{self.config.model.init_lora_weights}`. Loading..."
            )
            self.load_state_dict(
                torch.load(self.config.model.init_lora_weights), strict=False
            )
        else:
            logger.debug(
                f"Saving initial weights at `{self.config.model.init_lora_weights}`"
            )
            state_dict = self.model.state_dict()
            for name in self.model.state_dict():
                if "lora" not in name:
                    state_dict.pop(name)
            torch.save(state_dict, self.config.model.init_lora_weights)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        output = self(**batch)
        metrics = {
            "train/loss": output.loss,
            "train/acc": self.accuracy(output.logits, batch["labels"]),
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        if self.config.training.lr_scheduler:
            sch = self.lr_schedulers()
            sch.step(output.loss)

        self.manual_backward(output.loss)
        optimizer.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(**batch)
        if dataloader_idx == 0:
            metrics = {
                "val_matched/loss": output.loss,
                "val_matched/acc": self.accuracy(output.logits, batch["labels"]),
            }
        elif dataloader_idx == 1:
            metrics = {
                "val_mismatched/loss": output.loss,
                "val_mismatched/acc": self.accuracy(output.logits, batch["labels"]),
            }
        else:
            metrics = {
                "val/loss": output.loss,
                "val/acc": self.accuracy(output.logits, batch["labels"]),
            }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(**batch)
        if dataloader_idx == 0:
            metrics = {
                "test_matched/acc": self.accuracy(output.logits, batch["labels"]),
            }
        elif dataloader_idx == 1:
            metrics = {
                "test_mismatched/acc": self.accuracy(output.logits, batch["labels"]),
            }
        else:
            metrics = {
                "test/acc": self.accuracy(output.logits, batch["labels"]),
            }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        logger.debug("Configuring optimizer and lr scheduler...")
        optimizer = Adam(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        opt = {"optimizer": optimizer}
        if self.config.training.lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.training.lr_factor,
                patience=self.config.training.lr_scheduler_patience,
            )
            opt["lr_scheduler"] = {"scheduler": lr_scheduler}

        return opt

    def save_artifacts(self, trainer):
        """
        Save the best model state dict and run's hyperparameters.
        Args:
            trainer (Obj): Trainer object that stores the ModelCheckpoint callback.
        """

        logger.debug(f"Saving current run's hyperparameters...")
        latest_version = os.path.join(
            sorted(
                os.listdir(f"experiments/{self.config.exp_name}/logs/lightning_logs")
            )[-1],
            "hparams.yaml",
        )
        source = os.path.join(
            f"experiments/{self.config.exp_name}/logs/lightning_logs/",
            latest_version,
        )
        destination = f"hparams/{self.config.exp_name}.yaml"

        logger.debug(f"Copying hyperparameters from `{source}` -> `{destination}`")
        shutil.copyfile(
            source,
            destination,
        )

        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                ckpt_path = callback.best_model_path
                break
        else:
            raise ValueError("ModelCheckpoint callback not found in trainer callbacks.")

        logger.debug(f"Best model path: `{ckpt_path}`")
        try:
            self.model = self.__class__.load_from_checkpoint(checkpoint_path=ckpt_path)
        except PermissionError as e:
            logger.error(f"Error loading model from checkpoint. {e}")

        logger.debug(
            f"Saving all trained weights at `state_dicts/{self.config.exp_name}_ALL.pt`"
        )
        state_dict = self.model.state_dict()
        for name in self.model.state_dict():
            if "lora" not in name and "classifier" not in name:
                state_dict.pop(name)
        torch.save(state_dict, f"state_dicts/{self.config.exp_name}_ALL.pt")
