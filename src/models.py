import os
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
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
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
        self.model = get_peft_model(self.base_model, self.lora_conf)

        logger.debug("Seeding model parameters...")
        self.seed_model_params()

    def seed_model_params(self):
        """
        Seed the model parameters.
        """
        if os.path.exists(self.config.model.init_weights_path):
            logger.debug("Initial weights already exist. Loading...")
            self.model.load_state_dict(
                torch.load(self.config.model.init_weights_path), strict=False
            )
        else:
            logger.debug("Saving initial weights...")
            state_dict = self.model.state_dict()
            for name in self.model.state_dict():
                if "lora" not in name:
                    state_dict.pop(name)
            torch.save(state_dict, self.config.model.init_weights_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_best_model_state_dict(self, ckpt_path):
        """
        Save the best model state dict.
        Args:
            ckpt_path (str): Path to the best model checkpoint.
        """
        logger.debug(f"Best model path: {ckpt_path}...")
        logger.debug(
            f"Saving final model weights as `state_dicts/{self.config.exp_name}.pt`..."
        )
        best_model = self.load_from_checkpoint(ckpt_path)
        final_state_dict = best_model.state_dict()
        for name in best_model.state_dict():
            if "lora" not in name:
                final_state_dict.pop(name)
        torch.save(final_state_dict, f"state_dicts/{self.config.exp_name}.pt")

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
        sch = self.lr_schedulers()
        sch.step(output.loss)

        self.manual_backward(output.loss)
        optimizer.step()

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
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

    def configure_optimizers(self):
        logger.debug("Configuring optimizer and lr scheduler...")
        optimizer = Adam(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.training.lr_factor,
            patience=self.config.training.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }
