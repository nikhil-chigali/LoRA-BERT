import os
import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torchmetrics.classification import Accuracy

from loguru import logger


class PeftModelForSequenceClassification(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.accuracy = Accuracy(task="multiclass", num_classes=config.data.num_labels)
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

        logger.debug("Saving initial weights...")
        if os.path.exists(config.model.init_weights_path):
            self.model.load_state_dict(torch.load(config.model.init_weights_path))
        else:
            torch.save(self.model.state_dict(), config.model.init_weights_path)

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

        sch = self.lr_schedulers()
        sch.step(output.loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.config.training.lr_factor,
            patience=self.config.training.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val/loss",
            },
        }
