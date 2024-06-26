import torch
from torch import nn
from collections import OrderedDict
from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification
from torchmetrics.classification import Accuracy

from loguru import logger


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, lora_config):
        super().__init__()
        self.lora_config = lora_config
        std_dev = 1 / torch.sqrt(torch.tensor(self.lora_config.r).float())
        self.lora_A = torch.nn.Parameter(
            torch.randn(in_dim, self.lora_config.r) * std_dev
        )
        self.lora_B = torch.nn.Parameter(torch.zeros(self.lora_config.r, out_dim))

    def forward(self, x):
        x = self.lora_config.lora_alpha * (x @ self.lora_A @ self.lora_B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, lora_config):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, lora_config)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class SequentialLoRAModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.curr_task = len(config.tasks) - 1
        self.accuracy = Accuracy(task="multiclass", num_classes=config.data.num_labels)
        self.accuracy.to(config.training.device)
        self.automatic_optimization = False

        logger.debug(f"Loading model {config.model.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=config.data.num_labels,
            cache_dir=config.model.cache_dir,
        )

        logger.debug("Model loaded successfully.")
        logger.debug("Injecting LoRA layers...")
        self.inject_lora_layers(config.config.lora.tasks[self.curr_task])

        logger.debug("Seeding model weights...")

    # def get_specific_state_dict(self, module: str):
    #     """
    #     Get the state dictionary of a specific module.

    #     """
    #     state_dict = self.model.state_dict()
    #     specific_state_dict = OrderedDict()
    #     for key in state_dict.keys():
    #         if module in key:
    #             specific_state_dict[key] = state_dict[key]
    #     return specific_state_dict

    # def set_extra_state(self, state: torch.Any):
    #     return super().set_extra_state(state)

    def inject_lora_layers(self, lora_config):
        """
        Inject LoRA layers into the model.

        Parameters:
        lora_config (ConfigDict): The LoRA configuration.

        Returns:
        None
        """
        for i in range(12):
            if "query" in lora_config.target_modules:
                self.model.bert.encoder.layer[i].attention.self.query = LinearWithLoRA(
                    self.model.bert.encoder.layer[i].attention.self.query, lora_config
                )
            if "key" in lora_config.target_modules:
                self.model.bert.encoder.layer[i].attention.self.key = LinearWithLoRA(
                    self.model.bert.encoder.layer[i].attention.self.key, lora_config
                )
            if "value" in lora_config.target_modules:
                self.model.bert.encoder.layer[i].attention.self.value = LinearWithLoRA(
                    self.model.bert.encoder.layer[i].attention.self.value, lora_config
                )
            if "output" in lora_config.target_modules:
                self.model.bert.encoder.layer[i].attention.output.dense = (
                    LinearWithLoRA(
                        self.model.bert.encoder.layer[i].attention.output.dense,
                        lora_config,
                    )
                )
