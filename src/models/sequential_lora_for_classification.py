from collections import OrderedDict
from copy import deepcopy
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification
from torchmetrics.classification import Accuracy

from loguru import logger


## Config structure:
# exp_name: str
# tasks: List[str]
# num_tasks: int
# seed: int
# data: List[ConfigDict]
#     "dataset": List[str]
#     "num_labels": int
#     "class_labels": List[str]
#     "state_dicts_dir": str
#     "cache_dir": str
#     "num_workers": int
# lora: ConfigDict
#     r: int
#     target_modules: List[str]
#     modules_to_save: List[str]
#     lora_alpha: int
# model: ConfigDict
#     model_name: str
#     cache_dir: str
#     state_dicts_dir: str
# training: ConfigDict
#     batch_size: int
#     lr: float
#     device: str
#     accelerator: str
#     lr_scheduler: bool
#     lr_scheduler_patience: int
#     early_stopping_patience: int
#     lr_factor: float
#     val_check_interval: float
#     weight_decay: float
#     max_epochs: int
#     precision: str


class LoRAStack(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        self.lora_config = config.lora
        self.device = config.training.device
        self.in_features = in_dim
        self.out_features = out_dim
        self.std_dev = 1 / torch.sqrt(torch.tensor(self.lora_config.r).float())

        self.tasks = []
        self.num_tasks = 0
        self.curr_task = None

        self.lora_As = []
        self.lora_Bs = []
        self.lora_A = None
        self.lora_B = None

    def add_task(self, task: str):
        self.tasks.append(task)
        self.num_tasks += 1
        self.lora_As.append(
            torch.nn.Parameter(torch.randn(self.in_features, 1) * self.std_dev).to(
                self.device
            )
        )
        self.lora_Bs.append(
            torch.nn.Parameter(torch.zeros(1, self.out_features)).to(self.device)
        )

    def set_curr_task(self, task: str):
        try:
            self.curr_task = self.tasks.index(task)
        except ValueError:
            raise ValueError(f"`{task}` is not a valid task.")

        lora_A = []
        lora_B = []
        for i in range(self.curr_task):
            lora_A.append(self.lora_As[i].requires_grad_(False))
            lora_B.append(self.lora_Bs[i].requires_grad_(False))
        lora_A.append(self.lora_As[self.curr_task].requires_grad_(True))
        lora_B.append(self.lora_Bs[self.curr_task].requires_grad_(True))
        self.lora_A = torch.cat(lora_A, dim=1)
        self.lora_B = torch.cat(lora_B, dim=0)

    def forward(self, x):
        if self.curr_task is None:
            raise ValueError("No current task is set")
        x = self.lora_config.lora_alpha * (x @ self.lora_A @ self.lora_B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, config):
        super().__init__()
        self.linear = linear
        self.lora = LoRAStack(linear.in_features, linear.out_features, config)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class SequentialLoRAModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False
        self.curr_task = None

        self.accuracy = {
            self.config.tasks[0]: Accuracy(
                task="multiclass", num_classes=config.data[0].num_labels
            )
        }
        self.accuracy[self.config.tasks[0]].to(config.training.device)

        logger.debug(f"Loading model {config.model.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=config.data[0].num_labels,
            cache_dir=config.model.cache_dir,
        )
        self.in_features = self.model.classifier.in_features
        self.heads = {
            self.config.tasks[0]: deepcopy(self.model.classifier),
        }

        logger.debug("Model loaded successfully.")
        logger.debug("Injecting LoRA layers...")
        self.inject_lora_to_base_model()

        logger.debug("Add task to LoRA")
        self.add_task(self.config.tasks[0])
        self.set_curr_task(self.config.tasks[0])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        output = self(**batch)
        metrics = {
            "train/loss": output.loss,
            "train/acc": self.accuracy[self.curr_task](output.logits, batch["labels"]),
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
                "val_matched/acc": self.accuracy[self.curr_task](
                    output.logits, batch["labels"]
                ),
            }
        elif dataloader_idx == 1:
            metrics = {
                "val_mismatched/loss": output.loss,
                "val_mismatched/acc": self.accuracy[self.curr_task](
                    output.logits, batch["labels"]
                ),
            }
        else:
            metrics = {
                "val/loss": output.loss,
                "val/acc": self.accuracy[self.curr_task](
                    output.logits, batch["labels"]
                ),
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

    def inject_lora_to_base_model(self):
        """
        Inject LoRA layers into the model.
        """
        for i in range(12):
            if "query" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.query = LinearWithLoRA(
                    self.model.bert.encoder.layer[i].attention.self.query,
                    self.config,
                )
            if "key" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.key = LinearWithLoRA(
                    self.model.bert.encoder.layer[i].attention.self.key,
                    self.config,
                )
            if "value" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.value = LinearWithLoRA(
                    self.model.bert.encoder.layer[i].attention.self.value,
                    self.config,
                )
            if "output" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.output.dense = (
                    LinearWithLoRA(
                        self.model.bert.encoder.layer[i].attention.output.dense,
                        self.config,
                    )
                )

    def add_task(self, task: str):
        """
        Add a new task to the model. Assumes that the `add_task_to_seq_lora_config` is called.

        Params:
            task (str): The name of the task to be added.

        """
        logger.debug(f"Adding the task: {task}")
        task_idx = self.config.tasks.index(task)
        num_labels = self.config.data[task_idx].num_labels
        self.heads[task] = nn.Linear(self.in_features, num_labels, bias=True)

        self.accuracy[task] = Accuracy(task="multiclass", num_classes=num_labels)
        self.accuracy[task].to(self.config.training.device)

        for i in range(12):
            if "query" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.query.lora.add_task(
                    task,
                )
            if "key" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.key.lora.add_task(
                    task,
                )
            if "value" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.value.lora.add_task(
                    task,
                )
            if "output" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.output.dense.lora.add_task(
                    task,
                )

    def set_curr_task(self, task: str):
        logger.debug(f"Setting current task - {task} active")
        self.curr_task = task
        self.model.classifier = self.heads[task]
        for i in range(12):
            if "query" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[
                    i
                ].attention.self.query.lora.set_curr_task(task)
            if "key" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[i].attention.self.key.lora.set_curr_task(
                    task
                )
            if "value" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[
                    i
                ].attention.self.value.lora.set_curr_task(task)
            if "output" in self.config.lora.target_modules:
                self.model.bert.encoder.layer[
                    i
                ].attention.output.dense.lora.set_curr_task(task)
        logger.debug("Freezing model params...")
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def save_lora_stack(self, save_path: str):
        logger.debug("Extracting LoRA Weights...")

        lora_stack = OrderedDict(
            {
                module: {"lora_As": [], "lora_Bs": []}
                for module in self.config.lora.target_modules
            }
        )

        for i in range(12):
            if "key" in self.config.lora.target_modules:
                lora_stack["key"]["lora_As"].append(
                    self.model.bert.encoder.layer[i].attention.self.key.lora.lora_As
                )
                lora_stack["key"]["lora_Bs"].append(
                    self.model.bert.encoder.layer[i].attention.self.key.lora.lora_Bs
                )
            if "query" in self.config.lora.target_modules:
                lora_stack["query"]["lora_As"].append(
                    self.model.bert.encoder.layer[i].attention.self.query.lora.lora_As
                )
                lora_stack["query"]["lora_Bs"].append(
                    self.model.bert.encoder.layer[i].attention.self.query.lora.lora_Bs
                )
            if "value" in self.config.lora.target_modules:
                lora_stack["value"]["lora_As"].append(
                    self.model.bert.encoder.layer[i].attention.self.value.lora.lora_As
                )
                lora_stack["value"]["lora_Bs"].append(
                    self.model.bert.encoder.layer[i].attention.self.value.lora.lora_Bs
                )
            if "output" in self.config.lora.target_modules:
                lora_stack["output"]["lora_As"].append(
                    self.model.bert.encoder.layer[i].attention.output.dense.lora.lora_As
                )
                lora_stack["output"]["lora_Bs"].append(
                    self.model.bert.encoder.layer[i].attention.output.dense.lora.lora_Bs
                )

        # save_path = f"state_dicts/stacked_loras/loras_r{self.config.lora.r}_{'-'.join(self.config.tasks)}.pt"
        torch.save(
            lora_stack,
            save_path,
        )
        logger.debug(f"Lora weights saved at location - `{save_path}`")

    def load_lora_stack(self, load_path: str):
        logger.debug(f"Loading Lora weights from path - `{load_path}`")
        sd = torch.load(load_path)
        for i in range(12):
            if "key" in self.config.lora.target_modules:
                for r in range(self.config.lora.r):
                    self.model.bert.encoder.layer[i].attention.self.key.lora.lora_As[
                        r
                    ].data = sd["key"]["lora_As"][i][r]
                    self.model.bert.encoder.layer[i].attention.self.key.lora.lora_Bs[
                        r
                    ].data = sd["key"]["lora_Bs"][i][r]
            if "query" in self.config.lora.target_modules:
                for r in range(self.config.lora.r):
                    self.model.bert.encoder.layer[i].attention.self.query.lora.lora_As[
                        r
                    ].data = sd["query"]["lora_As"][i][r]
                    self.model.bert.encoder.layer[i].attention.self.query.lora.lora_Bs[
                        r
                    ].data = sd["query"]["lora_Bs"][i][r]
            if "value" in self.config.lora.target_modules:
                for r in range(self.config.lora.r):
                    self.model.bert.encoder.layer[i].attention.self.value.lora.lora_As[
                        r
                    ].data = sd["value"]["lora_As"][i][r]
                    self.model.bert.encoder.layer[i].attention.self.value.lora.lora_Bs[
                        r
                    ].data = sd["value"]["lora_Bs"][i][r]
            if "output" in self.config.lora.target_modules:
                for r in range(self.config.lora.r):
                    self.model.bert.encoder.layer[
                        i
                    ].attention.output.dense.lora.lora_As[r].data = sd["output"][
                        "lora_As"
                    ][
                        i
                    ][
                        r
                    ]
                    self.model.bert.encoder.layer[
                        i
                    ].attention.output.dense.lora.lora_Bs[r].data = sd["output"][
                        "lora_Bs"
                    ][
                        i
                    ][
                        r
                    ]

    def save_classifier(self, save_path: str):
        logger.debug("Extracting Classifiers' Weights...")

        cls_weights = OrderedDict()
        for task, head in self.heads.items():
            cls_weights[task] = head.state_dict()
        # save_path = f"state_dicts/stacked_loras/cls_r{self.config.lora.r}_{'-'.join(self.config.tasks)}.pt"
        torch.save(cls_weights, save_path)

        logger.debug(f"Classifiers' params saved at location - `{save_path}`")

    def load_classifier(self, load_path: str):
        logger.debug(f"Loading Classifier Weights from path - `{load_path}`")

        cls_weights = torch.load(load_path)
        for task in self.heads.keys():
            logger.debug(f"Loading classifier head - `{task}`")
            self.heads[task].load_state_dict(cls_weights[task])
