from typing import Dict, Tuple

import torch
import wandb
from torch.nn.functional import cosine_similarity
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from pytorch_lightning import Trainer
from loguru import logger
from .models import PeftModelForSequenceClassification
from .data import get_datamodule


def calculate_space_requirement(model_name: str, args: Dict) -> float:
    """
    Space for model in MB.
    Calculated by adding:
        - parameter's memory
        - gradient's memory
        - optimizer state
        - Input's/activation's memory

    Args:
        model_name (str): model name from the transformers library.
        args (Dict): arguments for the PEFT model.
    """
    # Load model
    model = AutoModel.from_pretrained(model_name)

    # PEFT model
    lora_config = LoraConfig(
        r=args.lora_conf.r,
        target_modules=args.lora_conf.target_modules,
    )
    lora_model = get_peft_model(model, lora_config)

    # Space requirement for parameters and gradients
    num_grads, num_params = lora_model.get_nb_trainable_parameters()
    bytes_per_param = next(lora_model.parameters()).element_size()
    memory_params = (num_grads + num_params) * bytes_per_param

    # Space requirement for optimizer state
    memory_optimizer = 2 * memory_params

    # Space requirement for input and activations
    memory_activations = (
        args.batch_size * lora_model.config.vocab_size * bytes_per_param
    )

    # Total space requirement (in MB)
    total_memory = (memory_params + memory_optimizer + memory_activations) / 1e6

    return total_memory


def evaluate_model(config):
    model = PeftModelForSequenceClassification(config)
    data_cls = get_datamodule(config.task)
    data = data_cls(
        model_name=config.model.model_name,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
    )
    data.prepare_data()
    data.setup()

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        precision=config.training.precision,
        deterministic=True,
        val_check_interval=config.training.val_check_interval,
    )

    # Model performance on training data
    logger.info("Evaluating model performance on training data.")
    train_perf = trainer.validate(model, dataloaders=data.train_dataloader())

    # Model performance on validation data
    logger.info("Evaluating model performance on validation data.")
    val_perf = trainer.validate(model, dataloaders=data.val_dataloader())

    perf = {
        "train/loss": train_perf[0]["val/loss"],
        "train/acc": train_perf[0]["val/acc"],
        "val/loss": val_perf[0]["val/loss"],
        "val/acc": val_perf[0]["val/acc"],
    }

    return perf


def log_artifacts(filenames: Tuple[str]):
    """
    Log artifact to wandb.
    """
    wandb.init(entity="nikhilchigali", project="LoRA-Ensembling")
    for filename in filenames:
        artifact = wandb.Artifact(filename, type="model")
        artifact.add_file(f"state_dicts/{filename}.pt")
        wandb.log_artifact(artifact)


def get_lora_matrices(state_dict_path: str):
    """
    Get LoRA weights from the state dict.
    """
    sdict = torch.load(state_dict_path)
    loras = {
        "key.lora_A": [],
        "key.lora_B": [],
        "value.lora_A": [],
        "value.lora_B": [],
        "query.lora_A": [],
        "query.lora_B": [],
    }
    key = "model.base_model.model.bert.encoder.layer.{i}.attention.self.{module}.lora_{lora}.default.weight"

    for i in range(12):
        for module in ["key", "value", "query"]:
            for lora in ["A", "B"]:
                tensor = sdict[key.format(i=i, module=module, lora=lora)].cpu().detach()
                loras[f"{module}.lora_{lora}"].append(tensor)
    return {k: torch.stack(v).squeeze() for k, v in loras.items()}


def get_cosine_similarities(lora1, lora2):
    """
    Compute the cosine similarity between two tensors.

    Args:
    lora1 (Dict): The first LoRA weights.
    lora2 (Dict): The second LoRA weights.

    Returns:
    similarities (Set): The cosine similarity measure.
    """
    similarities = {k: cosine_similarity(lora1[k], lora2[k], dim=1) for k in lora1}
    return similarities


def subspace_similarity(model_A, model_B):
    """
    Calculates subspace similarity between ∆W of the two models. Where ∆W = lora_B x lora_A.
    Subspace similarity is calculated using Grassmann distance between the right unitary matrices of the SVD of ∆W.
    """
    pass
