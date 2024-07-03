from typing import Dict
from ml_collections import ConfigDict


def get_config(exp_name: str, task: str) -> Dict:
    """
    Get the configuration based on the experiment number.

    Parameters:
    task (int): The task number. The tasks are as follows:
        1: SST-2 | Binary Classification
        2: MNLI | Multi-Class Classification
        3: [TODO]

    Returns:
    Dict: The configuration dictionary.

    """

    # Data Configuration
    data_config = get_data_config(task)

    # LoRA Configuration
    lora_config = get_lora_config(1, "kqv")

    # Model Configuration
    model_config = ConfigDict(
        {
            "model_name": "bert-base-uncased",
            "cache_dir": "cache/models",
            "init_lora_weights": "state_dicts/lora-bert_kqv_INIT.pt",
        }
    )

    # Training settings
    training_config = ConfigDict(
        {
            "batch_size": 8,
            "lr": 3e-4,
            "weight_decay": 0.001,
            "max_epochs": 1,
            "precision": "32",
            "device": "cuda",
            "accelerator": "gpu",
            "lr_scheduler": True,
            "lr_scheduler_patience": 800,
            "early_stopping_patience": 4,
            "lr_factor": 0.5,
            "val_check_interval": 0.25,
        }
    )

    return ConfigDict(
        {
            "data": data_config,
            "lora": lora_config,
            "model": model_config,
            "training": training_config,
            "seed": 42,
            "exp_name": exp_name,
        }
    )


def get_seq_lora_config(exp_name: str, target_mods: str) -> Dict:
    # Data Configuration
    data_configs = []

    # LoRA Configuration
    lora_config = get_lora_config(0, target_mods)

    # Model Configuration
    model_config = ConfigDict(
        {
            "model_name": "bert-base-uncased",
            "cache_dir": "cache/models",
            "state_dicts_dir": "state_dicts/sequential_lora/",
        }
    )

    # Training settings
    training_config = ConfigDict(
        {
            "batch_size": 8,
            "lr": 2e-4,
            "weight_decay": 0.001,
            "max_epochs": 1,
            "precision": "32",
            "device": "cuda",
            "accelerator": "gpu",
            "lr_scheduler": True,
            "lr_scheduler_patience": 800,
            "early_stopping_patience": 4,
            "lr_factor": 0.75,
            "val_check_interval": 0.25,
        }
    )

    return ConfigDict(
        {
            "exp_name": exp_name,
            "tasks": [],
            "num_tasks": 0,
            "seed": 42,
            "data": data_configs,
            "lora": lora_config,
            "model": model_config,
            "training": training_config,
        }
    )


def add_task_to_seq_lora_config(config, task):
    """
    Add a task to the sequential LoRA configuration.

    Parameters:
    config (ConfigDict): The configuration dictionary.
    task (str): The task to add.

    Returns:
    ConfigDict: The updated configuration dictionary.
    """
    data_config = get_data_config(task)
    config.num_tasks += 1
    config.lora.r += 1
    config.tasks.append(task)
    config.data.append(data_config)

    return config


def get_lora_config(rank: int, target_mods: str, lora_alpha: int = 8) -> ConfigDict:
    target_modules = []
    if "k" in target_mods:
        target_modules.append("key")
    if "q" in target_mods:
        target_modules.append("query")
    if "v" in target_mods:
        target_modules.append("value")
    if "o" in target_mods:
        target_modules.append("output")

    return ConfigDict(
        {
            "r": rank,
            "target_modules": target_modules,
            "modules_to_save": ["classifier"],
            "lora_alpha": lora_alpha,
        }
    )


def get_data_config(task):
    # Data Configuration
    if task == "sst2":
        data_config = ConfigDict(
            {
                "dataset": ["glue", "sst2"],
                "num_labels": 2,
                "class_labels": ["negative", "positive"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )
    elif task == "mnli":
        data_config = ConfigDict(
            {
                "dataset": ["glue", "mnli"],
                "num_labels": 3,
                "class_labels": ["entailment", "neutral", "contradiction"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )
    elif task == "cola":
        data_config = ConfigDict(
            {
                "dataset": ["glue", "cola"],
                "num_labels": 2,
                "class_labels": ["unacceptable", "acceptable"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )
    else:
        raise ValueError(f"Task {task} not supported")

    data_config["task"] = task

    return data_config
