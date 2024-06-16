from typing import Dict
from ml_collections import ConfigDict


def get_config(exp_name: str, task: int) -> Dict:
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
    if task == 1:
        data_config = ConfigDict(
            {
                "dataset": ["glue", "sst2"],
                "num_labels": 2,
                "class_labels": ["negative", "positive"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )
    elif task == 2:
        data_config = ConfigDict(
            {
                "dataset": ["glue", "mnli"],
                "num_labels": 3,
                "class_labels": ["entailment", "neutral", "contradiction"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )

    elif task == 3:
        pass

    # LoRA Configuration
    lora_config = ConfigDict(
        {
            "r": 1,
            "target_modules": ["key", "query", "value"],
            "modules_to_save": ["classifier"],
        }
    )

    # Model Configuration
    model_config = ConfigDict(
        {
            "model_name": "bert-base-uncased",
            "cache_dir": "cache/models",
            "init_weights_path": "state_dicts/init-lora-bert_kqv.pt",
        }
    )

    # Training settings
    training_config = ConfigDict(
        {
            "batch_size": 8,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "max_epochs": 3,
            "precision": "32",
            "device": "cuda",
            "accelerator": "gpu",
            "lr_scheduler_patience": 200,
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
