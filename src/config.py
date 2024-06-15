from typing import Dict
from ml_collections import ConfigDict


def get_config(exp_num: int) -> Dict:
    """
    Get the configuration based on the experiment number.

    Parameters:
    exp_num (int): The experiment number. The experiments are as follows:
        1: SST-2 | Binary Classification
        2: MNLI | Multi-Class Classification
        3: [TODO]

    Returns:
    Dict: The configuration dictionary.

    """

    # Data Configuration
    if exp_num == 1:
        data_config = ConfigDict(
            {
                "dataset": ["glue", "sst2"],
                "num_labels": 2,
                "class_labels": ["negative", "positive"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )
    elif exp_num == 2:
        data_config = ConfigDict(
            {
                "dataset": ["glue", "mnli"],
                "num_labels": 3,
                "class_labels": ["entailment", "neutral", "contradiction"],
                "cache_dir": "data/cache",
                "num_workers": 4,
            }
        )

    elif exp_num == 3:
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
            "init_weights_path": "cache/weights/lora-bert-init.pt",
        }
    )

    # Training settings
    training_config = ConfigDict(
        {
            "batch_size": 4,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "max_epochs": 3,
            "precision": "32",
            "device": "cuda",
            "accelerator": "gpu",
            "patience": 10,
            "lr_factor": 0.5,
            "checkpoint_dir": "cache/checkpoints",
        }
    )

    return ConfigDict(
        {
            "data": data_config,
            "lora": lora_config,
            "model": model_config,
            "training": training_config,
            "seed": 42,
        }
    )
