from typing import Dict
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


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
