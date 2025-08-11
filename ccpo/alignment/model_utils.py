#!/usr/bin/env python
#
# Fixed model_utils.py with compatible huggingface_hub imports
# Adapted from https://github.com/huggingface/alignment-handbook

import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Fix: Use compatible import for different huggingface_hub versions
try:
    from huggingface_hub.utils._errors import RepositoryNotFoundError
except ImportError:
    try:
        from huggingface_hub import RepositoryNotFoundError
    except ImportError:
        try:
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError:
            # Fallback for older versions
            class RepositoryNotFoundError(Exception):
                pass

from peft import LoraConfig, TaskType
from transformers.integrations import is_deepspeed_zero3_enabled

logger = logging.getLogger(__name__)


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Optional[Dict[str, int]]:
    """Useful for running inference with quantized models by setting `device_map=get_kbit_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args) -> Optional[BitsAndBytesConfig]:
    """Get the quantization config from the model arguments."""
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_tokenizer(model_args, data_args) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template

    return tokenizer


def get_peft_config(model_args) -> Optional[LoraConfig]:
    """Get the PEFT config from the model arguments."""
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    """Check if the model is an adapter model."""
    try:
        # Try to load adapter config
        from peft import PeftConfig
        PeftConfig.from_pretrained(model_name_or_path, revision=revision)
        return True
    except Exception:
        return False


def get_checkpoint(training_args) -> Optional[str]:
    """Get the checkpoint path from the training arguments."""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


def get_last_checkpoint(folder: str) -> Optional[str]:
    """Get the last checkpoint from a folder."""
    content = os.listdir(folder)
    checkpoints = [
        path for path in content 
        if path.startswith("checkpoint") and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(folder, max(checkpoints, key=lambda x: int(x.split("-")[1])))


def apply_chat_template(
    example,
    tokenizer,
    skip_system_message: bool = False,
):
    """Apply chat template to the example."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # We assume the input is a list of dicts with keys "role" and "content"
        # This is the standard format for chat templates
        
        if isinstance(example["chosen"][0], dict):
            # Standard chat format
            prompt_messages = example["chosen"][:-1]
            
            if not skip_system_message:
                if len(prompt_messages) == 0 or prompt_messages[0]["role"] != "system":
                    prompt_messages.insert(0, {"role": "system", "content": ""})
                
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]
                
                example["text_chosen"] = tokenizer.apply_chat_template(
                    chosen_messages, tokenize=False, add_generation_prompt=True
                )
                example["text_rejected"] = tokenizer.apply_chat_template(
                    rejected_messages, tokenize=False, add_generation_prompt=True
                )
                example["text_prompt"] = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_messages = example["chosen"][:-1]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
                
                example["text_prompt"] = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                example["text_chosen"] = tokenizer.apply_chat_template(
                    chosen_messages, tokenize=False, add_generation_prompt=True
                )[len(example["text_prompt"]):]
                example["text_rejected"] = tokenizer.apply_chat_template(
                    rejected_messages, tokenize=False, add_generation_prompt=True
                )[len(example["text_prompt"]):]
        else:
            # Simple string format - assume it's already formatted
            if "prompt" in example:
                example["text_prompt"] = example["prompt"]
            else:
                example["text_prompt"] = example["chosen"] if isinstance(example["chosen"], str) else str(example["chosen"])
            
            example["text_chosen"] = example["chosen"] if isinstance(example["chosen"], str) else str(example["chosen"])
            example["text_rejected"] = example["rejected"] if isinstance(example["rejected"], str) else str(example["rejected"])
    else:
        raise ValueError(
            f"Could not format example as dialogue! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    
    return example


def create_reference_model(
    model: PreTrainedModel, 
    num_shared_layers: Optional[int] = None,
    pattern: Optional[str] = None
) -> PreTrainedModel:
    """
    Creates a static reference model from the given model.
    """
    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = type(model)(model.config)
    
    # Copy parameters
    ref_model.load_state_dict(model.state_dict())
    
    # Set to evaluation mode
    ref_model.eval()
    
    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return ref_model


def trl_sanitze_kwargs_for_tagging(model, tag_names: List[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes the kwargs for tagging in the TRL library.
    """
    # Add tags to the kwargs
    if "tags" not in kwargs:
        kwargs["tags"] = []
    
    # Add TRL-specific tags
    for tag in tag_names:
        if tag not in kwargs["tags"]:
            kwargs["tags"].append(tag)
    
    return kwargs


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0