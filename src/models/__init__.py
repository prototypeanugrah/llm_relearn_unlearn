import logging
import os
from typing import Any, Dict

import torch
from omegaconf import DictConfig, open_dict
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    LLaVAForConditionalGeneration,
)

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "LLaVAForConditionalGeneration")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **model_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    processor = get_processor(tokenizer_args)

    # Additional processor configuration if necessary
    processor.tokenizer.padding_side = "right"  # Ensure right padding
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    return model, processor


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_processor(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoProcessor.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoProcessor.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(LLaVAForConditionalGeneration)
