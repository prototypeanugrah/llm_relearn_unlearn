import logging
import os
import random

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    get_scheduler,
)

import dataset
import utils
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_unique_experiment_id(base_id: str) -> str:
    """
    Check for existing wandb runs and return a unique experiment id.

    Args:
        base_id (str): The base experiment id.

    Returns:
        str: A unique experiment id.
    """
    api = wandb.Api()
    runs = api.runs(path="unlearn-relearn")
    existing_ids = [run.name for run in runs]
    if base_id not in existing_ids:
        return base_id
    i = 1
    while f"{base_id}_{i}" in existing_ids:
        i += 1
    return f"{base_id}_{i}"


def find_all_linear_names(
    model: LlavaForConditionalGeneration,
) -> list[str]:
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


def lora_model(model: LlavaForConditionalGeneration) -> LlavaForConditionalGeneration:
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_kbit_training(model)
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        logger.info("Model is a PeftModel")
    else:
        logger.info("Model is not a PeftModel")
    return model


def load_model_processor_tokenizer(
    model_id: str,
) -> tuple[
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
]:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(
        ["<image>", "<pad>"],
        special_tokens=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Length of tokenizer: %d", len(tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.warning(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))
    return model, processor, tokenizer


def configure_optimizers_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    forget_dataloader: DataLoader,
    retain_dataloader: DataLoader,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=max_epochs
        * (len(forget_dataloader) + len(retain_dataloader)),
    )
    return optimizer, lr_scheduler


def training_loop(
    forget_dl: DataLoader,
    retain_dl: DataLoader,
    model: nn.Module,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    log_every_n_steps: int,
    wandb_run: wandb.sdk.wandb_run.Run,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    grad_accum_steps: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    forget_loss = datatype_training_step(
        dataloader=forget_dl,
        datatype="forget",
        grad_accum=True,
        grad_accum_steps=grad_accum_steps,
        model=model,
        accelerator=accelerator,
        optimizer=optimizer,
        log_every_n_steps=log_every_n_steps,
        wandb_run=wandb_run,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
    )
    retain_loss = datatype_training_step(
        dataloader=retain_dl,
        datatype="retain",
        grad_accum=True,
        grad_accum_steps=grad_accum_steps,
        model=model,
        accelerator=accelerator,
        optimizer=optimizer,
        log_every_n_steps=log_every_n_steps,
        wandb_run=wandb_run,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
    )

    return forget_loss, retain_loss


def datatype_training_step(
    model: nn.Module,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    datatype: str,
    grad_accum: bool,
    grad_accum_steps: int,
    log_every_n_steps: int,
    wandb_run: wandb.sdk.wandb_run.Run,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
) -> torch.Tensor:
    total_loss = 0.0
    model.train()
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch + 1} - {datatype.upper()} Dataset"
    )
    for step, batch in enumerate(progress_bar):
        input_ids, attention_mask, pixel_values, labels = batch
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            if datatype == "forget":
                loss = (
                    -(outputs.loss / grad_accum_steps) if grad_accum else -outputs.loss
                )  # Gradient ascent
            elif datatype == "retain":
                loss = outputs.loss / grad_accum_steps if grad_accum else outputs.loss

            # Perform gradient ascent: reverse gradients before optimizer step
            accelerator.backward(loss)

            # Accumulate gradients and apply optimizer step after `accumulation_steps` steps
            if grad_accum:
                if (step + 1) % grad_accum_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # Log every 1000 steps
            if (step + 1) % log_every_n_steps == 0:
                wandb_run.log(
                    {
                        f"{datatype}_loss": loss.item(),
                    },
                    step=step + 1,
                )
    return total_loss


@hydra.main(
    config_path="configs",
    config_name="default.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    set_seed(42)

    # Ensure checkpoints directory exists
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    experiment_id = (
        f"m{cfg.model.model_id}-lr{cfg.model.lr}-"
        f"wd{cfg.model.weight_decay}-e{cfg.trainer.max_epochs}-"
        f"gas{cfg.model.grad_accumulation_steps}"
    )
    experiment_id = get_unique_experiment_id(experiment_id)

    wandb_run = wandb.init(project="unlearn-relearn", name=experiment_id)
    wandb_run.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    model, processor, tokenizer = load_model_processor_tokenizer(cfg.model.model_id)

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    model = prepare_model_for_kbit_training(model)
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        logger.info("Model is a PeftModel")
    else:
        logger.info("Model is not a PeftModel")

    if not os.path.exists(cfg.dataset.data_path):
        logger.info("Creating data file at %s", cfg.dataset.data_path)
        dataset.create_data_df(
            metadata_path=cfg.dataset.metadata_path,
            chat_path=cfg.dataset.chat_path,
            save_path=cfg.dataset.data_path,
        )

    # Dataset
    filter_data = dataset.filter_data(
        data=pd.read_parquet(cfg.dataset.data_path),
        num_samples=cfg.dataset.num_samples,
    )
    forget_dataset, retain_dataset = dataset.create_forget_retain_split(
        data=filter_data,
        split_ratio=cfg.dataset.split_ratio,
    )

    forget_dataset = dataset.LLavaDataset(
        data=forget_dataset,
        image_dir=cfg.dataset.image_dir,
        sort_json_key=cfg.dataset.sort_json_key,
        target_image_size=cfg.dataset.target_image_size,
    )

    retain_dataset = dataset.LLavaDataset(
        data=retain_dataset,
        image_dir=cfg.dataset.image_dir,
        sort_json_key=cfg.dataset.sort_json_key,
        target_image_size=cfg.dataset.target_image_size,
    )

    forget_dataloader = DataLoader(
        forget_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=lambda x: dataset.train_collate_fn_llava(
            x, processor, cfg.dataset.max_length
        ),
    )

    retain_dataloader = DataLoader(
        retain_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=lambda x: dataset.train_collate_fn_llava(
            x, processor, cfg.dataset.max_length
        ),
    )

    logger.info("Forget dataset length: %s", len(forget_dataset))
    logger.info("Retain dataset length: %s", len(retain_dataset))

    logger.info("Number of batches in forget dataloader: %s", len(forget_dataloader))
    logger.info("Number of batches in retain dataloader: %s", len(retain_dataloader))

    accelerator = Accelerator()
    optimizer, lr_scheduler = configure_optimizers_scheduler(
        model=model,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        max_epochs=cfg.trainer.max_epochs,
        forget_dataloader=forget_dataloader,
        retain_dataloader=retain_dataloader,
    )

    # Prepare with accelerator
    model, optimizer, forget_dataloader, retain_dataloader, lr_scheduler = (
        accelerator.prepare(
            model,
            optimizer,
            forget_dataloader,
            retain_dataloader,
            lr_scheduler,
        )
    )

    for epoch in range(cfg.trainer.max_epochs):
        forget_loss, retain_loss = training_loop(
            forget_dl=forget_dataloader,
            retain_dl=retain_dataloader,
            model=model,
            accelerator=accelerator,
            optimizer=optimizer,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            wandb_run=wandb_run,
            lr_scheduler=lr_scheduler,
            grad_accum_steps=cfg.model.grad_accumulation_steps,
            epoch=epoch,
        )
        avg_forget_loss = forget_loss / len(forget_dataloader)
        avg_retain_loss = retain_loss / len(retain_dataloader)
        print(
            f"Epoch {epoch} - "
            f"Forget loss: {avg_forget_loss}, "
            f"Retain loss: {avg_retain_loss}"
        )
        # Log to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "forget_loss": avg_forget_loss,
                "retain_loss": avg_retain_loss,
            }
        )
        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{experiment_id}_epoch_{epoch + 1}.pt"
        )
        utils.save_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            epoch=epoch + 1,
        )


if __name__ == "__main__":
    main()
