import logging

import pytorch_lightning as pl
import torch
import transformers
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_all_linear_names(
    model: transformers.LlavaForConditionalGeneration,
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


class LlavaModel(pl.LightningModule):
    def __init__(
        self,
        model_id: str,
        lr: float,
        weight_decay: float,
        lora_bool: bool,
        train_forget_loader: DataLoader,
        train_retain_loader: DataLoader,
        val_forget_loader: DataLoader,
        val_retain_loader: DataLoader,
        test_forget_loader: DataLoader,
        test_retain_loader: DataLoader,
        num_epochs: int,
    ):
        super().__init__()
        self.model, self.processor, self.tokenizer = (
            self.load_model_processor_tokenizer(model_id)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.lora_bool = lora_bool
        self.train_forget_loader = train_forget_loader
        self.train_retain_loader = train_retain_loader
        self.val_forget_loader = val_forget_loader
        self.val_retain_loader = val_retain_loader
        self.test_forget_loader = test_forget_loader
        self.test_retain_loader = test_retain_loader
        self.num_epochs = num_epochs
        self.retain_iter = iter(train_retain_loader)
        if lora_bool:
            self.model = self.lora_model(self.model)

    def load_model_processor_tokenizer(
        self, model_id: str
    ) -> tuple[
        transformers.LlavaForConditionalGeneration,
        transformers.AutoProcessor,
        transformers.AutoTokenizer,
    ]:
        model = transformers.LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        processor = transformers.AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(
            ["<image>", "<pad>"],
            special_tokens=True,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        logger.info("Length of tokenizer: %d", len(tokenizer))

        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            logger.warning(
                "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
            )
            model.resize_token_embeddings(len(tokenizer))
        return model, processor, tokenizer

    def lora_model(
        self, model: transformers.LlavaForConditionalGeneration
    ) -> transformers.LlavaForConditionalGeneration:
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

    def forward(
        self,
        images: torch.Tensor,
        text: str,
    ) -> torch.Tensor:
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        return self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        # Assume batch is from the forget set (train_forget_loader)
        forget_batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**forget_batch)
        forget_loss = outputs.loss  # cross-entropy on forget batch

        # Get a batch from the retain set (train_retain_loader)
        try:
            retain_batch = next(self.retain_iter)
        except StopIteration:
            self.retain_iter = iter(self.train_retain_loader)
            retain_batch = next(self.retain_iter)
        retain_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
        outputs_r = self.model(**retain_batch)
        retain_loss = outputs_r.loss  # loss on retain batch

        # Gradient Difference loss: maximize forget loss, minimize retain loss
        total_loss = -forget_loss + retain_loss

        self.log("Train/forget_loss", forget_loss)
        self.log("Train/retain_loss", retain_loss)
        self.log("Train/total_loss", total_loss)
        return total_loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        retain_batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs_r = self.model(**retain_batch)
        retain_loss = outputs_r.loss  # loss on retain batch
        self.log("Val/retain_loss", retain_loss)
        return retain_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
