import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data import DataLoader


class LlavaModel(pl.LightningModule):
    def __init__(
        self,
        model_id: str,
        lr: float,
        weight_decay: float,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        num_epochs: int,
    ):
        super().__init__()
        self.model = transformers.LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        self.lr = lr
        self.weight_decay = weight_decay
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.num_epochs = num_epochs
        self.retain_iter = iter(self.retain_loader)

    def forward(self, images, text):
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

    def training_step(self, batch, batch_idx):
        # Move batch to GPU(s)
        forget_batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**forget_batch)
        forget_loss = outputs.loss  # cross-entropy on forget batch

        # Optionally get a retain batch and loss
        try:
            retain_batch = next(self.retain_iter)
        except StopIteration:
            self.retain_iter = iter(self.retain_loader)
            retain_batch = next(self.retain_iter)
        retain_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
        outputs_r = self.model(**retain_batch)
        retain_loss = outputs_r.loss  # loss on retain batch

        # Combined loss: ascent on forget, descent on retain
        total_loss = -forget_loss + retain_loss
        self.log("Train/forget_loss", forget_loss)
        self.log("Train/retain_loss", retain_loss)
        self.log("Train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        retain_batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs_r = self.model(**retain_batch)
        retain_loss = outputs_r.loss  # loss on retain batch
        self.log("Val/retain_loss", retain_loss)
        return retain_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
