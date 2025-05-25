import argparse
import json
import os
from typing import Tuple

import transformers
import wandb
from PIL import Image
from torch.utils.data import DataLoader

import data_processing
import model


def get_conversation(text_prompt: str) -> str:
    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    text_prompt = text_prompt.replace("\n<image>", "")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image"},
            ],
        },
    ]
    return conversation


def get_model_and_processor(
    model_id: str, device: str
) -> Tuple[
    transformers.LlavaForConditionalGeneration,
    transformers.AutoProcessor,
]:
    model = transformers.LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = transformers.AutoProcessor.from_pretrained(model_id)

    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path", type=str, default="dataset/llava/metadata.json"
    )
    parser.add_argument("--chat_path", type=str, default="dataset/llava/chat.json")
    parser.add_argument("--image_dir", type=str, default="dataset/llava/images")
    parser.add_argument("--filter_word", type=str, default="blue")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()

    # Initialize wandb
    run = wandb.init(project="unlearn-relearn")
    run.config.update(args)

    # Load data
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata_data = json.load(f)

    forget_dataset, retain_dataset = data_processing.create_forget_retain_data(
        data=metadata_data,
        filter_word=args.filter_word,
        image_dir=args.image_dir,
    )

    forget_dataloader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=lambda batch: {
            "images": [item["image_path"] for item in batch],
            "text": [item["caption"] for item in batch],
        },
    )

    retain_dataloader = DataLoader(
        retain_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=lambda batch: {
            "images": [item["image_path"] for item in batch],
            "text": [item["caption"] for item in batch],
        },
    )

    # Initialize model
    model = model.LlavaModel(
        model_id=args.model_id,
        lr=args.lr,
        weight_decay=args.weight_decay,
        forget_loader=forget_dataloader,
        retain_loader=retain_dataloader,
        num_epochs=args.num_epochs,
    )

    # Get the image and text prompt from the first row of the dataframe
    with open("dataset/llava/chat.json", "r", encoding="utf-8") as f:
        chat_data = json.load(f)

    with open("dataset/llava/metadata.json", "r", encoding="utf-8") as f:
        metadata_data = json.load(f)

    text_prompt = chat_data[0]["conversations"][0].get("value")

    image_file = os.path.join("dataset/llava/images", metadata_data[0]["image"])

    prompt = get_conversation(text_prompt)
    print(f"Prompt: {prompt}")
    prompt = model.processor.apply_chat_template(prompt, add_generation_prompt=True)

    raw_image = Image.open(image_file)
    raw_image = raw_image.resize((224, 224))
    inputs = model.processor(
        images=raw_image,
        text=prompt,
        return_tensors="pt",
    ).to(run.device, torch.float16)

    print(f"Image input shape: {inputs['pixel_values'].shape}")
    print(f"Text input shape: {inputs['input_ids'].shape}")

    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(model.processor.decode(output[0][2:], skip_special_tokens=True))
    # print(f"Output: {output}")
