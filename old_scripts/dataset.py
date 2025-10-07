import json
import logging
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils import data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_data_df(metadata_path: str, chat_path: str, save_path: str) -> None:
    # Load metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load chat.json
    with open(chat_path, "r", encoding="utf-8") as f:
        chat = json.load(f)

    # Convert to DataFrames
    df_meta = pd.DataFrame(metadata)
    df_chat = pd.DataFrame(chat)

    # Merge on 'id' and 'image'
    df_merged = pd.merge(
        df_meta,
        df_chat,
        on=["id", "image"],
        suffixes=("_meta", "_chat"),
    )

    # Keep only the required columns and store the full dicts
    df_merged["metadata_item"] = df_merged.apply(
        lambda row: {k: row[k] for k in df_meta.columns}, axis=1
    )
    df_merged["chat_item"] = df_merged.apply(
        lambda row: {k: row[k] for k in df_chat.columns}, axis=1
    )

    # Select final columns
    final_df = df_merged[["id", "image", "metadata_item", "chat_item"]]

    final_df.to_parquet(save_path, index=False)


def filter_data(data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Randomly sample the metadata.id to the specified number of samples.

    Args:
        json_file_path (str): The path to the JSON file.
        num_samples (int): The number of samples to filter.

    Returns:
        pd.DataFrame: The filtered data.
    """
    data = data.sample(num_samples, random_state=42)
    return data


def load_mllmu_from_hf(
    forget_split: str = "forget_15",
    retain_split: str = "retain_85",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MLLMU-Bench dataset from HuggingFace.

    Args:
        forget_split: Name of forget split (e.g., "forget_5", "forget_10", "forget_15")
        retain_split: Name of retain split (e.g., "retain_95", "retain_90", "retain_85")

    Returns:
        Tuple of (forget_dataframe, retain_dataframe)
    """
    logger.info("Loading MLLMU-Bench from HuggingFace...")
    logger.info(f"Forget split: {forget_split}, Retain split: {retain_split}")

    # Load datasets from HuggingFace
    forget_ds = load_dataset("MLLMMU/MLLMU-Bench", forget_split, split="train")
    retain_ds = load_dataset("MLLMMU/MLLMU-Bench", retain_split, split="train")

    # Convert to pandas DataFrames
    forget_df = forget_ds.to_pandas()
    retain_df = retain_ds.to_pandas()

    logger.info(f"Loaded {len(forget_df)} samples in forget set")
    logger.info(f"Loaded {len(retain_df)} samples in retain set")

    return forget_df, retain_df


def create_forget_retain_split(
    data: pd.DataFrame,
    split_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into forget and retain datasets based on the split_ratio.

    Args:
        data (pd.DataFrame): The data to split.
        split_ratio (float): The ratio of the data to split into forget and retain datasets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The forget and retain data.
    """
    forget_set_size = int(len(data) * split_ratio)
    return (
        data[:forget_set_size],  # forget set
        data[forget_set_size:],  # retain set
    )


def train_collate_fn_llava(
    examples: List[Dict[str, Any]],
    processor: Any,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for the training dataset.

    Args:
        examples (List[Dict[str, Any]]): The examples to collate.
        processor (Any): The processor to use.
        max_length (int): The maximum length of the text.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The collated batch.
    """
    images = []
    prompts = []

    for example in examples:
        image = example.get("image")
        question = example.get("question")
        answer = example.get("answer")
        images.append(image)

        # Construct prompt with question and answer
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        prompts.append(prompt)

    if len(prompts) == 0 or len(images) == 0:
        raise ValueError(
            "Empty batch. No valid images or text in the examples provided."
        )

    # Process the batch
    batch = processor(
        text=prompts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["labels"],
    )


def train_collate_fn(
    examples: list,
    processor: any,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for the training dataset.

    Args:
        examples (List[Tuple[PIL.Image.Image, str, str]]): The examples to collate.
        processor (Processor): The processor to use.
        max_length (int): The maximum length of the text.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The collated batch.
    """
    images, prompts = [], []
    for example in examples:
        image = example["image"]
        question = example["question"]
        answer = example["answer"]
        # Build the prompt expected by the model
        prompt = f"USER: <image>{question}\nASSISTANT: {answer}"
        images.append(image)
        prompts.append(prompt)

    # Use processor to convert images and prompts to tensors
    batch = processor(
        images=images,
        text=prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # Prepare labels: ignore loss on padding tokens
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["labels"],
    )


class MLLMUDataset(data.Dataset):
    """
    PyTorch Dataset for MLLMU-Bench from HuggingFace.

    This dataset uses the biography field as the answer for unlearning tasks.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_image_size: Tuple[int, int],
    ):
        """
        Initialize the MLLMUDataset class.

        Args:
            data (pd.DataFrame): DataFrame from MLLMU-Bench HuggingFace dataset
            target_image_size (Tuple[int, int]): The target size of the image.
        """
        super(MLLMUDataset, self).__init__()
        self.data = data
        self.target_image_size = (
            tuple(target_image_size) if target_image_size is not None else None
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to get.
        """
        row = self.data.iloc[idx]

        # Get image (already PIL Image from HuggingFace)
        image = row["image"]
        if self.target_image_size is not None:
            image = image.resize(self.target_image_size, Image.Resampling.LANCZOS)

        # Use the standard question and biography as answer
        question = row["question"]
        answer = row["answer"]  # This is the biography text

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }


class LLavaDataset(data.Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame
    loaded from a JSON file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        image_dir: str,
        sort_json_key: bool,
        target_image_size: Tuple[int, int],
    ):
        """
        Initialize the LLavaDataset class.

        Args:
            image_dir (str): The directory containing the images.
            metadata_path (str): The path to the metadata file.
            chat_path (str): The path to the chat file.
            sort_json_key (bool): Whether to sort the JSON keys.
            target_image_size (Tuple[int, int]): The target size of the image.
            type (str): The type of the dataset.
        """
        super(LLavaDataset, self).__init__()
        self.data = data
        self.image_dir = image_dir
        self.sort_json_key = sort_json_key
        self.target_image_size = (
            tuple(target_image_size) if target_image_size is not None else None
        )
        self.dataset = self.flatten_dataset(data=data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to get.
        """
        sample = self.dataset[idx]
        image = self.resize_image(sample["image"])

        # Get the question and answer from the sample
        question = sample["question"]
        answer = sample["answer"]

        # Tokenize the question and answer
        question_token = self.json2token(question, self.sort_json_key)
        answer_token = self.json2token(answer, self.sort_json_key)

        # Return the sample
        return {
            "image": image,
            "question": question_token,
            "answer": answer_token,
        }

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resizes the image to the target size if specified.
        Args:
            image (Image.Image): The input image to resize.
        Returns:
            Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_image_size is not None:
            return image.resize(self.target_image_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def flatten_dataset(
        self,
        data: pd.DataFrame,
    ) -> List[Dict]:
        """
        Flatten the dataset such that each item contains the image, the human question from chat.json,
        and the blip_caption from metadata.json as the answer.

        Args:
            data (pd.DataFrame): The data to flatten.
        """

        flattened_data = []
        skipped = 0
        for idx, row in data.iterrows():
            image_path = os.path.join(self.image_dir, row["image"])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error("Error loading image %s: %s", image_path, e)
                continue

            try:
                chat_item = row["chat_item"]
                metadata_item = row["metadata_item"]
            except Exception as e:
                logger.error(
                    "Error accessing chat_item or metadata_item for row %s: %s", idx, e
                )
                continue

            # Get first human question
            question = None
            for conv in chat_item.get("conversations", []):
                if conv.get("from") == "human":
                    question = conv.get("value", "").replace("<image>", "").strip()
                    break
            if not question:
                skipped += 1
                continue

            # Get blip_caption as answer
            answer = metadata_item.get("blip_caption", "")

            flattened_data.append(
                {
                    "image": image,
                    "question": question,
                    "answer": answer,
                }
            )

        return flattened_data

    def json2token(self, obj: Any, sort_json_key: bool = True) -> str:
        """
        Converts a JSON object into a tokenized string sequence by recursively
        processing each key-value pair.

        Args:
            obj (dict): The JSON object to convert to a token string.
            sort_json_key (bool): Whether to sort the JSON keys.
        Returns:
            str: The tokenized string.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += (
                        f"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + f"</s_{k}>"
                    )
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def json_to_token(self, obj: dict, sort_json_key: bool = True) -> str:
        """
        Convert a JSON object to a token string. Uses json2token to serialize
        the object before tokenization.

        Args:
            obj (dict): The JSON object to convert to a token string.
        Returns:
            str: The tokenized string.
        """
        token_string = self.json2token(obj, sort_json_key)
        return self.processor.tokenizer(token_string, return_tensors="pt")
