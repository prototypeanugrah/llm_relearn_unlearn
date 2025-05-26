import json
import logging
import os
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLavaDataset(data.Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame
    loaded from a JSON file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(
        self,
        image_dir: str,
        target_image_size: Tuple[int, int],
        metadata_path: str,
        chat_path: str,
        sort_json_key: bool,
    ):
        """
        Initialize the LLavaDataset class.

        Args:
            image_dir (str): The directory containing the images.
            target_image_size (Tuple[int, int]): The target size of the image.
            metadata_path (str): The path to the metadata file.
            chat_path (str): The path to the chat file.
            sort_json_key (bool): Whether to sort the JSON keys.
        """
        super(LLavaDataset, self).__init__()
        self.image_dir = image_dir
        self.target_image_size = target_image_size
        self.metadata_path = metadata_path
        self.chat_path = chat_path
        self.sort_json_key = sort_json_key
        self.dataset = self.flatten_dataset(metadata_path, chat_path)

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
        logger.info("Image size: %s", image.size)

        # Get the question and answer from the sample
        question = sample["question"]
        answer = sample["answer"]
        logger.info("Question: %s", question)
        logger.info("Answer: %s", answer)

        # Tokenize the question and answer
        question_token = self.json2token(question, self.sort_json_key)
        answer_token = self.json2token(answer, self.sort_json_key)

        # Return the sample
        return {
            "image": image,
            "question": question_token,
            "answer": answer_token,
        }

    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_image_size is not None:
            return image.resize(self.target_image_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def flatten_dataset(
        self,
        metadata_path: str,
        chat_path: str,
    ) -> List[Dict]:
        """
        Flatten the dataset such that each item contains the image, the human question from chat.json,
        and the blip_caption from metadata.json as the answer.
        """
        metadata_list = json.load(open(metadata_path))
        chat_list = json.load(open(chat_path))
        # Build id -> metadata mapping
        metadata_map = {item["id"]: item for item in metadata_list}
        flattened_data = []

        for chat in chat_list:
            chat_id = chat["id"]
            meta = metadata_map.get(chat_id)
            if not meta:
                continue  # skip if no metadata

            # Load image
            image_path = os.path.join(self.image_dir, chat["image"])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error("Error loading image %s: %s", image_path, e)
                continue

            # Get first human question
            question = None
            for conv in chat.get("conversations", []):
                if conv.get("from") == "human":
                    question = conv.get("value", "").replace("<image>", "").strip()
                    break
            if not question:
                continue

            # Get blip_caption as answer
            answer = meta.get("blip_caption", "")

            flattened_data.append(
                {"image": image, "question": question, "answer": answer}
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


class LLavaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: LLavaDataset,
        batch_size: int,
        num_workers: int,
    ):
        super(LLavaDataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        if stage == "fit":
            self.train_dataset = LLavaDataset(
                self.dataset.data[
                    : int(len(self.dataset.data) * 0.7)
                ],  # 70% for training
                self.dataset.image_dir,
                self.dataset.target_image_size,
                self.dataset.metadata_path,
                self.dataset.chat_path,
                self.dataset.sort_json_key,
            )
            self.val_dataset = LLavaDataset(
                self.dataset.data[
                    int(len(self.dataset.data) * 0.7) : int(
                        len(self.dataset.data) * 0.85
                    )
                ],  # 15% for validation
                self.dataset.image_dir,
                self.dataset.target_image_size,
                self.dataset.metadata_path,
                self.dataset.chat_path,
                self.dataset.sort_json_key,
            )
        elif stage == "test":
            self.test_dataset = LLavaDataset(
                self.dataset.data[
                    int(len(self.dataset.data) * 0.85) :
                ],  # 15% for testing
                self.dataset.image_dir,
                self.dataset.target_image_size,
                self.dataset.metadata_path,
                self.dataset.chat_path,
                self.dataset.sort_json_key,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
