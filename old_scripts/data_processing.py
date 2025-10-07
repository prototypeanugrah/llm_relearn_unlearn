import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset


def filter_data(
    josn_file_path: str,
    num_samples: int,
) -> Tuple[List[Dict[str, Any]], int]:
    with open(josn_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        data[:num_samples],
        len(data),
    )


def create_forget_retain_data(
    data: List[Dict[str, Any]],
    filter_word: str,
    image_dir: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create forget and retain data based on the filter word.

    Args:
        data (List[Dict[str, Any]]): The json data to process.
        filter_word (str): The word to filter the data by.
        image_dir (str): The directory containing the images.
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: The forget and
        retain data.
    """
    # Convert data to a DataFrame
    df = pd.DataFrame(data)

    # Construct full image path
    df["image_path"] = df["image"].apply(lambda x: os.path.join(image_dir, f"{x}"))

    # Filter data using DataFrame operations
    forget_df = df[df["caption"].str.contains(filter_word, na=False)]
    retain_df = df[~df["caption"].str.contains(filter_word, na=False)]

    # Convert DataFrames to Hugging Face Datasets
    forget_dataset = Dataset.from_pandas(forget_df)
    retain_dataset = Dataset.from_pandas(retain_df)

    # Remove certain columns
    forget_dataset = forget_dataset.remove_columns(
        ["__index_level_0__", "url", "image"]
    )
    retain_dataset = retain_dataset.remove_columns(
        ["__index_level_0__", "url", "image"]
    )

    return forget_dataset, retain_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path", type=str, default="dataset/llava/metadata.json"
    )
    parser.add_argument("--chat_path", type=str, default="dataset/llava/chat.json")
    parser.add_argument("--image_dir", type=str, default="dataset/llava/images")
    parser.add_argument("--num_samples", type=int, default=20000)
    args = parser.parse_args()

    metadata_filtered, total_samples_metadata = filter_data(
        args.metadata_path, args.num_samples
    )
    chat_filtered, total_samples_chat = filter_data(args.chat_path, args.num_samples)

    print(f"Total samples: {total_samples_metadata}")
    print(f"Filtering samples: {args.num_samples}")

    forget_dataset, retain_dataset = create_forget_retain_data(
        data=metadata_filtered,
        filter_word="blue",
        image_dir=args.image_dir,
    )
    print(f"Size of forget dataset: {len(forget_dataset)}")
    print(f"Size of retain dataset: {len(retain_dataset)}")

    # Print the first sample of the forget dataset
    print(forget_dataset)
    print(retain_dataset)
