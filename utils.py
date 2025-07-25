import logging
import re
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from nltk.corpus import stopwords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_and_plot(json_data: List[Dict[str, Any]]) -> None:
    # Extract captions
    captions = [item["caption"] for item in json_data if "caption" in item]

    # Remove all punctuation
    captions = [re.sub(r"[^\w\s]", "", caption) for caption in captions]

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [
        word
        for caption in captions
        for word in caption.lower().split()
        if word not in stop_words
    ]

    # Count word frequencies
    word_counts = Counter(words)

    # Get top 20 most and least frequent words
    most_common = word_counts.most_common(20)
    least_common = word_counts.most_common()[:-21:-1]

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot most common words
    axs[0].bar(
        [word for word, count in most_common], [count for word, count in most_common]
    )
    axs[0].set_title("Top 20 Most Frequent Words")
    axs[0].set_xticks(range(len(most_common)))
    axs[0].set_xticklabels(
        [word for word, count in most_common], rotation=45, ha="right"
    )

    # Plot least common words
    axs[1].bar(
        [word for word, count in least_common], [count for word, count in least_common]
    )
    axs[1].set_title("Top 20 Least Frequent Words")
    axs[1].set_xticks(range(len(least_common)))
    axs[1].set_xticklabels(
        [word for word, count in least_common], rotation=45, ha="right"
    )

    # save the plot
    plt.savefig("word_frequency.png")


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
):
    """Save model and optimizer state dicts to the given path."""
    save_dict = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        save_dict["epoch"] = epoch
    torch.save(save_dict, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
