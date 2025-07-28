"""
Prepare the Psychology 10K dataset for training by transforming it into a JSONL format.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from loguru import logger

logger = logger.bind(name="prepare_data")


def transform_entry(example):
    """Transform a single entry from the dataset to the desired"""
    instruction = example.get(
        "instruction", "Please provide a response based on the following input."
    )
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    return {"instruction": instruction, "input": input_text, "output": output_text}


def main():
    """Main function to load, transform, and save the dataset"""
    dataset = load_dataset("samhog/psychology-10k")

    # Assuming the data is in the 'train' split, adjust if needed
    raw_data = dataset["train"]

    # Transform all entries
    transformed_data = [transform_entry(entry) for entry in raw_data]

    # keep only the first 20 entries for testing
    transformed_data = transformed_data[:20]

    # Save to JSONL format
    output_file = "../data/psychology-10k.jsonl"
    logger.info(f"Saving transformed dataset to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in transformed_data:
            # Ensure proper JSON formatting for each line
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Transformed dataset saved to {output_file}")
    logger.info(f"Total entries processed: {len(transformed_data)}")


if __name__ == "__main__":
    main()
