"""
Fine-tune a Phi-3 model for mental health assistance using Unsloth.
"""

import os

# isort: off
from unsloth import FastLanguageModel

# isort: on

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import TrainingArguments
from trl import SFTTrainer

# --- Configuration ---
# Model
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
DTYPE = None

# LoRA Parameters (Unsloth defaults are often good, these are examples)
LORA_R = 64  # Increased rank for potentially better performance
LORA_ALPHA = 128  # Usually 2x rank
LORA_DROPOUT = 0  # 0 is optimized in Unsloth
BIAS = "none"  # "none" is optimized
USE_GRADIENT_CHECKPOINTING = "unsloth"  # Use Unsloth's optimized version


# Training Hyperparameters
OUTPUT_DIR = "../outputs/mental-llm-phi3-unsloth"
BATCH_SIZE = 2  # Per device batch size (adjust based on your GPU memory)
GRADIENT_ACCUMULATION_STEPS = (
    4  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)
LEARNING_RATE = 2e-4
EPOCHS = 1  # Adjust as needed
WARMUP_STEPS = 10
LOGGING_STEPS = 25
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2

# --- End Configuration ---


def main():
    """Main function to set up and run the Unsloth training."""
    logger.info("Starting Unsloth fine-tuning process...")

    # --- 1. GPU Check ---
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available. Training will be very slow on CPU.")

    # --- 2. Load Model and Tokenizer (using Unsloth) ---
    logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=True,  # Ensure it's loaded in 4-bit as intended by the model name
    )
    logger.info("Model and tokenizer loaded successfully.")

    # --- 3. Configure LoRA (using Unsloth) ---
    logger.info("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=BIAS,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=3407,  # For reproducibility
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("LoRA adapters configured.")
    model.print_trainable_parameters()

    # --- 4. Load and Prepare Dataset ---
    dataset = load_dataset("samhog/psychology-10k", split="train")
    # Assuming the dataset has 'instruction', 'input', and 'output' fields
    text_data = {"text": []}

    # Iterate over the dataset and format the data
    for example in dataset:  # Iterate directly over the dataset
        instruction = example["instruction"]
        input_text = example["input"]
        output_text = example["output"]

        # Format the text
        text_format = f"<|system|>{instruction}<|end|><|user|>{input_text}<|end|><|assistant|>{output_text}<|end|>"
        text_data["text"].append(text_format)

    # Convert the dictionary to a Dataset object
    train_dataset = Dataset.from_dict(text_data)
    del text_data
    for i, row in enumerate(train_dataset.select(range(1))):
        for key in row.keys():
            logger.info(f"Row {i + 1}:\n{key}: {row[key]}\n")

    # --- 5. Setup Trainer ---
    logger.info("Setting up SFTTrainer...")
    sample_train_dataset = train_dataset.shuffle().select(range(1000))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sample_train_dataset,
        dataset_text_field="text",  # This matches the field name in your formatted data
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,  # Number of processes for data loading (adjust if needed)
        packing=False,  # Packing can improve efficiency but changes data format slightly
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),  # Use FP16 if BF16 not supported
            bf16=torch.cuda.is_bf16_supported(),  # Use BF16 if supported
            logging_steps=LOGGING_STEPS,
            optim="adamw_8bit",  # Optimizer optimized for QLoRA/Unsloth
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            save_strategy=SAVE_STRATEGY,
            save_total_limit=SAVE_TOTAL_LIMIT,
            dataloader_pin_memory=False,  # Recommended for Unsloth
            report_to="none",  # Disable reporting to avoid issues with Unsloth
        ),
    )
    logger.info("SFTTrainer initialized.")

    # --- 6. Train ---
    logger.info("Starting training...")
    trainer_stats = trainer.train()  # This returns training statistics
    logger.info("Training complete!")
    logger.info(f"Training stats: {trainer_stats}")

    # --- 7. Save Final Model (Adapters) ---
    logger.info(f"Saving final LoRA adapters to {OUTPUT_DIR}")
    # Save the PEFT model (adapters) and tokenizer in Hugging Face format
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Model and tokenizer saved in Hugging Face format.")

    # --- 8. Export to GGUF for Ollama ---
    logger.info("Exporting model to GGUF format for Ollama...")
    try:
        # Enable faster inference mode for saving
        FastLanguageModel.for_inference(model)

        # Export to GGUF
        # Quantization method: "q4_k_m" is a good balance, "q8_0" is higher quality
        gguf_output_dir = os.path.join(OUTPUT_DIR, "gguf")
        model.save_pretrained_gguf(
            save_directory=gguf_output_dir,
            tokenizer=tokenizer,
            quantization_method="q4_k_m",  # You can experiment with "q8_0" if you have more space/speed needs
        )
        logger.info(f"Model exported to GGUF format in {gguf_output_dir}")
        logger.info("Look for the .gguf file(s) in that directory for use with Ollama.")

        # List the generated GGUF files
        gguf_files = [f for f in os.listdir(gguf_output_dir) if f.endswith(".gguf")]
        if gguf_files:
            logger.info("Generated GGUF files:")
            for f in gguf_files:
                logger.info(f" - {os.path.join(gguf_output_dir, f)}")
        else:
            logger.warning("No .gguf files found in the output directory after export.")

    except Exception as e:
        logger.error(f"Error during GGUF export: {e}")
        logger.info(
            "Model saved in Hugging Face format. You can convert to GGUF manually later using llama.cpp tools if needed."
        )

    logger.info("Unsloth fine-tuning and export process finished!")


if __name__ == "__main__":
    main()
