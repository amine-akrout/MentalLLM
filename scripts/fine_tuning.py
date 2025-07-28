"""Fine-tuning a Phi-3 model with LoRA adapters using Hugging Face Transformers and PEFT."""

import json

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logger = logger.bind(name="train_lora")

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "../data/psychology-10k.jsonl"
OUTPUT_DIR = "../adapters/lora_adapters_phi3"
USE_QLORA = True

# Training Hyperparameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS = 500


def format_instruction(example):
    """Formats the instruction, input, output for the model."""
    full_prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return full_prompt


def preprocess_dataset(tokenizer, max_length, dataset):
    """Tokenizes the dataset."""

    def formatting_prompts_func(example):
        # Format the instruction
        output_texts = []
        for i in range(len(example["instruction"])):
            text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts

    def tokenize_function(example):
        return tokenizer(
            formatting_prompts_func(example),
            truncation=True,
            padding=False,  # DataCollator will handle padding
            max_length=max_length,
            return_tensors=None,  # Return lists, let collator handle tensors
        )

    # Apply tokenization
    dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    return dataset


def main():
    """Main function to set up and run the training."""

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    logger.info("Loading model...")
    if USE_QLORA:
        # Configure BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        # Prepare model for k-bit training (QLoRA specific)
        model = prepare_model_for_kbit_training(model)
    else:
        # Standard LoRA (loads full precision model)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # Move model to GPU if available
        model = model.to("cuda") if torch.cuda.is_available() else model

    # Configure LoRA ---
    logger.info("Configuring LoRA adapters...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Add LoRA adapters to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and Preprocess Dataset
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    # Shuffle and potentially split for validation
    dataset = dataset.shuffle(seed=42)
    train_val_split = dataset.train_test_split(test_size=0.05)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]

    # Tokenize datasets
    MAX_LENGTH = 512
    train_dataset = preprocess_dataset(tokenizer, MAX_LENGTH, train_dataset)
    eval_dataset = preprocess_dataset(tokenizer, MAX_LENGTH, eval_dataset)

    # Setup Training Arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,  # Often same or slightly larger
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",  # Evaluate every 'eval_steps'
        eval_steps=SAVE_STEPS,  # Match save steps or adjust
        save_total_limit=2,  # Keep only last 2 checkpoints
        report_to="none",  # Change if using wandb/tensorboard
        logging_dir="./results/logs",
        lr_scheduler_type="cosine",  # Common choice
        warmup_ratio=0.1,
        fp16=not USE_QLORA,  # Often used for standard LoRA if model supports it (QLoRA handles precision differently)
        bf16=USE_QLORA
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),  # Use bf16 if available and using QLoRA
        # optim="paged_adamw_8bit", # For QLoRA, often recommended
    )

    # --- 6. Setup Trainer ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save Final Model (Adapters) ---
    logger.info(f"Saving final LoRA adapters to {OUTPUT_DIR}")
    trainer.save_model()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
