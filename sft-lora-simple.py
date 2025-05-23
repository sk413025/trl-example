#!/usr/bin/env python3
"""
Simple LoRA Fine-tuning with TRL SFTTrainer
===========================================

A simplified and stable LoRA training script using SFTTrainer instead of GRPO.
This version provides a reliable baseline for LoRA parameter-efficient fine-tuning.

Features:
- LoRA fine-tuning (only 0.22% trainable parameters)
- Stable supervised fine-tuning
- Automatic model and tokenizer setup
- Memory-efficient training configuration

Usage:
    python grpo-lora-simple.py

Dependencies:
    pip install torch transformers datasets accelerate peft trl
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./lora_simple_output"
LORA_OUTPUT_DIR = f"{OUTPUT_DIR}/lora_adapter"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Simple LoRA Fine-tuning with TRL")
print("=" * 40)

# Load dataset
print("📊 Loading TLDR dataset...")
dataset = load_dataset("trl-lib/tldr", split="train")
# Take a small subset for quick training
dataset = dataset.select(range(100))  # Only 100 samples for testing
print(f"Dataset loaded: {len(dataset)} samples")
print(f"Dataset columns: {dataset.column_names}")
print(f"Sample data: {dataset[0]}")

# Load model and tokenizer
print(f"🤖 Loading model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Add padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Model loaded: {model.num_parameters():,} parameters")

# Configure LoRA
print("🔧 Configuring LoRA...")
lora_config = LoraConfig(
    r=8,                                # Smaller rank for stability
    lora_alpha=16,                      # Lower alpha
    target_modules=["q_proj", "v_proj"], # Target attention modules
    lora_dropout=0.1,                   # Higher dropout for regularization
    bias="none",                        # Bias type
    task_type=TaskType.CAUSAL_LM        # Task type
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
trainable_params = model.get_nb_trainable_parameters()
if isinstance(trainable_params, tuple):
    trainable_params = trainable_params[0]

print(f"✅ LoRA applied:")
print(f"   Total parameters: {model.num_parameters():,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Trainable percentage: {trainable_params/model.num_parameters()*100:.3f}%")

# Prepare dataset for SFT
def format_prompts(example):
    """Format the dataset for supervised fine-tuning"""
    # Use the actual fields from TLDR dataset
    prompt = example['prompt']
    completion = example['completion']
    return {"text": f"Post: {prompt}\n\nTL;DR: {completion}"}

print("📝 Formatting dataset...")
formatted_dataset = dataset.map(format_prompts)

# Training Configuration
print("⚙️ Configuring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                 # Single epoch for quick training
    per_device_train_batch_size=1,      # Small batch size for memory efficiency
    gradient_accumulation_steps=4,      # Effective batch size of 4
    learning_rate=2e-4,                 # Learning rate for LoRA
    logging_steps=5,                    # Log every 5 steps
    save_steps=50,                      # Save every 50 steps
    max_steps=20,                       # Limit training steps for testing
    warmup_steps=2,                     # Quick warmup
    weight_decay=0.01,
    fp16=False,                         # Disable fp16 for stability
    bf16=False,                         # Disable bf16 for stability
    remove_unused_columns=False,
    gradient_checkpointing=False,       # Disable for stability
    dataloader_drop_last=True,
    eval_strategy="no",                 # No evaluation for simplicity
    save_total_limit=1,                 # Keep only 1 checkpoint
    report_to=None,                     # Disable reporting
)

# Initialize trainer
print("🏋️ Initializing SFT trainer...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=formatted_dataset,
)

print("🎯 Starting LoRA training...")
print(f"   Dataset size: {len(formatted_dataset)}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Max steps: {training_args.max_steps}")
print(f"   Output directory: {OUTPUT_DIR}")

# Train the model
trainer.train()

print("✅ Training completed!")

# Save LoRA adapter
print("💾 Saving LoRA adapter...")
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

print(f"📁 LoRA adapter saved to: {LORA_OUTPUT_DIR}")
print(f"📁 Training logs saved to: {OUTPUT_DIR}")

# Verify saved files
if os.path.exists(LORA_OUTPUT_DIR):
    saved_files = os.listdir(LORA_OUTPUT_DIR)
    print(f"📄 Saved files: {', '.join(saved_files)}")
    
    # Calculate total size
    total_size = 0
    for file in saved_files:
        file_path = os.path.join(LORA_OUTPUT_DIR, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
    
    print(f"📊 Total adapter size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
else:
    print("⚠️ Warning: LoRA adapter directory not found")

print("🎉 LoRA training completed successfully!")
print("\n" + "=" * 40)
print("📖 To load the trained LoRA adapter:")
print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{LORA_OUTPUT_DIR}")

# Use for inference
prompt = "Summarize: Your text here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
""")