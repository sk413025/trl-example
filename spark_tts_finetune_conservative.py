#!/usr/bin/env python3
"""
SparkTTS Ultra-Conservative Fine-tuning
=======================================

This script uses extremely conservative settings to avoid training instability.
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import snapshot_download

# Configuration
BASE_MODEL_NAME = "Spark-TTS-0.5B"
OUTPUT_DIR = "./spark_tts_conservative_output"
LORA_OUTPUT_DIR = f"{OUTPUT_DIR}/lora_adapter"
MAX_SEQ_LENGTH = 1024  # Shorter sequences

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 SparkTTS Ultra-Conservative Fine-tuning")
print("=" * 40)

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using MPS device")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

# Load model and tokenizer
print(f"🤖 Loading model: {BASE_MODEL_NAME}/LLM")
tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_NAME}/LLM")
model = AutoModelForCausalLM.from_pretrained(
    f"{BASE_MODEL_NAME}/LLM",
    torch_dtype=torch.float32,
)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Model loaded: {model.num_parameters():,} parameters")

# Ultra-conservative LoRA configuration
print("🔧 Configuring ultra-conservative LoRA...")
lora_config = LoraConfig(
    r=16,                               # Very small rank
    lora_alpha=32,                      # Conservative alpha
    target_modules=["q_proj", "v_proj"], # Only attention value and query
    lora_dropout=0.2,                   # Higher dropout
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model = model.to(device)
model.print_trainable_parameters()

# Load a tiny dataset
print("📊 Loading tiny dataset...")
dataset = load_dataset("trl-lib/tldr", split="train")
dataset = dataset.select(range(50))  # Only 50 samples
print(f"Dataset loaded: {len(dataset)} samples")

# Simple formatting function
def format_prompts(example):
    """Simple formatting without audio processing"""
    text = f"Summarize: {example['prompt']}\n\nSummary: {example['completion']}"
    return {"text": text}

print("📝 Formatting dataset...")
formatted_dataset = dataset.map(format_prompts)

# Ultra-conservative training configuration
print("⚙️ Configuring ultra-conservative training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,      # No accumulation
    learning_rate=1e-5,                 # Very low learning rate
    logging_steps=1,
    save_steps=10,                      # Save very frequently
    save_strategy="steps",
    save_safetensors=True,
    max_steps=30,                       # Only 30 steps
    warmup_steps=5,                     # Adequate warmup
    weight_decay=0.1,                   # Higher weight decay
    max_grad_norm=0.5,                  # Aggressive gradient clipping
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    lr_scheduler_type="constant_with_warmup",  # Constant after warmup
    seed=42,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=False,       # Disable for stability
    load_best_model_at_end=False,
    metric_for_best_model=None,
    greater_is_better=None,
    push_to_hub=False,
    hub_model_id=None,
    hub_token=None,
)

# Initialize trainer
print("🏋️ Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
)

# Train with monitoring
print("🎯 Starting ultra-conservative training...")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Max gradient norm: {training_args.max_grad_norm}")
print(f"   Weight decay: {training_args.weight_decay}")
print(f"   Total steps: {training_args.max_steps}")

try:
    trainer.train()
    print("✅ Training completed successfully!")
    
    # Save the model
    print("💾 Saving LoRA adapter...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"📁 LoRA adapter saved to: {LORA_OUTPUT_DIR}")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("💡 Even ultra-conservative settings failed. The model might need different approaches.")

print("\n" + "=" * 40)
print("💡 Next steps:")
print("1. Try the saved checkpoints for inference")
print("2. Consider using QLoRA (4-bit quantization)")
print("3. Try freezing more layers")
print("4. Use an even smaller learning rate")
print("5. Try training on CPU instead of MPS")