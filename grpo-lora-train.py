#!/usr/bin/env python3
"""
GRPO Training with LoRA - Simple Version
========================================

A simplified GRPO training script with LoRA parameter-efficient fine-tuning.
This version uses the TLDR dataset and a basic reward function while only
training LoRA adapters for memory efficiency.

Features:
- LoRA fine-tuning (only 0.22% trainable parameters)
- GRPO (Group Relative Policy Optimization)
- Automatic model and tokenizer setup
- Memory-efficient training configuration
- Custom reward functions

Usage:
    python grpo-lora-train.py

Dependencies:
    pip install torch transformers datasets accelerate peft trl
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./grpo_lora_simple_output"
LORA_OUTPUT_DIR = f"{OUTPUT_DIR}/lora_adapter"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 GRPO Training with LoRA - Simple Version")
print("=" * 50)

# Load dataset
print("📊 Loading TLDR dataset...")
dataset = load_dataset("trl-lib/tldr", split="train")
print(f"Dataset loaded: {len(dataset)} samples")

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
    r=16,                               # Rank of adaptation
    lora_alpha=32,                      # LoRA scaling parameter
    target_modules=["q_proj", "v_proj"], # Target attention modules
    lora_dropout=0.05,                  # Dropout probability
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

# Reward function: count unique characters in completions
def reward_num_unique_chars(completions, **kwargs):
    """
    Simple reward function that counts unique characters.
    Higher character diversity = higher reward.
    """
    rewards = []
    for completion in completions:
        # Count unique characters and normalize
        unique_chars = len(set(completion.lower()))
        # Normalize to 0-1 range (assuming max ~50 unique chars)
        reward = min(unique_chars / 50.0, 1.0)
        rewards.append(reward)
    return rewards

# GRPO Configuration
print("⚙️ Configuring GRPO trainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                 # Single epoch for quick training
    per_device_train_batch_size=1,      # Very small batch size for stability
    gradient_accumulation_steps=4,      # Effective batch size of 4
    learning_rate=1e-4,                 # Lower LR for stability
    logging_steps=10,                   # Log every 10 steps
    save_steps=100,                     # Save every 100 steps
    max_prompt_length=256,              # Shorter prompts for stability
    max_completion_length=128,          # Shorter completions
    temperature=1.0,                    # Higher temperature for stability
    top_k=20,                           # Lower top_k for stability
    top_p=0.9,                          # Lower top_p for stability
    num_generations=2,                  # Reduce generations for speed
    warmup_steps=5,                     # Quick warmup
    weight_decay=0.01,
    remove_unused_columns=False,
    max_steps=50,                       # Limit training steps for testing
)

# Initialize trainer
print("🏋️ Initializing GRPO trainer...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_num_unique_chars],
    args=training_args,
    train_dataset=dataset,
)

print("🎯 Starting GRPO training with LoRA...")
print(f"   Dataset size: {len(dataset)}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
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
else:
    print("⚠️ Warning: LoRA adapter directory not found")

print("🎉 GRPO LoRA training completed successfully!")
print("\n" + "=" * 50)
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
prompt = "Summarize this text: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
""")