#!/usr/bin/env python3
"""
GRPO Training with LoRA - Based on grpo-train.py
================================================

A LoRA version of the original grpo-train.py script that maintains the same
GRPO training approach while adding parameter-efficient fine-tuning.

Features:
- GRPO (Group Relative Policy Optimization) training
- LoRA parameter-efficient fine-tuning
- Same reward function as original (unique character counting)
- Memory-efficient training configuration

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
OUTPUT_DIR = "./grpo_lora_train_output"
LORA_OUTPUT_DIR = f"{OUTPUT_DIR}/lora_adapter"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 GRPO Training with LoRA")
print("=" * 30)

# Load dataset (same as original)
print("📊 Loading TLDR dataset...")
dataset = load_dataset("trl-lib/tldr", split="train")
# Take a smaller subset for quick training
dataset = dataset.select(range(50))  # Reduced for stability
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
    r=8,                                # Small rank for stability
    lora_alpha=16,                      # Conservative scaling
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

# Dummy reward function: count the number of unique characters in the completions
# (Same as original grpo-train.py)
def reward_num_unique_chars(completions, **kwargs):
    """
    Same reward function as original grpo-train.py:
    Count unique characters in completions for diversity reward.
    """
    rewards = []
    for completion in completions:
        unique_count = len(set(completion))
        # Normalize to reasonable range
        reward = min(unique_count / 100.0, 1.0)
        rewards.append(reward)
    return rewards

# GRPO Configuration (adapted for LoRA and stability)
print("⚙️ Configuring GRPO trainer...")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                 # Single epoch for testing
    per_device_train_batch_size=1,      # Very small batch for stability
    gradient_accumulation_steps=2,      # Effective batch size of 2
    learning_rate=5e-5,                 # Lower LR for stability with LoRA
    logging_steps=5,                    # Frequent logging
    save_steps=25,                      # Save every 25 steps
    max_prompt_length=256,              # Shorter for stability
    max_completion_length=128,          # Shorter completions
    temperature=1.2,                    # Higher temperature for stability
    top_k=10,                           # Very conservative sampling
    top_p=0.85,                         # Conservative top_p
    num_generations=2,                  # Minimum required for GRPO
    warmup_steps=2,                     # Quick warmup
    weight_decay=0.01,
    remove_unused_columns=False,
    max_steps=20,                       # Limit steps for testing
)

# Initialize trainer
print("🏋️ Initializing GRPO trainer...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_num_unique_chars],  # Same reward function as original
    args=training_args,
    train_dataset=dataset,
)

print("🎯 Starting GRPO training with LoRA...")
print(f"   Dataset size: {len(dataset)}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Max steps: {training_args.max_steps}")
print(f"   Reward function: unique character counting (same as original)")
print(f"   Output directory: {OUTPUT_DIR}")

try:
    # Train the model
    trainer.train()
    print("✅ GRPO training completed!")
    
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
    
    print("🎉 GRPO LoRA training completed successfully!")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("💡 This is expected with GRPO as it can be numerically unstable.")
    print("💡 Try the SFT version (sft-lora-simple.py) for more stable training.")

print("\n" + "=" * 30)
print("📖 To load the trained LoRA adapter (if successful):")
print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{LORA_OUTPUT_DIR}")

# Use for inference
prompt = "Your text here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
""")