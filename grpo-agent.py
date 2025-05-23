# -*- coding: utf-8 -*-
"""
Protocol-Native Agent  +  Chain-of-Thought  +  GRPO (Qwen1.5-0.5B) with LoRA
==============================================================================

*Key Features*
--------------
1. **LoRA Fine-tuning**: Uses Parameter Efficient Fine-Tuning with LoRA adapters
2. **GRPO Training**: Group Relative Policy Optimization for preference learning
3. **Memory Efficient**: Reduced batch size and gradient accumulation for LoRA
4. **Chain-of-Thought**: Agent reasoning with <THINK> tokens
5. **Task-specific Rewards**: Custom reward function for agent actions
6. **Checkpoint Saving**: Saves LoRA adapters for later use

> Installation:
> ```bash
> pip install torch transformers trl accelerate peft sentencepiece tqdm datasets
> ```
"""
from __future__ import annotations
import json, random, re, uuid, torch, os
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import GRPOTrainer, AutoModelForCausalLMWithValueHead, GRPOConfig
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from datasets import Dataset

# ---------------------------------------------------------------------------
# 0. Special tokens
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
    print("[Device] Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("[Device] Using CUDA GPU")
else:
    DEVICE = "cpu"
    print("[Device] Using CPU")

SPECIAL = (
    ["<|obs|>", "<|agent|>", "<THINK>", "<END_THINK>", "<|pad|>"]
    + [f"<ID_{i}>" for i in range(1, 301)]
)

Tok = AutoTokenizer.from_pretrained(BASE_MODEL)
Tok.add_special_tokens({"additional_special_tokens": SPECIAL})
Tok.pad_token = "<|pad|>"
if Tok.pad_token_id is None:
    Tok.pad_token_id = Tok.eos_token_id
print(f"[Tokenizer] Vocab size: {len(Tok)}")
print(f"[Tokenizer] Pad token: {Tok.pad_token}, Pad token ID: {Tok.pad_token_id}")

# ---------------------------------------------------------------------------
# 1. Environment / Task Generation (Mostly unchanged)
# ---------------------------------------------------------------------------
def generate_task(difficulty: int) -> Tuple[List[int], int]:
    num_items = difficulty + 1
    if num_items > 299:
        num_items = 299
    possible_ids = list(range(1, 301))
    random.shuffle(possible_ids)
    item_ids = sorted(random.sample(possible_ids, k=num_items))
    target_id = item_ids[0]
    return item_ids, target_id

# ---------------------------------------------------------------------------
# 2. Action Parsing and Observation Formatting (Mostly unchanged)
# ---------------------------------------------------------------------------
ACTION_RE = re.compile(r"<END_THINK>\s*<ID_(\d+)>")

def format_observation_prompt(item_ids: List[int]) -> str:
    obs_string = " ".join([f"<ID_{i}>" for i in item_ids])
    return f"<|obs|>{obs_string}<|agent|><THINK>The items are {obs_string}. I need to identify the first item and select it.<END_THINK>"

def parse_agent_action(generated_sequence: str) -> int | None:
    match = ACTION_RE.search(generated_sequence)
    if match:
        return int(match.group(1))
    return None

# ---------------------------------------------------------------------------
# 3. Model Setup with LoRA
# ---------------------------------------------------------------------------
LM = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
LM.resize_token_embeddings(len(Tok))

# LoRA configuration
lora_config = LoraConfig(
    r=16,                           # Rank of adaptation
    lora_alpha=32,                  # LoRA scaling parameter
    target_modules=["q_proj", "v_proj"],  # Target attention modules
    lora_dropout=0.05,              # Dropout probability
    bias="none",                    # Bias type
    task_type="CAUSAL_LM"           # Task type
)

# Apply LoRA to the model
LM = get_peft_model(LM, lora_config)
print(f"[LoRA] Model parameters: {LM.num_parameters():,}")
trainable_params = LM.get_nb_trainable_parameters()
if isinstance(trainable_params, tuple):
    trainable_params = trainable_params[0]
print(f"[LoRA] Trainable parameters: {trainable_params:,}")

# Setup generation config for LoRA model
if hasattr(LM, 'base_model'): # PEFT model
    base_model = LM.base_model.model
    if not hasattr(base_model, 'generation_config') or base_model.generation_config is None:
        from transformers import GenerationConfig
        base_model.generation_config = GenerationConfig(
            max_new_tokens=60,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=Tok.pad_token_id,
            eos_token_id=Tok.eos_token_id
        )
    # Also set on the PEFT model for compatibility
    LM.generation_config = base_model.generation_config
else:
    print("[Warning] Model structure unexpected. Generation might use defaults.")

LM.to(DEVICE)
LM.warnings_issued = {} # Added for GRPOTrainer
def dummy_add_model_tags_lm(tags_list):
    print(f"[INFO] LM Model tags (notional): {tags_list}")
LM.add_model_tags = dummy_add_model_tags_lm # Added for GRPOTrainer


# POL (AutoModelForCausalLMWithValueHead) is instantiated from LM.
# However, for the current GPRO setup, GRPOTrainer uses the base LM directly.
# POL and its value head are not used in the training loop or the demo.
# You can uncomment the following lines if you need to access the value head explicitly for other purposes.
# POL = AutoModelForCausalLMWithValueHead.from_pretrained(LM)
# POL.to(DEVICE)

# ---------------------------------------------------------------------------
# 4. GRPOConfig Configuration for GRPOTrainer with LoRA
# ---------------------------------------------------------------------------
EPOCHS_GRPO = 3
GRPO_BATCH_SIZE = 2                    # Reduced for LoRA training
GRPO_LEARNING_RATE = 5e-4              # Higher LR for LoRA
TRAIN_DATASET_SAMPLES = 32             # Reduced dataset size
INITIAL_DIFFICULTY_GRPO = 0
OUTPUT_DIR = "./grpo_lora_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS_GRPO,
    learning_rate=GRPO_LEARNING_RATE,
    per_device_train_batch_size=GRPO_BATCH_SIZE,
    gradient_accumulation_steps=2,      # For effective batch size of 4
    remove_unused_columns=False,
    logging_steps=5,                    # More frequent logging for LoRA
    save_steps=50,                      # Save checkpoints
    max_prompt_length=64,
    max_completion_length=64,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_generations=4,
    warmup_steps=10,                    # Warmup for LoRA training
    weight_decay=0.01,                  # Weight decay for regularization
)

# ---------------------------------------------------------------------------
# 5. Dataset Preparation and Reward Function for GRPOTrainer
# ---------------------------------------------------------------------------
def create_grpo_dataset(num_samples: int, difficulty: int) -> Dataset:
    data_list = []
    for _ in range(num_samples):
        item_ids, target_id = generate_task(difficulty)
        prompt = format_observation_prompt(item_ids)
        data_list.append({
            "prompt": prompt, 
            "target_id": target_id, 
            "item_ids_str": " ".join(map(str, item_ids))
        })
    return Dataset.from_list(data_list)

def grpo_reward_function(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    target_ids = kwargs["target_id"]
    
    if len(completions) != len(target_ids):
        raise ValueError(f"Mismatch between completions ({len(completions)}) and target_ids ({len(target_ids)})")

    for i in range(len(completions)):
        completion = completions[i]
        target_id = target_ids[i]
        
        chosen_id = parse_agent_action(completion)
        reward_val = 0.0
        if chosen_id is not None and chosen_id == target_id:
            reward_val = 1.0
        elif chosen_id is not None and chosen_id != target_id:
            reward_val = -0.5
        else:
            reward_val = -1.0
        rewards.append(reward_val)
    return rewards

print(f"Creating GRPO training dataset with {TRAIN_DATASET_SAMPLES} samples at difficulty {INITIAL_DIFFICULTY_GRPO}...")
train_dataset_grpo = create_grpo_dataset(TRAIN_DATASET_SAMPLES, INITIAL_DIFFICULTY_GRPO)
print(f"Dataset created. Example prompt: {train_dataset_grpo[0]['prompt']}")

# ---------------------------------------------------------------------------
# 6. GRPOTrainer Initialization and Training
# ---------------------------------------------------------------------------
print("Initializing GRPOTrainer...")
grpo_trainer = GRPOTrainer(
    model=LM, # Changed from POL to LM
    processing_class=Tok,
    args=grpo_config,
    reward_funcs=[grpo_reward_function],
    train_dataset=train_dataset_grpo,
    # ref_model=LM, # GRPOTrainer creates its own reference model from the provided model (LM)
)

print("Starting GRPO training with LoRA...")
grpo_trainer.train()
print("GRPO training with LoRA finished.")

# Save the LoRA adapter
print(f"Saving LoRA adapter to {OUTPUT_DIR}/lora_adapter...")
LM.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
Tok.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print("LoRA adapter saved successfully.")

# ---------------------------------------------------------------------------
# 7. Demo after training (Adapted from PPO version)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Running Demo Post-Training (GRPO with LoRA) ---")
    
    # IMPORTANT: For the demo, we should use the trained LoRA model from GRPOTrainer.
    # GRPOTrainer modifies the 'model' it was given (LM in this case).
    # So, LM is the trained LoRA model.
    
    demo_difficulty = INITIAL_DIFFICULTY_GRPO
    item_ids, target_id = generate_task(demo_difficulty)
    
    prompt_str = format_observation_prompt(item_ids)
    print(f"Demo Task: Select the first item from: {' '.join([f'<ID_{i}>' for i in item_ids])}")
    print(f"Target ID: <ID_{target_id}>")
    print(f"Input Prompt to Model:\n{prompt_str}")

    input_ids = Tok.encode(prompt_str, return_tensors="pt").to(DEVICE)
    
    # Use LM for generation, as it's the trained LoRA model from GRPOTrainer
    # Ensure LM has generation_config set correctly for LoRA model
    if not hasattr(LM, 'generation_config') or LM.generation_config is None:
        print("[Demo Warning] LM does not have generation_config. Setting default generation parameters for LoRA model.")
        # Set generation config for LoRA model
        from transformers import GenerationConfig
        LM.generation_config = GenerationConfig(
            max_new_tokens=60,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=Tok.pad_token_id,
            eos_token_id=Tok.eos_token_id
        )


    print(f"LM generation config for demo: {LM.generation_config}")

    # outputs = POL.generate( # Changed from POL to LM
    outputs = LM.generate(
        input_ids,
        generation_config=LM.generation_config
    )
    
    generated_part_decoded = Tok.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    full_output_decoded = Tok.decode(outputs[0], skip_special_tokens=False)
    
    print(f"\nModel Full Output:\n{full_output_decoded}")
    print(f"\nModel Generated Part (CoT + Action):\n{generated_part_decoded}")

    chosen_id = parse_agent_action(generated_part_decoded)
    
    if chosen_id is not None:
        print(f"\nParsed Action (Selected ID): <ID_{chosen_id}>")
        if chosen_id == target_id:
            print("Result: CORRECT! Agent selected the target item.")
        else:
            print(f"Result: INCORRECT. Agent selected <ID_{chosen_id}>, but target was <ID_{target_id}>.")
    else:
        print("\nResult: INVALID ACTION. Agent did not produce a recognizable action.")

    print("--- Demo End ---") 