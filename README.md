# TRL Example: Qwen2 Fine-tuning with GRPO and Supervised Learning

This repository demonstrates fine-tuning techniques for the Qwen2-0.5B-Instruct model using TRL (Transformer Reinforcement Learning) library. It includes implementations for both GRPO (Group Relative Policy Optimization) and supervised fine-tuning approaches.

## Overview

The project provides three main training scripts:
- **GRPO Training with LoRA**: Memory-efficient Group Relative Policy Optimization using LoRA adapters
- **GRPO Training**: Standard Group Relative Policy Optimization for preference-based learning  
- **Supervised Fine-tuning**: Traditional supervised learning approach with automatic model downloading

## Features

- 🤖 **Automatic Model Download**: Downloads Qwen2-0.5B-Instruct model from Hugging Face
- 📊 **Multiple Training Methods**: Supports GRPO, GRPO with LoRA, and supervised fine-tuning
- 🔧 **LoRA Fine-tuning**: Parameter-efficient training with only 0.22% trainable parameters
- 🧠 **Chain-of-Thought**: Agent reasoning with structured thinking tokens
- 🎯 **Custom Reward Functions**: Task-specific reward mechanisms for agent training
- 💾 **Checkpoint Saving**: Automatic model and LoRA adapter saving
- 📈 **Training Monitoring**: Built-in loss and accuracy tracking
- 🔧 **Compatibility Fixes**: Resolved dependency issues for modern environments

## Requirements

### Dependencies
```bash
pip install torch transformers datasets accelerate peft trl sentencepiece tqdm
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training supported
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM for faster training
- **Storage**: 2GB+ free space for model downloads and checkpoints

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sk413025/trl-example.git
   cd trl-example
   ```

2. **Install dependencies**:
   ```bash
   pip install torch transformers datasets accelerate peft trl sentencepiece tqdm
   ```

3. **Verify installation**:
   ```bash
   python -c "import transformers, trl; print('Setup complete!')"
   ```

## Usage

### GRPO Training with LoRA (Recommended)

The GRPO with LoRA script provides memory-efficient training using parameter-efficient fine-tuning. It trains only 1.08M parameters (0.22% of the full model) while maintaining competitive performance.

**Basic usage**:
```bash
python grpo-agent.py
```

**Quick test**:
```bash
python test_grpo_lora.py
```

**Key advantages**:
- **Memory Efficient**: Only trains LoRA adapters (~1M parameters)
- **Fast Training**: Reduced computational requirements
- **Chain-of-Thought**: Structured agent reasoning with `<THINK>` tokens
- **Custom Rewards**: Task-specific reward functions for agent actions
- **Automatic Saving**: Saves LoRA adapters to `./grpo_lora_output/lora_adapter/`

### Supervised Fine-tuning

The supervised fine-tuning script automatically downloads the Qwen2-0.5B-Instruct model and trains on the Stack Exchange dataset.

**Basic usage**:
```bash
python supervised_finetuning.py --model_path ./qwen2-saved
```

**Quick test run** (5 training steps):
```bash
python supervised_finetuning.py --model_path ./qwen2-saved --max_steps 5 --batch_size 1
```

**Extended training**:
```bash
python supervised_finetuning.py \
  --model_path ./qwen2-saved \
  --max_steps 1000 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --output_dir ./checkpoints
```

### Standard GRPO Training

The standard GRPO script implements preference-based training using policy optimization without LoRA.

**Basic usage**:
```bash
python grpo-train.py --model_path ./Qwen2-0.5B-Instruct
```

**Note**: This trains the full model and requires more memory than the LoRA version.

### Command Line Arguments

#### GRPO with LoRA Options (grpo-agent.py)
| Configuration | Default | Description |
|---------------|---------|-------------|
| `BASE_MODEL` | `"Qwen/Qwen2-0.5B-Instruct"` | Base model to fine-tune |
| `EPOCHS_GRPO` | `3` | Number of training epochs |
| `GRPO_BATCH_SIZE` | `2` | Training batch size (memory efficient) |
| `GRPO_LEARNING_RATE` | `5e-4` | Learning rate for LoRA training |
| `TRAIN_DATASET_SAMPLES` | `32` | Number of training samples |
| `OUTPUT_DIR` | `"./grpo_lora_output"` | Output directory for LoRA adapters |

#### LoRA Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | `16` | Rank of adaptation matrices |
| `lora_alpha` | `32` | LoRA scaling parameter |
| `target_modules` | `["q_proj", "v_proj"]` | Target attention modules |
| `lora_dropout` | `0.05` | Dropout probability |

#### Supervised Fine-tuning Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `""` | Path to save/load the model |
| `--dataset_name` | `"lvwerra/stack-exchange-paired"` | HuggingFace dataset name |
| `--max_steps` | `10000` | Maximum training steps |
| `--batch_size` | `4` | Training batch size |
| `--learning_rate` | `1e-4` | Learning rate |
| `--seq_length` | `1024` | Maximum sequence length |
| `--output_dir` | `"./checkpoints"` | Output directory for checkpoints |
| `--streaming` | `False` | Use streaming dataset loading |

#### GRPO Training Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `""` | Path to the base model |
| `--dataset_name` | `"lvwerra/stack-exchange-paired"` | Dataset for training |
| `--learning_rate` | `5e-7` | Learning rate for GRPO |
| `--output_dir` | `"./grpo_output"` | Output directory |
| `--num_train_epochs` | `1` | Number of training epochs |

## Model Architecture

### Qwen2-0.5B-Instruct
- **Parameters**: 495M total parameters
- **Architecture**: Transformer-based language model
- **Context Length**: 2048 tokens
- **Vocabulary**: 151,936 tokens
- **Fine-tuning**: Supports LoRA (Low-Rank Adaptation)

### LoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank of adaptation
    lora_alpha=32,          # LoRA scaling parameter
    target_modules=["q_proj", "v_proj"],  # Target attention modules
    lora_dropout=0.05,      # Dropout probability
    bias="none",            # Bias type
    task_type="CAUSAL_LM"   # Task type
)
```

## Training Results

### GRPO with LoRA Performance
Training configuration:
- **Total Model Parameters**: 495,127,552
- **Trainable LoRA Parameters**: 1,081,344 (0.22% of total)
- **Memory Efficiency**: 99.78% parameter reduction
- **Training Speed**: ~4s per step on Apple Silicon MPS
- **Output**: Saves LoRA adapters for efficient deployment

### Agent Task Performance
The model learns to:
- Parse observations with multiple item IDs: `<|obs|><ID_147> <ID_232> <ID_89><|agent|>`
- Generate chain-of-thought reasoning: `<THINK>The items are... I need to identify...<END_THINK>`
- Select correct actions: `<ID_147>` (first item selection task)

### Supervised Fine-tuning Performance
After 5 training steps with batch size 1:
- **Initial Loss**: 2.7793
- **Final Loss**: 2.1823
- **Token Accuracy**: 44% → 55%
- **Trainable Parameters**: 1,081,344 (0.22% of total)

### Training Metrics
The training script tracks several metrics:
- `loss`: Training loss per step
- `grad_norm`: Gradient norm for optimization monitoring
- `learning_rate`: Current learning rate
- `mean_token_accuracy`: Token-level prediction accuracy

## File Structure

```
trl-example/
├── README.md                    # This file
├── grpo-agent.py               # GRPO training with LoRA (recommended)
├── supervised_finetuning.py     # Supervised fine-tuning script  
├── grpo-train.py               # Standard GRPO training script
├── test_grpo_lora.py           # Test script for LoRA training
├── .gitignore                  # Git ignore patterns
├── checkpoints/                # Supervised fine-tuning checkpoints
│   └── final_checkpoint/       # Final model checkpoint
├── grpo_lora_output/           # GRPO LoRA training output
│   └── lora_adapter/           # Saved LoRA adapters
└── qwen2-saved/               # Downloaded model cache (created automatically)
    ├── config.json
    ├── tokenizer.json
    └── pytorch_model.bin
```

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   ```bash
   # Use LoRA training for memory efficiency (recommended)
   python grpo-agent.py
   
   # Or reduce batch size for supervised fine-tuning
   python supervised_finetuning.py --batch_size 1 --seq_length 512
   ```

2. **CUDA Out of Memory**:
   ```bash
   # Use LoRA training (uses only 0.22% of parameters)
   python grpo-agent.py
   
   # Or use CPU-only training
   export CUDA_VISIBLE_DEVICES=""
   python supervised_finetuning.py --model_path ./qwen2-saved
   ```

3. **Model Download Issues**:
   ```bash
   # Models are cached automatically, check internet connection
   # Cache location: ~/.cache/huggingface/
   ```

4. **LoRA Adapter Loading**:
   ```python
   # To load saved LoRA adapters
   from peft import PeftModel
   base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
   model = PeftModel.from_pretrained(base_model, "./grpo_lora_output/lora_adapter")
   ```

3. **Download Failures**:
   ```bash
   # Check internet connection and retry
   # Models are cached in ~/.cache/huggingface/
   ```

4. **Permission Errors**:
   ```bash
   # Ensure write permissions for output directories
   chmod 755 ./checkpoints
   ```

### Dependency Issues

The script has been optimized to avoid common dependency conflicts:
- ✅ **bitsandbytes**: 8-bit quantization disabled for compatibility
- ✅ **wandb**: Reporting disabled to avoid installation requirement
- ✅ **torch versions**: Compatible with both CPU and GPU environments

## Advanced Usage

### Loading Trained LoRA Adapters

After training with `grpo-agent.py`, you can load the LoRA adapters:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./grpo_lora_output/lora_adapter")

# Use for inference
prompt = "<|obs|><ID_147> <ID_232><|agent|><THINK>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

### Custom Dataset
For supervised fine-tuning, format your data similar to Stack Exchange format:
```json
{
  "question": "Your question here", 
  "response_j": "Preferred response",
  "response_k": "Alternative response"
}
```

For GRPO agent training, create tasks with observation-action pairs:
```python
def generate_custom_task():
    # Your custom task generation logic
    item_ids = [1, 2, 3]  # Example items
    target_id = 1         # Target to select
    return item_ids, target_id

def create_custom_dataset(num_samples):
    data_list = []
    for _ in range(num_samples):
        item_ids, target_id = generate_custom_task()
        prompt = format_observation_prompt(item_ids)
        data_list.append({
            "prompt": prompt,
            "target_id": target_id,
            "item_ids_str": " ".join(map(str, item_ids))
        })
    return Dataset.from_list(data_list)
```

### Multi-GPU Training
For multi-GPU setups:
```bash
accelerate config  # Configure accelerate
accelerate launch supervised_finetuning.py --model_path ./qwen2-saved
```

### Custom LoRA Configuration
Modify the LoRA settings in the script:
```python
lora_config = LoraConfig(
    r=32,                    # Increase rank for more capacity
    lora_alpha=64,          # Adjust scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More modules
    lora_dropout=0.1,       # Adjust dropout
)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit your changes: `git commit -m "Add feature"`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the original TRL license for details.

## Acknowledgments

- **TRL Library**: For providing the training framework
- **Hugging Face**: For the Transformers library and model hosting
- **Qwen Team**: For the Qwen2 model architecture
- **Stack Exchange**: For providing the training dataset

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{trl-qwen2-example,
  title={TRL Example: Qwen2 Fine-tuning with GRPO and Supervised Learning},
  author={TRL Example Contributors},
  year={2025},
  url={https://github.com/sk413025/trl-example}
}
```

## Contact

For questions or issues, please:
1. Check the [Issues](https://github.com/sk413025/trl-example/issues) page
2. Create a new issue with detailed information
3. Include your system information and error messages

---

**Happy Fine-tuning!** 🚀