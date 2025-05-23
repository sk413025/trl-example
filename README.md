# TRL Example: Qwen2 Fine-tuning with GRPO and Supervised Learning

This repository demonstrates fine-tuning techniques for the Qwen2-0.5B-Instruct model using TRL (Transformer Reinforcement Learning) library. It includes implementations for both GRPO (Group Relative Policy Optimization) and supervised fine-tuning approaches.

## Overview

The project provides two main training scripts:
- **GRPO Training**: Implements Group Relative Policy Optimization for preference-based learning
- **Supervised Fine-tuning**: Traditional supervised learning approach with automatic model downloading

## Features

- 🤖 **Automatic Model Download**: Downloads Qwen2-0.5B-Instruct model from Hugging Face
- 📊 **Multiple Training Methods**: Supports both GRPO and supervised fine-tuning
- 🔧 **Compatibility Fixes**: Resolved dependency issues for modern environments
- 💾 **Checkpoint Saving**: Automatic model checkpoint saving
- 📈 **Training Monitoring**: Built-in loss and accuracy tracking

## Requirements

### Dependencies
```bash
pip install torch transformers datasets accelerate peft trl
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
   pip install torch transformers datasets accelerate peft trl
   ```

3. **Verify installation**:
   ```bash
   python -c "import transformers, trl; print('Setup complete!')"
   ```

## Usage

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

### GRPO Training

The GRPO script implements preference-based training using policy optimization.

**Basic usage**:
```bash
python grpo-train.py --model_path ./Qwen2-0.5B-Instruct
```

### Command Line Arguments

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
├── supervised_finetuning.py     # Supervised fine-tuning script
├── grpo-train.py               # GRPO training script
├── .gitignore                  # Git ignore patterns
├── checkpoints/                # Training checkpoints (created during training)
│   └── final_checkpoint/       # Final model checkpoint
└── qwen2-saved/               # Downloaded model cache (created automatically)
    ├── config.json
    ├── tokenizer.json
    └── pytorch_model.bin
```

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   ```bash
   # Reduce batch size and sequence length
   python supervised_finetuning.py --batch_size 1 --seq_length 512
   ```

2. **CUDA Out of Memory**:
   ```bash
   # Use CPU-only training
   export CUDA_VISIBLE_DEVICES=""
   python supervised_finetuning.py --model_path ./qwen2-saved
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

### Custom Dataset
To use your own dataset:

1. Format your data similar to Stack Exchange format:
   ```json
   {
     "question": "Your question here",
     "response_j": "Preferred response",
     "response_k": "Alternative response"
   }
   ```

2. Modify the dataset loading:
   ```python
   dataset = load_dataset("path/to/your/dataset")
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