#!/usr/bin/env python3
"""
SparkTTS Simple Test Script
===========================

A simple test script to generate speech using the fine-tuned LoRA model.
"""

import os
import sys
import torch
# import numpy as np  # Not used in this script
import re
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "Spark-TTS-0.5B"
LORA_ADAPTER_PATH = "./spark_tts_conservative_output/lora_adapter"  # Use the conservative model
OUTPUT_DIR = "./generated_audio"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🎵 SparkTTS Test Generation")
print("=" * 40)

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using MPS device")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

# Add Spark-TTS to path
spark_tts_path = os.path.join(os.getcwd(), 'Spark-TTS')
sys.path.insert(0, spark_tts_path)

from sparktts.models.audio_tokenizer import BiCodecTokenizer

# Load models
print("🤖 Loading models...")
tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_NAME}/LLM")
base_model = AutoModelForCausalLM.from_pretrained(
    f"{BASE_MODEL_NAME}/LLM",
    torch_dtype=torch.float32,
)

# Only use the conservative model
checkpoints = [
    LORA_ADAPTER_PATH,  # Conservative model
]

model = None
for checkpoint in checkpoints:
    if os.path.exists(checkpoint):
        try:
            print(f"🔧 Trying to load LoRA from: {checkpoint}")
            model = PeftModel.from_pretrained(base_model, checkpoint)
            model = model.to(device)
            model.eval()
            print(f"✅ Successfully loaded checkpoint: {checkpoint}")
            break
        except Exception as e:
            print(f"⚠️ Failed to load {checkpoint}: {e}")

if model is None:
    print("❌ Failed to load any checkpoint!")
    sys.exit(1)

# Initialize audio tokenizer
print("🎵 Initializing audio tokenizer...")
audio_tokenizer = BiCodecTokenizer(BASE_MODEL_NAME, device.type)

# Test text
test_text = "Hello world. This is a test of SparkTTS with LoRA fine-tuning."
# test_text = "你好，世界。这是 SparkTTS 和 LoRA 微调的测试。"
print(f"\n📝 Test text: {test_text}")

# Format prompt
prompt = f"<|task_tts|><|start_content|>{test_text}<|end_content|><|start_global_token|>"

# Tokenize
inputs = tokenizer([prompt], return_tensors="pt").to(device)
print(f"📊 Input length: {inputs.input_ids.shape[1]} tokens")

# Try different generation strategies
generation_configs = [
    {
        "name": "Conservative sampling",
        "params": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.5,
            "top_k": 20,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
    },
    {
        "name": "Very conservative",
        "params": {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.3,
            "top_k": 10,
            "top_p": 0.8,
            "repetition_penalty": 1.2,
        }
    },
    {
        "name": "Greedy decoding",
        "params": {
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
        }
    },
]

for i, config in enumerate(generation_configs):
    print(f"\n🔄 Trying generation strategy {i+1}: {config['name']}")
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **config["params"]
            )
        
        # Extract generated text
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        print(f"📝 Generated {len(generated_ids[0])} tokens")
        
        # Extract tokens
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", generated_text)
        
        print(f"✅ Found {len(semantic_matches)} semantic tokens, {len(global_matches)} global tokens")
        
        if semantic_matches:
            # Show first few tokens for debugging
            print(f"   First semantic tokens: {semantic_matches[:10]}...")
            print(f"   First global tokens: {global_matches[:10] if global_matches else 'None'}...")
            
            # Try to decode audio
            try:
                pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
                
                if global_matches:
                    pred_global_ids = torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0)
                else:
                    # Use default global token
                    pred_global_ids = torch.zeros((1, 1, 1), dtype=torch.long)
                
                # Move tokenizer to CPU for MPS compatibility
                audio_tokenizer.model.to("cpu")
                audio_tokenizer.device = "cpu"
                
                # Decode on CPU
                wav_np = audio_tokenizer.detokenize(
                    pred_global_ids.to("cpu").squeeze(0),
                    pred_semantic_ids.to("cpu")
                )
                
                if wav_np.size > 0:
                    filename = f"test_output_strategy_{i+1}.wav"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
                    sf.write(filepath, wav_np, sample_rate)
                    print(f"✅ Audio saved to: {filepath}")
                    print(f"📊 Duration: {len(wav_np) / sample_rate:.2f} seconds")
                    break  # Success!
                else:
                    print("⚠️ Decoded audio is empty")
                    
            except Exception as e:
                print(f"⚠️ Audio decoding failed: {e}")
        else:
            print("⚠️ No semantic tokens generated")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")

print("\n✅ Test complete!")