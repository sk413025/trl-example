#!/usr/bin/env python3
"""
SparkTTS Base Model Test
========================

Test the base SparkTTS model without LoRA to verify it works.
"""

import os
import sys
import torch
import numpy as np
import re
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
BASE_MODEL_NAME = "Spark-TTS-0.5B"
OUTPUT_DIR = "./generated_audio"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🎵 SparkTTS Base Model Test")
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

# Load base model without LoRA
print("🤖 Loading base model (no LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_NAME}/LLM")
model = AutoModelForCausalLM.from_pretrained(
    f"{BASE_MODEL_NAME}/LLM",
    torch_dtype=torch.float32,
)
model = model.to(device)
model.eval()

print("✅ Base model loaded successfully!")

# Initialize audio tokenizer
print("🎵 Initializing audio tokenizer...")
audio_tokenizer = BiCodecTokenizer(BASE_MODEL_NAME, device.type)

# Test text
test_text = "Hello world. This is a test."
print(f"\n📝 Test text: {test_text}")

# Format prompt
prompt = f"<|task_tts|><|start_content|>{test_text}<|end_content|><|start_global_token|>"

# Tokenize
inputs = tokenizer([prompt], return_tensors="pt").to(device)
print(f"📊 Input length: {inputs.input_ids.shape[1]} tokens")

print("\n🔄 Generating with base model...")

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
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
        print(f"   First semantic tokens: {semantic_matches[:10]}...")
        print(f"   First global tokens: {global_matches[:10] if global_matches else 'None'}...")
        
        # Decode audio
        try:
            pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
            
            if global_matches:
                pred_global_ids = torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0)
            else:
                pred_global_ids = torch.zeros((1, 1, 1), dtype=torch.long)
            
            # Move tokenizer to device
            audio_tokenizer.model.to(device)
            
            # Decode
            wav_np = audio_tokenizer.detokenize(
                pred_global_ids.to(device).squeeze(0),
                pred_semantic_ids.to(device)
            )
            
            if wav_np.size > 0:
                filename = "base_model_output.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)
                sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
                sf.write(filepath, wav_np, sample_rate)
                print(f"✅ Audio saved to: {filepath}")
                print(f"📊 Duration: {len(wav_np) / sample_rate:.2f} seconds")
                print("\n🎉 Base model works correctly!")
            else:
                print("⚠️ Decoded audio is empty")
                
        except Exception as e:
            print(f"⚠️ Audio decoding failed: {e}")
    else:
        print("⚠️ No semantic tokens generated")
        # Show what was actually generated
        print("\n📄 Generated text preview:")
        print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
        
except Exception as e:
    print(f"❌ Generation failed: {e}")

print("\n✅ Test complete!")