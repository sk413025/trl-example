#!/usr/bin/env python3
"""
SparkTTS Inference with Fine-tuned LoRA Model
=============================================

This script uses the fine-tuned LoRA adapter to generate speech from text.

Usage:
    python spark_tts_inference.py

Dependencies:
    pip install torch transformers peft torchaudio soundfile omegaconf einx einops
"""

import os
import sys
import torch
import numpy as np
import re
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torchaudio.transforms as T

# Configuration
BASE_MODEL_NAME = "Spark-TTS-0.5B"
LORA_ADAPTER_PATH = "./spark_tts_lora_output/lora_adapter"  # Path to your LoRA adapter
OUTPUT_DIR = "./generated_audio"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🎵 SparkTTS Inference with LoRA")
print("=" * 40)

# Set device for macOS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using MPS (Metal Performance Shaders) for macOS")
else:
    device = torch.device("cpu")
    print("💻 MPS not available, using CPU")

# Add Spark-TTS directory to path for imports
spark_tts_path = os.path.join(os.getcwd(), 'Spark-TTS')
if os.path.exists(spark_tts_path):
    sys.path.insert(0, spark_tts_path)
else:
    print("❌ Spark-TTS not found. Please ensure it's cloned in the current directory.")
    sys.exit(1)

# Import SparkTTS components
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

# Load base model
print(f"🤖 Loading base model: {BASE_MODEL_NAME}/LLM")
tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_NAME}/LLM")
base_model = AutoModelForCausalLM.from_pretrained(
    f"{BASE_MODEL_NAME}/LLM",
    torch_dtype=torch.float32,  # SparkTTS requires float32
)

# Load LoRA adapter
print(f"🔧 Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.to(device)
model.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")

# Initialize audio tokenizer
print("🎵 Initializing audio tokenizer...")
audio_tokenizer = BiCodecTokenizer(BASE_MODEL_NAME, device.type)

@torch.no_grad()
def generate_speech(
    text: str,
    speaker: str = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.1,
) -> np.ndarray:
    """
    Generate speech from text using the fine-tuned model.
    
    Args:
        text: Input text to synthesize
        speaker: Optional speaker name for multi-speaker models
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum number of tokens to generate
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        Audio waveform as numpy array
    """
    # Format prompt
    if speaker:
        formatted_text = f"{speaker}: {text}"
    else:
        formatted_text = text
    
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        formatted_text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])
    
    print(f"📝 Input text: {text}")
    if speaker:
        print(f"🎭 Speaker: {speaker}")
    
    # Tokenize input
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    print("🔄 Generating audio tokens...")
    
    # Generate tokens with safer parameters
    try:
        # Set model to float32 for stability
        model.float()
        
        # Generate with conservative parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 0.1),  # Ensure minimum temperature
            top_k=max(top_k, 10),  # Ensure minimum top_k
            top_p=min(top_p, 0.99),  # Cap top_p
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bad_words_ids=None,
            force_words_ids=None,
            no_repeat_ngram_size=0,  # Disable n-gram blocking
            renormalize_logits=True,  # Renormalize to avoid numerical issues
        )
    except RuntimeError as e:
        print(f"⚠️ Generation failed, trying with more conservative parameters...")
        # Fallback to very conservative parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, 1024),
            do_sample=False,  # Use greedy decoding
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract generated tokens
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    print("📊 Extracting audio tokens...")
    
    # Extract semantic tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
    if not semantic_matches:
        print("⚠️ No semantic tokens found in generated output.")
        return np.array([], dtype=np.float32)
    
    # Extract global tokens
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", generated_text)
    if not global_matches:
        print("⚠️ No global tokens found in generated output.")
        # Use default global tokens
        global_matches = ["0"]  # Default to single zero token
    
    print(f"✅ Found {len(semantic_matches)} semantic tokens and {len(global_matches)} global tokens")
    
    # Convert to tensors
    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0).unsqueeze(0)
    
    # Move audio tokenizer to device
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    
    print("🎼 Decoding audio...")
    
    # Decode audio
    try:
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.to(device).squeeze(0),  # Shape (1, N_global)
            pred_semantic_ids.to(device)            # Shape (1, N_semantic)
        )
        print("✅ Audio decoded successfully!")
        return wav_np
    except Exception as e:
        print(f"❌ Error during audio decoding: {e}")
        return np.array([], dtype=np.float32)

def main():
    """Main function for interactive text-to-speech generation."""
    print("\n" + "=" * 40)
    print("🎤 SparkTTS Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 40 + "\n")
    
    # Sample texts for testing
    sample_texts = [
        "Hello, this is a test of the SparkTTS model with LoRA fine-tuning.",
        "The weather today is absolutely beautiful, perfect for a walk in the park.",
        "Artificial intelligence is transforming the way we interact with technology.",
        "Welcome to the future of text-to-speech synthesis!",
    ]
    
    print("📝 Sample texts (or enter your own):")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    print()
    
    while True:
        # Get user input
        user_input = input("Enter text (or number 1-4 for samples, 'quit' to exit): ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        # Check if user selected a sample
        if user_input.isdigit() and 1 <= int(user_input) <= len(sample_texts):
            text = sample_texts[int(user_input) - 1]
        else:
            text = user_input
        
        if not text:
            print("⚠️ Please enter some text.")
            continue
        
        # Optional: Get speaker name for multi-speaker models
        speaker = input("Enter speaker name (or press Enter for default): ").strip() or None
        
        # Generate speech
        print("\n🎯 Generating speech...")
        try:
            waveform = generate_speech(
                text=text,
                speaker=speaker,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                max_new_tokens=2048,
            )
            
            if waveform.size > 0:
                # Save audio file
                timestamp = torch.randint(0, 1000000, (1,)).item()
                filename = f"generated_{timestamp}.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
                sf.write(filepath, waveform, sample_rate)
                
                print(f"✅ Audio saved to: {filepath}")
                print(f"📊 Duration: {len(waveform) / sample_rate:.2f} seconds")
                
                # Optional: Play audio if in Jupyter/Colab
                try:
                    from IPython.display import Audio, display
                    display(Audio(waveform, rate=sample_rate))
                except ImportError:
                    pass
            else:
                print("❌ Failed to generate audio.")
        
        except Exception as e:
            print(f"❌ Error during generation: {e}")
        
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    # Test with a simple example first
    print("\n🧪 Running test generation...")
    test_text = "Hello, this is a test of the fine-tuned SparkTTS model."
    
    try:
        waveform = generate_speech(test_text, temperature=0.8)
        
        if waveform.size > 0:
            test_file = os.path.join(OUTPUT_DIR, "test_output.wav")
            sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
            sf.write(test_file, waveform, sample_rate)
            print(f"✅ Test audio saved to: {test_file}")
        else:
            print("⚠️ Test generation produced empty audio.")
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
    
    # Start interactive mode
    main()