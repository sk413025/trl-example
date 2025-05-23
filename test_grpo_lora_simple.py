#!/usr/bin/env python3
"""
Test script for simple GRPO LoRA training
=========================================

This script tests the simplified GRPO LoRA training with minimal parameters
for quick verification.
"""

import subprocess
import sys
import os
import time

def main():
    print("🚀 Testing Simple GRPO with LoRA training...")
    print("=" * 60)
    
    # Check if the script exists
    script_path = "grpo-lora-train.py"
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found!")
        return 1
    
    try:
        start_time = time.time()
        
        # Run the GRPO LoRA script
        print("📁 Running simple GRPO with LoRA training...")
        print("⏰ This may take a few minutes...")
        
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print output
        if result.stdout:
            print("📄 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ Simple GRPO LoRA training completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            
            # Check if LoRA adapter was saved
            lora_path = "./grpo_lora_simple_output/lora_adapter"
            if os.path.exists(lora_path):
                print(f"💾 LoRA adapter saved to: {lora_path}")
                
                # List saved files
                saved_files = os.listdir(lora_path)
                print(f"📁 Saved files: {', '.join(saved_files)}")
                
                # Check file sizes
                total_size = 0
                for file in saved_files:
                    file_path = os.path.join(lora_path, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        total_size += size
                        print(f"   📄 {file}: {size:,} bytes")
                
                print(f"📊 Total adapter size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
                
                # Test loading the adapter
                print("\n🧪 Testing LoRA adapter loading...")
                test_loading(lora_path)
                
            else:
                print("⚠️  LoRA adapter directory not found")
            
            return 0
        else:
            print(f"❌ Script failed with return code: {result.returncode}")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⏰ Script timed out after 10 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return 1

def test_loading(lora_path):
    """Test loading the trained LoRA adapter"""
    try:
        # Import required libraries
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print("   📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("   🔌 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("   🎯 Testing inference...")
        prompt = "Summarize this: The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   💬 Generated response: {response[len(prompt):]}")
        
        print("   ✅ LoRA adapter loaded and tested successfully!")
        
    except ImportError as e:
        print(f"   ⚠️  Import error (expected if torch not available): {e}")
    except Exception as e:
        print(f"   ❌ Error testing adapter: {e}")

if __name__ == "__main__":
    # Add torch import check
    try:
        import torch
        print("🔥 PyTorch available")
    except ImportError:
        print("⚠️  PyTorch not available - loading test will be skipped")
    
    exit_code = main()
    sys.exit(exit_code)