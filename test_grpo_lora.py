#!/usr/bin/env python3
"""
Test script for GRPO with LoRA training
=======================================

This script tests the LoRA-enabled GRPO training with minimal parameters
for quick verification.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Testing GRPO with LoRA training...")
    print("=" * 50)
    
    # Check if the script exists
    script_path = "grpo-lora-agent.py"
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found!")
        return 1
    
    try:
        # Run the GRPO LoRA script
        print("📁 Running GRPO with LoRA training...")
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Print output
        if result.stdout:
            print("📄 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ GRPO LoRA training completed successfully!")
            
            # Check if LoRA adapter was saved
            lora_path = "./grpo_lora_output/lora_adapter"
            if os.path.exists(lora_path):
                print(f"💾 LoRA adapter saved to: {lora_path}")
                
                # List saved files
                saved_files = os.listdir(lora_path)
                print(f"📁 Saved files: {', '.join(saved_files)}")
            else:
                print("⚠️  LoRA adapter directory not found")
            
            return 0
        else:
            print(f"❌ Script failed with return code: {result.returncode}")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⏰ Script timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)