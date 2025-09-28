#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic GGUF Conversion and Upload Script
Converts trained model to GGUF format and uploads to Hugging Face
"""

from unsloth import FastLanguageModel
import os
import json
from huggingface_hub import HfApi, create_repo

def create_gguf_directory():
    """Create gguf directory if it doesn't exist"""
    os.makedirs("gguf", exist_ok=True)
    print("Created gguf/ directory")

def load_config():
    """Load configuration from algebra.json"""
    with open('algebra.json', 'r') as f:
        config = json.load(f)
    return config

def load_model(config):
    """Load the fine-tuned model from config"""
    model_config = config['model_config']
    model_name = model_config['hub_model_name']
    max_seq_length = model_config['max_seq_length']
    load_in_4bit = model_config['load_in_4bit']
    
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    print("Model loaded successfully!")
    return model, tokenizer

def convert_to_gguf(model, tokenizer, quantization_type="Q8_0", base_model_name="model"):
    """Convert model to GGUF format"""
    # Variables
    gguf_directory = "gguf"
    
    # Create the main gguf directory if it doesn't exist
    os.makedirs(gguf_directory, exist_ok=True)
    
    # Create a temporary directory for the model, then convert to GGUF
    temp_model_dir = f"temp_model_{quantization_type.lower()}"
    final_filename = f"{gguf_directory}/{base_model_name}-{quantization_type.lower()}.gguf"
    
    print(f"Converting to GGUF format with {quantization_type} quantization...")
    print(f"Target file: {final_filename}")
    
    try:
        # First save the model in standard format to create proper directory structure
        print("Saving model in standard format...")
        model.save_pretrained(temp_model_dir)
        tokenizer.save_pretrained(temp_model_dir)
        
        # Now convert the saved model to GGUF
        print("Converting to GGUF...")
        model.save_pretrained_gguf(
            temp_model_dir,
            tokenizer,
            quantization_type=quantization_type,
        )
        
        # Move the GGUF file to the final location
        import glob
        import shutil
        
        # Look for GGUF files both in the temp directory and current directory
        gguf_files = glob.glob(f"{temp_model_dir}/*.gguf") + glob.glob(f"{temp_model_dir}.*.gguf")
        
        if gguf_files:
            shutil.move(gguf_files[0], final_filename)
            print(f"GGUF file moved to: {final_filename}")
        else:
            print(f"Warning: No GGUF file found in {temp_model_dir}")
            # List what files were created for debugging
            if os.path.exists(temp_model_dir):
                files = os.listdir(temp_model_dir)
                print(f"Files in {temp_model_dir}: {files}")
            # Also check current directory for gguf files with temp_model prefix
            current_dir_gguf = glob.glob(f"{temp_model_dir}*.gguf")
            if current_dir_gguf:
                print(f"Found GGUF files in current directory: {current_dir_gguf}")
                shutil.move(current_dir_gguf[0], final_filename)
                print(f"GGUF file moved to: {final_filename}")
                # Clean up any remaining temp gguf files
                for f in current_dir_gguf[1:]:
                    os.remove(f)
        
        # Clean up temporary directory
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        # Clean up on error
        import shutil
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir, ignore_errors=True)
        raise
    
    return final_filename

def upload_to_hf(gguf_files, repo_name, hf_token):
    """Upload GGUF files to Hugging Face"""
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables. Skipping upload.")
        return
    
    try:
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, token=hf_token, exist_ok=True)
            print(f"Repository {repo_name} ready")
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Upload each GGUF file
        for gguf_file in gguf_files:
            if os.path.exists(gguf_file):
                filename = os.path.basename(gguf_file)
                print(f"Uploading {filename} to {repo_name}...")
                api.upload_file(
                    path_or_fileobj=gguf_file,
                    path_in_repo=filename,
                    repo_id=repo_name,
                    token=hf_token
                )
                print(f"✓ Uploaded {filename}")
            else:
                print(f"Warning: {gguf_file} not found")
        
        print(f"\n🎉 All GGUF files uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

def main():
    """Main conversion function - fully automated"""
    # Load configuration
    config = load_config()
    model_config = config['model_config']
    
    # Get model name and create GGUF repo name
    hub_model_name = model_config['hub_model_name']
    base_model_name = hub_model_name.split('/')[-1]  # Get just the model name part
    gguf_repo_name = f"{hub_model_name}-gguf"
    
    print(f"Source model: {hub_model_name}")
    print(f"GGUF repository: {gguf_repo_name}")
    
    # Quantization types to convert
    quantization_types = [
        "F32",      # Full 32-bit precision (largest file)
        "F16",      # Full 16-bit precision 
        "BF16",     # Brain Float 16-bit precision
        "Q8_0",     # 8-bit quantization (good balance)
        "Q4_0",     # 4-bit quantization (smaller)
        "Q2_K",     # 2-bit quantization (smallest)
    ]
    
    # Create gguf directory
    create_gguf_directory()
    
    # Load model
    model, tokenizer = load_model(config)
    
    print(f"\nConverting all quantization types automatically...")
    gguf_files = []
    
    # Convert all quantization types
    for qtype in quantization_types:
        try:
            print(f"\n{'='*50}")
            print(f"Converting {qtype}...")
            gguf_file = convert_to_gguf(model, tokenizer, qtype, base_model_name)
            if os.path.exists(gguf_file):
                gguf_files.append(gguf_file)
                print(f"✓ {qtype} conversion completed: {gguf_file}")
            else:
                print(f"✗ {qtype} conversion failed")
        except Exception as e:
            print(f"✗ Error converting {qtype}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"GGUF conversion completed!")
    print(f"Successfully converted {len(gguf_files)} out of {len(quantization_types)} formats")
    
    # Upload to Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            hf_token = None
    
    if gguf_files:
        print(f"\nUploading to Hugging Face: {gguf_repo_name}")
        upload_to_hf(gguf_files, gguf_repo_name, hf_token)
    else:
        print("\nNo GGUF files to upload")
    
    print("\n🎉 Process completed!")
    if gguf_files:
        print(f"📁 Local files in: gguf/")
        if hf_token:
            print(f"🤗 Hugging Face: https://huggingface.co/{gguf_repo_name}")
        for f in gguf_files:
            print(f"   - {os.path.basename(f)}")

if __name__ == "__main__":
    main()