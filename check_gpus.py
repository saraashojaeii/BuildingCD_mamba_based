#!/usr/bin/env python3

import torch
import os

print("=== GPU Detection Script ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Test if we can create tensors on different GPUs
print("\n=== Testing GPU Access ===")
try:
    for i in range(min(4, torch.cuda.device_count())):
        device = f"cuda:{i}"
        x = torch.randn(10, 10).to(device)
        print(f"✓ Successfully created tensor on {device}")
except Exception as e:
    print(f"✗ Error accessing GPU: {e}")

print("\n=== Environment Variables ===")
for key in os.environ:
    if 'CUDA' in key or 'GPU' in key or 'NCCL' in key:
        print(f"{key}: {os.environ[key]}")
