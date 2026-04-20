#!/usr/bin/env python3
"""Quick validation: TRT VLAD engine vs database descriptors.
Loads a satellite patch, runs it through the TRT engine, and checks
if cosine similarity is reasonable (should be >0.5 for the matching patch).
"""
import numpy as np
import cv2
import sys
import os

MODEL_DIR = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/model"

# Load database
db_path = os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/vlad_descriptors_pca768.npy")
names_path = os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/patch_names.txt")

db = np.load(db_path).astype(np.float32)
db = db / np.linalg.norm(db, axis=1, keepdims=True).clip(1e-8)
with open(names_path) as f:
    names = [l.strip() for l in f if l.strip()]
print(f"Database: {db.shape} ({len(names)} names)")
print(f"  DB descriptor norms: min={np.linalg.norm(db, axis=1).min():.4f} max={np.linalg.norm(db, axis=1).max():.4f}")
print(f"  DB first 5 values of desc[0]: {db[0, :5]}")

# Load TRT engine
engine_path = os.path.join(MODEL_DIR, "dinov3_vlad_pca768_l10_value_256.engine")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("Need tensorrt + pycuda. Try: pip install pycuda")
    sys.exit(1)

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Print tensor info
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f"  Tensor '{name}': shape={shape} dtype={dtype} mode={mode}")

# Allocate buffers
import torch
input_tensor = torch.zeros(1, 3, 256, 256, dtype=torch.float32, device='cuda')
output_tensor = torch.zeros(1, 768, dtype=torch.float32, device='cuda')

context.set_tensor_address("input", input_tensor.data_ptr())
context.set_tensor_address("descriptor", output_tensor.data_ptr())

# Test 1: Load a satellite patch, preprocess like C++ does (gray->3ch, ImageNet norm)
patch_name = names[0]
patch_path = os.path.join(MODEL_DIR, f"data/patches/{patch_name}.png")
print(f"\n=== Test 1: Self-matching patch '{patch_name}' ===")
img = cv2.imread(patch_path)
if img is None:
    print(f"ERROR: Cannot read {patch_path}")
    sys.exit(1)

# Convert to gray, resize to 256x256
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)

# ImageNet normalize (same as C++ vpi_preprocessor)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
input_np = np.zeros((1, 3, 256, 256), dtype=np.float32)
for c in range(3):
    input_np[0, c] = (gray_resized.astype(np.float32) / 255.0 - mean[c]) / std[c]

print(f"  Input stats: min={input_np.min():.4f} max={input_np.max():.4f} mean={input_np.mean():.4f}")

input_tensor.copy_(torch.from_numpy(input_np).cuda())

# Run inference
context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

desc = output_tensor.cpu().numpy().flatten()
desc_norm = np.linalg.norm(desc)
print(f"  Output norm: {desc_norm:.4f} (should be ~1.0)")
print(f"  Output first 5: {desc[:5]}")
print(f"  Output nonzero: {np.count_nonzero(np.abs(desc) > 1e-8)}/{len(desc)}")

# L2 normalize
desc = desc / max(desc_norm, 1e-8)

# Compute cosine sim against all database entries
sims = db @ desc
top_idx = np.argsort(-sims)[:10]

print(f"\n  Top-10 matches for '{patch_name}':")
for i, idx in enumerate(top_idx):
    marker = " <-- SELF" if names[idx] == patch_name else ""
    print(f"    [{i}] {names[idx]}: sim={sims[idx]:.4f}{marker}")

# Test 2: Random noise (should give low similarity)
print(f"\n=== Test 2: Random noise input ===")
noise = np.random.randn(1, 3, 256, 256).astype(np.float32) * 0.1
input_tensor.copy_(torch.from_numpy(noise).cuda())
context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
desc_noise = output_tensor.cpu().numpy().flatten()
desc_noise = desc_noise / max(np.linalg.norm(desc_noise), 1e-8)
sims_noise = db @ desc_noise
print(f"  Max sim from noise: {sims_noise.max():.4f} (should be low ~0.1)")
print(f"  Mean sim from noise: {sims_noise.mean():.4f}")

# Test 3: Check if database descriptors are correct by computing self-similarity
print(f"\n=== Test 3: Database self-similarity check ===")
self_sims = db @ db.T
diag = np.diag(self_sims)
print(f"  Diagonal (self-sim): min={diag.min():.4f} max={diag.max():.4f} mean={diag.mean():.4f}")
# Off-diagonal
off_diag = self_sims[~np.eye(len(names), dtype=bool)]
print(f"  Off-diagonal: min={off_diag.min():.4f} max={off_diag.max():.4f} mean={off_diag.mean():.4f}")
