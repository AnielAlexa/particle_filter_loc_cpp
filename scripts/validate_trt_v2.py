#!/usr/bin/env python3
"""Validate TRT engine using torch CUDA tensors (no pycuda)."""
import numpy as np
import cv2
import os
import torch
import tensorrt as trt

MODEL_DIR = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/model"
engine_path = os.path.join(MODEL_DIR, "dinov3_vlad_pca768_l10_value_256.engine")

# Load engine
logger = trt.Logger(trt.Logger.INFO)
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

# Allocate torch CUDA tensors
input_t = torch.randn(1, 3, 256, 256, dtype=torch.float32, device='cuda').contiguous()
output_t = torch.zeros(1, 768, dtype=torch.float32, device='cuda').contiguous()

# Set addresses
context.set_tensor_address("input", input_t.data_ptr())
context.set_tensor_address("descriptor", output_t.data_ptr())

# Create a CUDA stream
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    print(f"execute_async_v3 returned: {ok}")
stream.synchronize()

desc = output_t.cpu().numpy().flatten()
print(f"Random input => norm={np.linalg.norm(desc):.4f}, first5={desc[:5]}")

# Now try with a real patch
db_path = os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/vlad_descriptors_pca768.npy")
names_path = os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/patch_names.txt")
db = np.load(db_path).astype(np.float32)
db = db / np.linalg.norm(db, axis=1, keepdims=True).clip(1e-8)
with open(names_path) as f:
    names = [l.strip() for l in f if l.strip()]

patch_name = names[0]
patch_path = os.path.join(MODEL_DIR, f"data/patches/{patch_name}.png")
img = cv2.imread(patch_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_resized = cv2.resize(gray, (256, 256))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inp = np.zeros((1, 3, 256, 256), dtype=np.float32)
for c in range(3):
    inp[0, c] = (gray_resized.astype(np.float32) / 255.0 - mean[c]) / std[c]

input_t.copy_(torch.from_numpy(inp).cuda())
with torch.cuda.stream(stream):
    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    print(f"\nPatch '{patch_name}' => execute_async_v3: {ok}")
stream.synchronize()

desc = output_t.cpu().numpy().flatten()
desc_norm = np.linalg.norm(desc)
print(f"  norm={desc_norm:.4f}, first5={desc[:5]}, nonzero={np.count_nonzero(np.abs(desc) > 1e-8)}/{len(desc)}")

if desc_norm > 1e-6:
    desc = desc / desc_norm
    sims = db @ desc
    top_idx = np.argsort(-sims)[:5]
    print(f"  Top-5 matches:")
    for i, idx in enumerate(top_idx):
        print(f"    {names[idx]}: sim={sims[idx]:.4f}")
else:
    print("  OUTPUT IS ALL ZEROS - engine execution failed!")

    # Try execute_v2 as fallback
    print("\n  Trying execute_v2 fallback...")
    import ctypes
    bindings = [input_t.data_ptr(), output_t.data_ptr()]
    ok2 = context.execute_v2(bindings)
    print(f"  execute_v2 returned: {ok2}")
    desc2 = output_t.cpu().numpy().flatten()
    print(f"  norm={np.linalg.norm(desc2):.4f}, first5={desc2[:5]}")
