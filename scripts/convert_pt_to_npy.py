#!/usr/bin/env python3
"""One-time conversion: torch .pt descriptor database -> numpy .npy for C++ loading."""

import sys
import numpy as np
import torch

def convert(pt_path: str):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        # Try common keys
        for key in ("descriptors", "boq_descriptors"):
            if key in data:
                desc = data[key]
                break
        else:
            desc = list(data.values())[0]
    else:
        desc = data

    desc = desc.float().numpy()
    # L2 normalize
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    desc = desc / norms

    out_path = pt_path.replace(".pt", ".npy")
    np.save(out_path, desc.astype(np.float32))
    print(f"Converted {pt_path} -> {out_path}")
    print(f"  Shape: {desc.shape}, dtype: float32")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <descriptors.pt> [descriptors2.pt ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        convert(path)
