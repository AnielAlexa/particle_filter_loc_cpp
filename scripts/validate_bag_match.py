#!/usr/bin/env python3
"""Extract a few drone frames from bag, run through TRT, check sims."""
import numpy as np
import cv2
import os
import torch
import tensorrt as trt
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

MODEL_DIR = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/model"
BAG_PATH = "/home/jetson/bags_remote/Day2.1/Day2.1_0.mcap"

# Load database
db = np.load(os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/vlad_descriptors_pca768.npy")).astype(np.float32)
db = db / np.linalg.norm(db, axis=1, keepdims=True).clip(1e-8)
with open(os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/patch_names.txt")) as f:
    names = [l.strip() for l in f if l.strip()]

# Load GPS metadata
import json
with open(os.path.join(MODEL_DIR, "data/patches/gps_metadata.json")) as f:
    gps_meta = json.load(f)

# Load TRT engine
engine_path = os.path.join(MODEL_DIR, "dinov3_vlad_pca768_l10_value_256.engine")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

input_t = torch.zeros(1, 3, 256, 256, dtype=torch.float32, device='cuda').contiguous()
output_t = torch.zeros(1, 768, dtype=torch.float32, device='cuda').contiguous()
context.set_tensor_address("input", input_t.data_ptr())
context.set_tensor_address("descriptor", output_t.data_ptr())
stream = torch.cuda.Stream()

mean = [0.485, 0.456, 0.406]
std_ = [0.229, 0.224, 0.225]

def preprocess_and_match(gray_img, label=""):
    """Preprocess gray image and match against database."""
    # Center crop to square
    h, w = gray_img.shape
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    cropped = gray_img[y0:y0+s, x0:x0+s]
    resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)

    inp = np.zeros((1, 3, 256, 256), dtype=np.float32)
    for c in range(3):
        inp[0, c] = (resized.astype(np.float32) / 255.0 - mean[c]) / std_[c]

    input_t.copy_(torch.from_numpy(inp).cuda())
    with torch.cuda.stream(stream):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    desc = output_t.cpu().numpy().flatten()
    desc = desc / max(np.linalg.norm(desc), 1e-8)

    sims = db @ desc
    top_idx = np.argsort(-sims)[:5]

    print(f"\n{label}")
    for i, idx in enumerate(top_idx):
        lat = gps_meta.get(names[idx], {}).get('lat', 0)
        lon = gps_meta.get(names[idx], {}).get('lon', 0)
        print(f"  [{i}] {names[idx]}: sim={sims[idx]:.4f} (lat={lat:.4f}, lon={lon:.4f})")
    return sims

# Read bag and extract frames + RTK positions
print("Reading bag...")
frame_count = 0
rtk_positions = []
with Reader(BAG_PATH) as reader:
    # First pass: get RTK positions
    for conn, timestamp, rawdata in reader.messages():
        if conn.topic == '/m300/rtk/fix':
            msg = deserialize_cdr(rawdata, conn.msgtype)
            rtk_positions.append((timestamp, msg.latitude, msg.longitude))

print(f"RTK positions: {len(rtk_positions)}")
if rtk_positions:
    first = rtk_positions[0]
    last = rtk_positions[-1]
    print(f"  Start: lat={first[1]:.6f} lon={first[2]:.6f}")
    print(f"  End:   lat={last[1]:.6f} lon={last[2]:.6f}")

    # Check if RTK area overlaps with patch database
    patch_lats = [gps_meta[n]['lat'] for n in names if n in gps_meta]
    patch_lons = [gps_meta[n]['lon'] for n in names if n in gps_meta]
    print(f"\n  Patch database coverage:")
    print(f"    Lat: {min(patch_lats):.6f} to {max(patch_lats):.6f}")
    print(f"    Lon: {min(patch_lons):.6f} to {max(patch_lons):.6f}")
    print(f"  RTK range:")
    rtk_lats = [p[1] for p in rtk_positions]
    rtk_lons = [p[2] for p in rtk_positions]
    print(f"    Lat: {min(rtk_lats):.6f} to {max(rtk_lats):.6f}")
    print(f"    Lon: {min(rtk_lons):.6f} to {max(rtk_lons):.6f}")

    # Overlap?
    lat_overlap = max(min(patch_lats), min(rtk_lats)) < min(max(patch_lats), max(rtk_lats))
    lon_overlap = max(min(patch_lons), min(rtk_lons)) < min(max(patch_lons), max(rtk_lons))
    print(f"\n  GPS OVERLAP: lat={lat_overlap} lon={lon_overlap}")

# Second pass: extract a few drone frames
with Reader(BAG_PATH) as reader:
    for conn, timestamp, rawdata in reader.messages():
        if conn.topic == '/camera/image_mono':
            msg = deserialize_cdr(rawdata, conn.msgtype)
            gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)

            # Find nearest RTK position
            rtk_lat, rtk_lon = 0, 0
            for ts, lat, lon in rtk_positions:
                if ts >= timestamp:
                    rtk_lat, rtk_lon = lat, lon
                    break

            frame_count += 1
            if frame_count in [1, 50, 100, 200, 500, 1000, 2000]:
                sims = preprocess_and_match(gray,
                    f"Frame {frame_count} (RTK: lat={rtk_lat:.6f} lon={rtk_lon:.6f}, "
                    f"img={msg.width}x{msg.height})")

            if frame_count > 2000:
                break

print(f"\nTotal frames processed: {frame_count}")
