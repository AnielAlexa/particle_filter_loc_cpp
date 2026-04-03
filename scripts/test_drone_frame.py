#!/usr/bin/env python3
"""Capture one frame from /camera/image_rect, run through TRT, check sims."""
import numpy as np
import cv2
import os
import torch
import tensorrt as trt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

MODEL_DIR = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/model"

# Load database
db = np.load(os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/vlad_descriptors_pca768.npy")).astype(np.float32)
db = db / np.linalg.norm(db, axis=1, keepdims=True).clip(1e-8)
with open(os.path.join(MODEL_DIR, "data/descriptors_vlad_l10_aerial/patch_names.txt")) as f:
    names = [l.strip() for l in f if l.strip()]

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

class FrameGrabber(Node):
    def __init__(self):
        super().__init__("frame_grabber")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(Image, "/camera/image_rect", self.cb, qos)
        self.done = False

    def cb(self, msg):
        if self.done:
            return
        self.done = True
        h, w = msg.height, msg.width
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        print(f"Got frame {w}x{h}, pixel avg={gray.mean():.1f}")

        # Save for inspection
        cv2.imwrite("/tmp/drone_frame.png", gray)

        # Preprocess exactly like C++: resize 320->256, gray->3ch, ImageNet normalize
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        inp = np.zeros((1, 3, 256, 256), dtype=np.float32)
        for c in range(3):
            inp[0, c] = (resized.astype(np.float32) / 255.0 - mean[c]) / std_[c]

        print(f"Input: min={inp.min():.4f} max={inp.max():.4f} mean={inp.mean():.4f}")

        input_t.copy_(torch.from_numpy(inp).cuda())
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        desc = output_t.cpu().numpy().flatten()
        norm = np.linalg.norm(desc)
        print(f"Output: norm={norm:.4f}, first5={desc[:5]}")
        desc = desc / max(norm, 1e-8)

        sims = db @ desc
        top_idx = np.argsort(-sims)[:10]
        print(f"\nTop-10 matches (Python TRT):")
        for i, idx in enumerate(top_idx):
            print(f"  [{i}] {names[idx]}: sim={sims[idx]:.4f}")

        print(f"\nAll sims: mean={sims.mean():.4f} std={sims.std():.4f} max={sims.max():.4f}")

def main():
    rclpy.init()
    node = FrameGrabber()
    import time
    t0 = time.time()
    while not node.done and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.done:
        print("No frame received within 10s")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
