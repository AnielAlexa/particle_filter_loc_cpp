# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A GPU-accelerated particle filter for drone visual geo-localization on Jetson Orin Nano. The system matches downward-facing camera images against a georeferenced patch database using a two-stage pipeline (coarse VLAD descriptors → fine ELOFTr feature matching), fused with RTK GPS and altimeter data via a particle filter.

## Build & Run

```bash
# Build
cd ~/ros2_ws && colcon build --packages-select particle_filter_loc_cpp --event-handlers console_direct+

# Source
source ~/ros2_ws/install/setup.bash

# Run (with config)
ros2 launch particle_filter_loc_cpp pf_geo_loc_cpp.launch.py config_path:=/path/to/pf_config.yaml

# Run node directly
ros2 run particle_filter_loc_cpp pf_geo_loc_node --ros-args -p config_path:=config/pf_config.yaml
```

No tests are configured in CMakeLists.txt. Validation scripts live in `scripts/` (Python).

## Platform & Dependencies

- **Target**: Jetson Orin Nano, ARM Cortex-A78AE, JetPack
- **Language**: C++17, CUDA
- **Key deps**: TensorRT (inference), VPI (GPU preprocessing, optional via `HAS_VPI`), PoseLib (pose estimation, optional via `HAS_POSELIB`), OpenCV 4.12 (custom-built at `/usr/local/lib`, NOT system 4.5d), Eigen3, yaml-cpp, nlohmann/json
- **OpenCV pitfall**: System has OpenCV 4.5d at `/usr/lib` AND 4.12 at `/usr/local/lib`. CMakeLists forces 4.12 via `OpenCV_DIR` and `CMAKE_INSTALL_RPATH`. The launch file uses `LD_PRELOAD` to force 4.12 at runtime. Do not remove these workarounds.

## Architecture

### Pipeline (per camera frame)

```
Camera Image (mono8) + RTK GPS + Yaw + Altimeter
    → VPI Preprocessing (resize/normalize for TRT)
    → Coarse Match: VLAD descriptor vs patch database → top-K candidates
    → Particle Filter Predict (RTK motion model)
    → Decision: run fine matching? (based on particle spread)
        → Fine Match Cascade (3 stages, early-exit on ≥15 inliers):
            Stage A: Satellite perspective-warped crop
            Stage B: Mosaic heading-aligned stitch
            Stage C: Single patch lookup
        → Trust Model scores result → modulates observation noise
    → Particle Filter Update + Resample
    → Phase Transitions
    → Publish (lat/lon, ESS, state)
```

### Core Library (`pf_core`) — non-ROS

| Component | File | Responsibility |
|-----------|------|----------------|
| `ParticleFilter` | `particle_filter.*` | Bayesian state estimation: predict, update (coarse/fine), resample, phase transitions |
| `ObservationModel` | `observation_model.*` | Orchestrates VPI → coarse → fine pipeline; owns all matcher/estimator instances |
| `TrustTracker` | `trust_model.*` | Weighted geometric mean of 5 signals (inliers, similarity, consistency, altitude, geometry) → confidence ∈ [0,1] |
| `CoarseMatcher` | `coarse_matcher.*` | VLAD-TRT descriptor extraction + cosine similarity top-K search |
| `FineMatcher` | `fine_matcher.*` | ELOFTr-TRT feature matching between drone image and reference patches |
| `PoseEstimator` | `pose_estimator.*` | PnP / homography RANSAC → GPS coordinates from keypoint correspondences |
| `DescriptorDatabase` | `descriptor_database.*` | In-memory patch descriptors (.npy), GPS metadata, spatial radius queries in ENU |
| `FootprintReconstructor` | `footprint_reconstruction.*` | Stitches satellite tiles into perspective/heading-aligned crops for fine matching |
| `VpiPreprocessor` | `vpi_preprocessor.*` | GPU image resize + normalization for TRT input |
| `TrtEngine` | `trt_engine.*` | Generic TensorRT engine wrapper (load .engine/.trt, allocate buffers, infer) |
| `Config` | `config.*` | YAML-based config loading with fallback defaults |
| `RTKMotionModel` | `motion_model.hpp` | Header-only; computes ENU motion delta from consecutive RTK fixes |

### ROS2 Node (`pf_geo_loc_node`)

| File | Role |
|------|------|
| `pf_node.*` | `PFGeoLocNode` — ROS2 lifecycle: subscriptions, callbacks, frame processing loop |
| `main.cpp` | Entry point, `rclcpp::spin()` |

### Shared Types

`types.hpp` defines all value types: `Particle`, `MotionDelta`, `CoarseResult`, `FineResult`, `PatchMeta`, `FlowMetrics`, etc.

## ROS2 Topics

**Subscriptions:**
- `/camera/image_rect` (sensor_msgs/Image, mono8) — drone camera
- `/m300/rtk/fix` (sensor_msgs/NavSatFix) — RTK position
- `/m300/rtk/yaw` (std_msgs/Float64) — compass yaw (×10 encoding)
- `/altimeter/range` (sensor_msgs/Range) — altitude AGL

**Publications:**
- `/pf_geo_loc/position` (sensor_msgs/NavSatFix) — filtered estimate
- `/pf_geo_loc/ess` (std_msgs/Float32) — effective sample size
- `/pf_geo_loc/state` (std_msgs/String) — phase: UNINIT/DISPERSED/CONVERGING/TRACKING

Topic names are configurable via the `replay:` section in `pf_config.yaml`.

## Key Conventions

- **Coordinate system**: Internal calculations in ENU (East-North-Up) relative to `origin_lat`/`origin_lon`. Conversions via flat-Earth approximation in `geo_utils.hpp` (METERS_PER_DEG = 111,319.5m).
- **Heading**: 0° = North, 90° = East. `camera.heading_offset_deg` (default 270°) compensates for camera mount orientation.
- **Exposure units**: Yaw arrives ×10 encoded from the RTK topic (divided on receipt).
- **Phase machine**: UNINIT → DISPERSED (300 particles) → CONVERGING → TRACKING (150 particles). Transitions triggered by particle spatial spread thresholds.
- **Adaptive regime**: Correction distance EMA detects RTK-accurate vs VIO-drifting regimes. Interpolates process noise (0.5–3.0m) and observation sigma accordingly.
- **Threading**: Single-threaded `rclcpp::spin()`. Frame processing guarded by atomic `processing_` flag (drops frames if busy). TRT/VPI execute synchronously on GPU.
- **Logging**: `RCLCPP_INFO` for key events, `RCLCPP_INFO_THROTTLE` for periodic debug, `RCLCPP_WARN` for recoverable errors.

## Configuration

All parameters in `config/pf_config.yaml`, loaded by `Config::load()`. Nested sections:

- **`camera`**: Intrinsics (fx/fy/cx/cy), resolution, heading offset
- **`pf`**: Particle counts, process/observation noise sigmas, phase spread thresholds, adaptive regime bounds, roughening scale, static detection threshold
- **`trust`**: Signal weights (inlier/sim/consistency/altitude/geometry), sigma modulation range, drift detection
- **`matcher`**: TRT engine paths, matcher resolution (320×320), patch cache LRU limit, mosaic context multiplier, camera intrinsics at matcher resolution
- **`init`**: Lock-in requirements (consecutive hits, similarity threshold)
- **`replay`**: Topic name overrides

## Model & Data Files

- `model/dinov3_vlad_pca768_l10_value_256.engine` — VLAD coarse descriptor TRT engine
- `model/MatchAnything/matchanything_eloftr_320.trt` — ELOFTr fine matcher TRT engine
- `model/data/patches/` — Georeferenced PNG tiles (~400×400 px)
- `model/data/patches/gps_metadata.json` — Per-tile lat/lon bounds
- `model/data/descriptors_vlad_l10_aerial/` — VLAD codebook, PCA components, patch descriptors (.npy)

## Scripts

| Script | Purpose |
|--------|---------|
| `image_crop_node.py` | ROS2 node: center-crop camera images to square for matcher input |
| `plot_positions.py` | Real-time plotter: RTK + PF estimate on satellite background |
| `validate_trt_v2.py` | Test VLAD TRT inference pipeline |
| `validate_bag_match.py` | Offline matching analysis on ROS bags |
| `convert_pt_to_npy.py` | Convert PyTorch descriptors → .npy for C++ loading |
