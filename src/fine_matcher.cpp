#include "particle_filter_loc_cpp/fine_matcher.hpp"

#include <iostream>
#include <stdexcept>

namespace pf {

FineMatcher::FineMatcher(
    const std::string& engine_path,
    VpiPreprocessor& vpi,
    int width,
    int height,
    float conf_threshold)
    : engine_(engine_path), vpi_(vpi),
      width_(width), height_(height), conf_threshold_(conf_threshold)
{
    // Discover tensor indices by name
    // LiteSAM Full: image0, image1 (inputs), mkpts0_f, mkpts1_f, mconf, valid_count (outputs)
    // Also supports ELoFTR names as fallback: keypoints0, keypoints1, num_matches
    img0_idx_        = engine_.find_tensor("image0");
    img1_idx_        = engine_.find_tensor("image1");
    kpts0_idx_       = engine_.find_tensor("mkpts0_f");
    kpts1_idx_       = engine_.find_tensor("mkpts1_f");
    conf_idx_        = engine_.find_tensor("mconf");
    valid_count_idx_ = engine_.find_tensor("valid_count");

    // Fallback: ELoFTR-style names
    if (kpts0_idx_ < 0) kpts0_idx_ = engine_.find_tensor("keypoints0");
    if (kpts1_idx_ < 0) kpts1_idx_ = engine_.find_tensor("keypoints1");
    if (valid_count_idx_ < 0) valid_count_idx_ = engine_.find_tensor("num_matches");

    // Last resort: assign by position
    if (img0_idx_ < 0 || img1_idx_ < 0) {
        int input_count = 0;
        for (int i = 0; i < engine_.num_io(); ++i) {
            if (engine_.is_input(i)) {
                if (input_count == 0) img0_idx_ = i;
                else img1_idx_ = i;
                input_count++;
            }
        }
    }
    if (kpts0_idx_ < 0 || kpts1_idx_ < 0 || conf_idx_ < 0) {
        int output_count = 0;
        for (int i = 0; i < engine_.num_io(); ++i) {
            if (!engine_.is_input(i)) {
                if (output_count == 0) kpts0_idx_ = i;
                else if (output_count == 1) kpts1_idx_ = i;
                else if (output_count == 2) conf_idx_ = i;
                output_count++;
            }
        }
    }

    std::cout << "[FineMatcher] Engine loaded, input=" << width_ << "x" << height_
              << " conf_threshold=" << conf_threshold_ << std::endl;
}

MatchOutput FineMatcher::match(
    const uint8_t* drone_mono, int dw, int dh,
    const uint8_t* patch_bgr, int pw, int ph)
{
    MatchOutput result;

    // Prepare inputs: write preprocessed images directly to TRT unified memory buffers
    vpi_.prepare_fine(drone_mono, dw, dh, engine_.buffer_as<float>(img0_idx_));
    vpi_.prepare_fine_patch(patch_bgr, pw, ph, engine_.buffer_as<float>(img1_idx_));

    // Run ELoFTR TRT inference
    engine_.infer();
    engine_.sync();

    // Read outputs from unified memory
    float* kpts0_ptr = engine_.buffer_as<float>(kpts0_idx_);
    float* kpts1_ptr = engine_.buffer_as<float>(kpts1_idx_);
    float* conf_ptr = engine_.buffer_as<float>(conf_idx_);

    // Get actual match count from valid_count output tensor (int64 for LiteSAM, int32 for ELoFTR)
    int total_valid = 0;
    if (valid_count_idx_ >= 0) {
        int64_t* vc_ptr = engine_.buffer_as<int64_t>(valid_count_idx_);
        total_valid = static_cast<int>(vc_ptr[0]);
    } else {
        // Fallback: count non-zero confidence entries
        int max_buf = static_cast<int>(engine_.tensor_bytes(conf_idx_) / sizeof(float));
        for (int i = 0; i < max_buf; ++i) {
            if (conf_ptr[i] > 0.0f) total_valid++;
            else break;
        }
    }
    // Clamp to buffer capacity
    int max_kpts = static_cast<int>(engine_.tensor_bytes(conf_idx_) / sizeof(float));
    total_valid = std::min(total_valid, max_kpts);

    if (total_valid == 0) return result;

    // Filter by confidence threshold
    std::vector<int> good_idx;
    good_idx.reserve(total_valid);
    for (int i = 0; i < total_valid; ++i) {
        if (conf_ptr[i] > conf_threshold_)
            good_idx.push_back(i);
    }

    result.num_matches = static_cast<int>(good_idx.size());
    if (result.num_matches == 0) return result;

    result.keypoints0.resize(result.num_matches, 2);
    result.keypoints1.resize(result.num_matches, 2);
    result.confidence.resize(result.num_matches);

    for (int j = 0; j < result.num_matches; ++j) {
        int i = good_idx[j];
        result.keypoints0(j, 0) = kpts0_ptr[2 * i + 0];
        result.keypoints0(j, 1) = kpts0_ptr[2 * i + 1];
        result.keypoints1(j, 0) = kpts1_ptr[2 * i + 0];
        result.keypoints1(j, 1) = kpts1_ptr[2 * i + 1];
        result.confidence(j) = conf_ptr[i];
    }

    return result;
}

}  // namespace pf
