#pragma once

#include <Eigen/Dense>

#include "trt_engine.hpp"
#include "vpi_preprocessor.hpp"

namespace pf {

struct MatchOutput {
    Eigen::MatrixXf keypoints0;  // [M, 2]
    Eigen::MatrixXf keypoints1;  // [M, 2]
    Eigen::VectorXf confidence;  // [M]
    int num_matches = 0;
};

class FineMatcher {
public:
    FineMatcher(const std::string& engine_path,
                VpiPreprocessor& vpi,
                int width = 512,
                int height = 288,
                float conf_threshold = 0.2f);

    // Match drone mono image against a BGR patch image
    MatchOutput match(const uint8_t* drone_mono, int dw, int dh,
                      const uint8_t* patch_bgr, int pw, int ph);

private:
    TrtEngine engine_;
    VpiPreprocessor& vpi_;
    int width_;
    int height_;
    float conf_threshold_;
    int img0_idx_, img1_idx_;
    int kpts0_idx_, kpts1_idx_, conf_idx_, valid_count_idx_;
};

}  // namespace pf
