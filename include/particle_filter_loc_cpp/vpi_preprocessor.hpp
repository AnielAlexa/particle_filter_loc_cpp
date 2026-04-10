#pragma once

#include <cstddef>
#include <cstdint>

namespace pf {

class VpiPreprocessor {
public:
    VpiPreprocessor(int coarse_size = 256, int fine_size = 320);
    ~VpiPreprocessor() = default;

    VpiPreprocessor(const VpiPreprocessor&) = delete;
    VpiPreprocessor& operator=(const VpiPreprocessor&) = delete;

    // Prepare coarse input: mono8 -> resize to coarse_size -> gray->3ch + ImageNet normalize -> [1,3,cs,cs] float
    void prepare_coarse(const uint8_t* mono_data, int width, int height, float* output_ptr);

    // Prepare fine drone input: mono8 -> resize to fine_size -> normalize to [0,1] -> [1,1,fs,fs] float
    void prepare_fine(const uint8_t* mono_data, int width, int height, float* output_ptr);

    // Prepare fine patch input: BGR -> gray -> resize to fine_size -> [1,1,fs,fs] float
    void prepare_fine_patch(const uint8_t* bgr_data, int width, int height, float* output_ptr);

private:
    int coarse_size_;
    int fine_size_;

    // CPU bilinear resize
    void resize_mono8_cpu(const uint8_t* src, int sw, int sh,
                          uint8_t* dst, int dw, int dh);
};

}  // namespace pf
