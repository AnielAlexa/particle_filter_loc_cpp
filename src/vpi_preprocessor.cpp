#include "particle_filter_loc_cpp/vpi_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace pf {

// ImageNet normalization constants
static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
static constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

VpiPreprocessor::VpiPreprocessor(int coarse_size, int fine_size)
    : coarse_size_(coarse_size), fine_size_(fine_size) {}

void VpiPreprocessor::resize_mono8_cpu(
    const uint8_t* src, int sw, int sh,
    uint8_t* dst, int dw, int dh)
{
    for (int y = 0; y < dh; ++y) {
        float sy = static_cast<float>(y) * (sh - 1) / (dh - 1);
        int y0 = static_cast<int>(sy);
        int y1 = std::min(y0 + 1, sh - 1);
        float fy = sy - y0;
        for (int x = 0; x < dw; ++x) {
            float sx = static_cast<float>(x) * (sw - 1) / (dw - 1);
            int x0 = static_cast<int>(sx);
            int x1 = std::min(x0 + 1, sw - 1);
            float fx = sx - x0;
            float val = (1 - fy) * ((1 - fx) * src[y0 * sw + x0] + fx * src[y0 * sw + x1]) +
                        fy * ((1 - fx) * src[y1 * sw + x0] + fx * src[y1 * sw + x1]);
            dst[y * dw + x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val + 0.5f)));
        }
    }
}

void VpiPreprocessor::prepare_coarse(
    const uint8_t* mono_data, int width, int height, float* output_ptr)
{
    int cs = coarse_size_;
    std::vector<uint8_t> resized(cs * cs);

    if (width == cs && height == cs)
        std::memcpy(resized.data(), mono_data, cs * cs);
    else
        resize_mono8_cpu(mono_data, width, height, resized.data(), cs, cs);

    // Gray -> 3 channel + ImageNet normalize -> CHW format [1, 3, cs, cs]
    int plane_size = cs * cs;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < plane_size; ++i) {
            float val = static_cast<float>(resized[i]) / 255.0f;
            output_ptr[c * plane_size + i] = (val - kMean[c]) / kStd[c];
        }
    }
}

void VpiPreprocessor::prepare_fine(
    const uint8_t* mono_data, int width, int height, float* output_ptr)
{
    int fs = fine_size_;
    int total = fs * fs;

    if (width == fs && height == fs) {
        for (int i = 0; i < total; ++i)
            output_ptr[i] = static_cast<float>(mono_data[i]) / 255.0f;
    } else {
        std::vector<uint8_t> resized(total);
        resize_mono8_cpu(mono_data, width, height, resized.data(), fs, fs);
        for (int i = 0; i < total; ++i)
            output_ptr[i] = static_cast<float>(resized[i]) / 255.0f;
    }
}

void VpiPreprocessor::prepare_fine_patch(
    const uint8_t* bgr_data, int width, int height, float* output_ptr)
{
    int fs = fine_size_;
    int total = fs * fs;

    // BGR -> grayscale
    std::vector<uint8_t> gray(width * height);
    for (int i = 0; i < width * height; ++i) {
        int b = bgr_data[3 * i + 0];
        int g = bgr_data[3 * i + 1];
        int r = bgr_data[3 * i + 2];
        gray[i] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
    }

    std::vector<uint8_t> resized(total);
    if (width == fs && height == fs)
        resized = gray;
    else
        resize_mono8_cpu(gray.data(), width, height, resized.data(), fs, fs);

    for (int i = 0; i < total; ++i)
        output_ptr[i] = static_cast<float>(resized[i]) / 255.0f;
}

}  // namespace pf
