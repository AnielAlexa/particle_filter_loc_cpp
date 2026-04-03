#pragma once

#include <cstddef>
#include <cstdint>

#ifdef HAS_VPI
#include <vpi/Image.h>
#include <vpi/Stream.h>
#endif

namespace pf {

class VpiPreprocessor {
public:
    VpiPreprocessor(int coarse_size = 256, int fine_size = 320);
    ~VpiPreprocessor();

    VpiPreprocessor(const VpiPreprocessor&) = delete;
    VpiPreprocessor& operator=(const VpiPreprocessor&) = delete;

    // Prepare coarse input: mono8 320x320 -> resize to 256 -> gray->3ch + ImageNet normalize -> [1,3,256,256] float
    // Writes directly into output_ptr (cudaMallocManaged buffer)
    void prepare_coarse(const uint8_t* mono_data, int width, int height, float* output_ptr);

    // Prepare fine drone input: mono8 320x320 -> normalize to [0,1] -> [1,1,320,320] float
    void prepare_fine(const uint8_t* mono_data, int width, int height, float* output_ptr);

    // Prepare fine patch input: BGR -> gray -> resize to fine_size -> [1,1,fine_size,fine_size] float
    void prepare_fine_patch(const uint8_t* bgr_data, int width, int height, float* output_ptr);

private:
    int coarse_size_;
    int fine_size_;

    // CPU fallback resize (used when VPI unavailable)
    void resize_mono8_cpu(const uint8_t* src, int sw, int sh,
                          uint8_t* dst, int dw, int dh);

#ifdef HAS_VPI
    VPIStream stream_ = nullptr;
    VPIImage input_vpi_ = nullptr;
    VPIImage resized_vpi_ = nullptr;
    bool vpi_initialized_ = false;
    void* pinned_src_ = nullptr;
    size_t pinned_src_size_ = 0;
#endif
};

}  // namespace pf
