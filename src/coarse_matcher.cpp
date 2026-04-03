#include "particle_filter_loc_cpp/coarse_matcher.hpp"

#include <cmath>
#include <iostream>

namespace pf {

CoarseMatcher::CoarseMatcher(
    const std::string& engine_path,
    DescriptorDatabase& db,
    VpiPreprocessor& vpi,
    int image_size)
    : engine_(engine_path), db_(db), vpi_(vpi), image_size_(image_size)
{
    // Find input/output tensor indices
    input_idx_ = 0;
    output_idx_ = -1;
    for (int i = 0; i < engine_.num_io(); ++i) {
        if (engine_.is_input(i))
            input_idx_ = i;
        else
            output_idx_ = i;
    }
    if (output_idx_ < 0)
        throw std::runtime_error("VLAD TRT engine has no output tensor");

    std::cout << "[CoarseMatcher] Engine loaded, desc_dim="
              << engine_.tensor_bytes(output_idx_) / sizeof(float) << std::endl;
}

CoarseResult CoarseMatcher::match(
    const uint8_t* mono_data, int width, int height,
    const std::vector<int>* candidate_indices, int top_k)
{
    // Preprocess: mono -> resize -> 3ch + normalize -> write to TRT input buffer
    float* input_buf = engine_.buffer_as<float>(input_idx_);
    vpi_.prepare_coarse(mono_data, width, height, input_buf);

    // Check input buffer values after preprocessing
    float in_min = 1e9f, in_max = -1e9f, in_sum = 0.0f;
    int in_count = 256 * 256 * 3;
    for (int i = 0; i < in_count; ++i) {
        in_sum += input_buf[i];
        if (input_buf[i] < in_min) in_min = input_buf[i];
        if (input_buf[i] > in_max) in_max = input_buf[i];
    }
    std::cout << "[CoarseMatcher] DBG input: min=" << in_min
              << " max=" << in_max
              << " mean=" << (in_sum / in_count)
              << " (expected ~0 mean for ImageNet norm)" << std::endl;

    // Save first preprocessed input to file for Python comparison
    static bool saved_input = false;
    if (!saved_input) {
        saved_input = true;
        FILE* fp = fopen("/tmp/cpp_coarse_input.bin", "wb");
        if (fp) {
            fwrite(input_buf, sizeof(float), in_count, fp);
            fclose(fp);
            std::cout << "[CoarseMatcher] Saved preprocessed input to /tmp/cpp_coarse_input.bin" << std::endl;
        }
        // Also save raw mono pixels
        FILE* fp2 = fopen("/tmp/cpp_mono_input.bin", "wb");
        if (fp2) {
            fwrite(mono_data, 1, width * height, fp2);
            fclose(fp2);
            std::cout << "[CoarseMatcher] Saved raw mono to /tmp/cpp_mono_input.bin (" << width << "x" << height << ")" << std::endl;
        }
    }

    // Run TRT inference (GPU reads from unified memory, writes output to unified memory)
    engine_.infer();
    engine_.sync();

    // Read descriptor from output buffer (already accessible on CPU via unified memory)
    float* output_buf = engine_.buffer_as<float>(output_idx_);
    int desc_dim = static_cast<int>(engine_.tensor_bytes(output_idx_) / sizeof(float));

    // Check raw output before normalization
    float raw_norm = 0.0f;
    float out_min = 1e9f, out_max = -1e9f;
    int nonzero = 0;
    for (int i = 0; i < desc_dim; ++i) {
        raw_norm += output_buf[i] * output_buf[i];
        if (output_buf[i] < out_min) out_min = output_buf[i];
        if (output_buf[i] > out_max) out_max = output_buf[i];
        if (std::abs(output_buf[i]) > 1e-8f) nonzero++;
    }
    raw_norm = std::sqrt(raw_norm);
    std::cout << "[CoarseMatcher] DBG output: dim=" << desc_dim
              << " raw_norm=" << raw_norm
              << " min=" << out_min << " max=" << out_max
              << " nonzero=" << nonzero << "/" << desc_dim
              << " first5=[" << output_buf[0] << "," << output_buf[1] << ","
              << output_buf[2] << "," << output_buf[3] << "," << output_buf[4] << "]"
              << std::endl;

    // Save first output for comparison
    static bool saved_output = false;
    if (!saved_output) {
        saved_output = true;
        FILE* fp = fopen("/tmp/cpp_trt_output.bin", "wb");
        if (fp) {
            fwrite(output_buf, sizeof(float), desc_dim, fp);
            fclose(fp);
            std::cout << "[CoarseMatcher] Saved TRT output to /tmp/cpp_trt_output.bin" << std::endl;
        }
    }

    // L2 normalize the query descriptor
    float norm = 0.0f;
    for (int i = 0; i < desc_dim; ++i)
        norm += output_buf[i] * output_buf[i];
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (int i = 0; i < desc_dim; ++i)
            output_buf[i] /= norm;
    }

    // Cosine similarity search against database
    auto result = db_.cosine_top_k(output_buf, desc_dim, candidate_indices, top_k);

    std::cout << "[CoarseMatcher] DBG top sims:";
    for (size_t i = 0; i < result.top_k_sims.size(); ++i)
        std::cout << " " << result.top_k_sims[i];
    std::cout << std::endl;

    return result;
}

}  // namespace pf
