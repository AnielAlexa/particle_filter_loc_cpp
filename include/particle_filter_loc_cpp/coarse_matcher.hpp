#pragma once

#include <memory>
#include <vector>

#include "descriptor_database.hpp"
#include "trt_engine.hpp"
#include "types.hpp"
#include "vpi_preprocessor.hpp"

namespace pf {

class CoarseMatcher {
public:
    CoarseMatcher(const std::string& engine_path,
                  DescriptorDatabase& db,
                  VpiPreprocessor& vpi,
                  int image_size = 256);

    CoarseResult match(const uint8_t* mono_data, int width, int height,
                       const std::vector<int>* candidate_indices = nullptr,
                       int top_k = 5);

private:
    TrtEngine engine_;
    DescriptorDatabase& db_;
    VpiPreprocessor& vpi_;
    int image_size_;
    int input_idx_;
    int output_idx_;
};

}  // namespace pf
