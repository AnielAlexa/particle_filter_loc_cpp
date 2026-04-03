#include "particle_filter_loc_cpp/trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace pf {

void TrtEngine::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

static size_t dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

// Max elements for dynamic-shape dimensions (e.g. ELoFTR match count)
static constexpr int64_t kDynamicDimMax = 2000;

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        int64_t dim = d.d[i];
        if (dim < 0)
            dim = kDynamicDimMax;  // dynamic dimension: allocate max buffer
        v *= std::max(dim, static_cast<int64_t>(1));
    }
    return v;
}

TrtEngine::TrtEngine(const std::string& engine_path) {
    // Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open TRT engine: " + engine_path);

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);

    // Deserialize
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_)
        throw std::runtime_error("Failed to create TRT runtime");

    engine_ = runtime_->deserializeCudaEngine(data.data(), size);
    if (!engine_)
        throw std::runtime_error("Failed to deserialize TRT engine: " + engine_path);

    context_ = engine_->createExecutionContext();
    if (!context_)
        throw std::runtime_error("Failed to create TRT execution context");

    // Create CUDA stream
    cudaStreamCreate(&owned_stream_);

    // Enumerate I/O tensors and allocate separate GPU + pinned host buffers
    int n_io = engine_->getNbIOTensors();
    buffers_.resize(n_io);

    for (int i = 0; i < n_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        buffers_[i].name = name;
        buffers_[i].is_input = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        buffers_[i].dims = engine_->getTensorShape(name);
        buffers_[i].dtype = engine_->getTensorDataType(name);
        buffers_[i].byte_size = volume(buffers_[i].dims) * dtype_size(buffers_[i].dtype);

        // GPU device memory for TRT inference
        cudaError_t err = cudaMalloc(&buffers_[i].dev_ptr, buffers_[i].byte_size);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));

        // Pinned host memory for CPU read/write
        err = cudaMallocHost(&buffers_[i].host_ptr, buffers_[i].byte_size);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMallocHost failed: ") + cudaGetErrorString(err));

        std::memset(buffers_[i].host_ptr, 0, buffers_[i].byte_size);

        std::cout << "[TRT] Tensor '" << name << "' "
                  << (buffers_[i].is_input ? "INPUT" : "OUTPUT")
                  << " " << buffers_[i].byte_size << " bytes" << std::endl;
    }
}

TrtEngine::~TrtEngine() {
    for (auto& buf : buffers_) {
        if (buf.dev_ptr) cudaFree(buf.dev_ptr);
        if (buf.host_ptr) cudaFreeHost(buf.host_ptr);
    }
    if (owned_stream_) cudaStreamDestroy(owned_stream_);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

int TrtEngine::find_tensor(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(buffers_.size()); ++i)
        if (buffers_[i].name == name) return i;
    return -1;
}

void* TrtEngine::buffer_ptr_by_name(const std::string& name) {
    int idx = find_tensor(name);
    if (idx < 0) return nullptr;
    return buffers_[idx].host_ptr;
}

void TrtEngine::infer(cudaStream_t stream) {
    cudaStream_t s = stream ? stream : owned_stream_;

    // Set tensor addresses to device pointers
    for (auto& buf : buffers_)
        context_->setTensorAddress(buf.name.c_str(), buf.dev_ptr);

    // Copy input buffers from host (pinned) to device — synchronous to ensure completion
    for (auto& buf : buffers_) {
        if (buf.is_input) {
            cudaMemcpy(buf.dev_ptr, buf.host_ptr, buf.byte_size,
                       cudaMemcpyHostToDevice);
        }
    }

    // Execute on stream
    if (!context_->enqueueV3(s))
        std::cerr << "[TRT] enqueueV3 failed!" << std::endl;

    // Wait for inference to complete
    cudaStreamSynchronize(s);

    // Copy output buffers from device to host — synchronous
    for (auto& buf : buffers_) {
        if (!buf.is_input) {
            cudaMemcpy(buf.host_ptr, buf.dev_ptr, buf.byte_size,
                       cudaMemcpyDeviceToHost);
        }
    }
}

void TrtEngine::sync(cudaStream_t stream) {
    cudaStream_t s = stream ? stream : owned_stream_;
    cudaStreamSynchronize(s);
}

}  // namespace pf
