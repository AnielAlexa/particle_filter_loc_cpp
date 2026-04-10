#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace pf {

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Number of I/O tensors
    int num_io() const { return static_cast<int>(buffers_.size()); }

    // Get tensor info
    const std::string& tensor_name(int idx) const { return buffers_[idx].name; }
    bool is_input(int idx) const { return buffers_[idx].is_input; }
    size_t tensor_bytes(int idx) const { return buffers_[idx].byte_size; }

    // Unified memory buffer — accessible from both CPU and GPU (Jetson shared DRAM)
    void* buffer_ptr(int idx) { return buffers_[idx].managed_ptr; }
    const void* buffer_ptr(int idx) const { return buffers_[idx].managed_ptr; }

    // Typed access
    template <typename T>
    T* buffer_as(int idx) { return static_cast<T*>(buffers_[idx].managed_ptr); }

    // Find buffer by name
    int find_tensor(const std::string& name) const;
    void* buffer_ptr_by_name(const std::string& name);

    // Execute inference
    void infer(cudaStream_t stream = nullptr);

    // Synchronize stream
    void sync(cudaStream_t stream = nullptr);

private:
    struct IOBuffer {
        std::string name;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype;
        size_t byte_size;
        void* managed_ptr = nullptr;  // Unified memory (cudaMallocManaged) — Jetson shares DRAM
        bool is_input;
    };

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::vector<IOBuffer> buffers_;
    cudaStream_t owned_stream_ = nullptr;
};

}  // namespace pf
