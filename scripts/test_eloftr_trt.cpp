/**
 * Standalone C++ test for ELoFTR TensorRT inference.
 * Compares output with Python TRT wrapper to verify correctness.
 *
 * Build:
 *   g++ -std=c++17 -O2 test_eloftr_trt.cpp \
 *       -I/usr/include/aarch64-linux-gnu \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lnvinfer -lcudart \
 *       -o test_eloftr_trt
 *
 * Usage:
 *   ./test_eloftr_trt <engine_path> <image0> <image1>
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Max matches to pre-allocate for dynamic-shape outputs
static constexpr int MAX_MATCHES = 2000;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

static size_t dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default: return 4;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <engine.trt> <img0> <img1>" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::string img0_path = argv[2];
    std::string img1_path = argv[3];
    int resolution = 320;

    // ── Load engine ────────────────────────────────────────────
    Logger logger;
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open engine\n"; return 1; }
    file.seekg(0, std::ios::end);
    size_t sz = file.tellg();
    file.seekg(0);
    std::vector<char> engine_data(sz);
    file.read(engine_data.data(), sz);

    auto* runtime = nvinfer1::createInferRuntime(logger);
    auto* engine = runtime->deserializeCudaEngine(engine_data.data(), sz);
    if (!engine) { std::cerr << "Deserialization failed\n"; return 1; }
    auto* context = engine->createExecutionContext();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── Discover I/O tensors ───────────────────────────────────
    int n_io = engine->getNbIOTensors();
    std::cout << "Engine has " << n_io << " I/O tensors:\n";

    struct TensorInfo {
        std::string name;
        bool is_input;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype;
        size_t alloc_bytes;
        void* ptr = nullptr;
    };
    std::vector<TensorInfo> tensors(n_io);

    for (int i = 0; i < n_io; i++) {
        auto& t = tensors[i];
        t.name = engine->getIOTensorName(i);
        t.is_input = (engine->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT);
        t.dims = engine->getTensorShape(t.name.c_str());
        t.dtype = engine->getTensorDataType(t.name.c_str());

        // Compute allocation size — use MAX_MATCHES for dynamic dims
        size_t vol = 1;
        bool has_dynamic = false;
        std::cout << "  [" << i << "] " << t.name << " "
                  << (t.is_input ? "INPUT" : "OUTPUT") << " shape=(";
        for (int d = 0; d < t.dims.nbDims; d++) {
            if (d > 0) std::cout << ",";
            std::cout << t.dims.d[d];
            if (t.dims.d[d] < 0) {
                vol *= MAX_MATCHES;
                has_dynamic = true;
            } else {
                vol *= t.dims.d[d];
            }
        }
        std::cout << ") dtype=" << static_cast<int>(t.dtype);

        t.alloc_bytes = vol * dtype_size(t.dtype);
        std::cout << " alloc=" << t.alloc_bytes << " bytes";
        if (has_dynamic) std::cout << " [DYNAMIC, max=" << MAX_MATCHES << "]";
        std::cout << "\n";

        cudaMallocManaged(&t.ptr, t.alloc_bytes);
        memset(t.ptr, 0, t.alloc_bytes);
    }

    // ── Preprocess images ──────────────────────────────────────
    auto preprocess_mono = [&](const std::string& path) -> std::vector<float> {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Cannot read: " << path << "\n";
            exit(1);
        }
        cv::Mat resized;
        cv::resize(img, resized, {resolution, resolution});
        std::vector<float> buf(resolution * resolution);
        for (int i = 0; i < resolution * resolution; i++)
            buf[i] = resized.data[i] / 255.0f;
        return buf;
    };

    auto img0_data = preprocess_mono(img0_path);
    auto img1_data = preprocess_mono(img1_path);

    // Copy to input tensors
    for (auto& t : tensors) {
        if (!t.is_input) continue;
        float* dst = static_cast<float*>(t.ptr);
        if (t.name.find("0") != std::string::npos) {
            memcpy(dst, img0_data.data(), img0_data.size() * sizeof(float));
        } else {
            memcpy(dst, img1_data.data(), img1_data.size() * sizeof(float));
        }
    }

    // ── Set tensor addresses ───────────────────────────────────
    for (auto& t : tensors)
        context->setTensorAddress(t.name.c_str(), t.ptr);

    // ── Run inference ──────────────────────────────────────────
    std::cout << "\nRunning inference..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << ms << " ms\n\n";

    // ── Read outputs ───────────────────────────────────────────
    for (auto& t : tensors) {
        if (t.is_input) continue;

        // Get actual output shape after inference
        auto actual_dims = context->getTensorShape(t.name.c_str());
        size_t actual_vol = 1;
        std::cout << t.name << " actual_shape=(";
        for (int d = 0; d < actual_dims.nbDims; d++) {
            if (d > 0) std::cout << ",";
            std::cout << actual_dims.d[d];
            actual_vol *= std::max(1, static_cast<int>(actual_dims.d[d]));
        }
        std::cout << ") vol=" << actual_vol << "\n";

        float* data = static_cast<float*>(t.ptr);
        if (t.name.find("conf") != std::string::npos || t.name.find("mconf") != std::string::npos) {
            int count = 0;
            for (size_t i = 0; i < actual_vol; i++) {
                if (data[i] > 0.2f) count++;
            }
            std::cout << "  Matches with conf > 0.2: " << count << "/" << actual_vol << "\n";
            // Print first 10 confidence values
            int show = std::min(static_cast<size_t>(10), actual_vol);
            std::cout << "  First " << show << " values: ";
            for (int i = 0; i < show; i++) std::cout << data[i] << " ";
            std::cout << "\n";
        } else if (t.name.find("num_matches") != std::string::npos) {
            // Could be int32
            if (t.dtype == nvinfer1::DataType::kINT32) {
                int32_t* idata = static_cast<int32_t*>(t.ptr);
                std::cout << "  num_matches = " << idata[0] << "\n";
            } else {
                std::cout << "  num_matches = " << data[0] << "\n";
            }
        } else if (t.name.find("keypoints") != std::string::npos || t.name.find("kpts") != std::string::npos) {
            int show = std::min(static_cast<size_t>(5), actual_vol / 2);
            std::cout << "  First " << show << " keypoints: ";
            for (int i = 0; i < show; i++)
                std::cout << "(" << data[2*i] << "," << data[2*i+1] << ") ";
            std::cout << "\n";
        }
    }

    // ── Cleanup ────────────────────────────────────────────────
    for (auto& t : tensors) cudaFree(t.ptr);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
