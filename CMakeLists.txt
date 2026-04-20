cmake_minimum_required(VERSION 3.22)

# Inject FindOpenCV.cmake before any find_package(OpenCV) — including cv_bridge's transitive call.
# The system /usr/lib/cmake/opencv4 references libopencv_core.so.4.8.0 which does not exist.
get_filename_component(_WS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
list(INSERT CMAKE_MODULE_PATH 0 "${_WS_ROOT}/cmake/modules")

project(particle_filter_loc_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH "/usr/local/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# ─── ROS 2 dependencies ─────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# ─── System dependencies ────────────────────────────────────────
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT (JetPack provides headers + libs)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/include/aarch64-linux-gnu)
find_library(NVINFER_LIB nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)

# PoseLib
find_package(PoseLib QUIET)

# CUDA
find_package(CUDAToolkit REQUIRED)

# nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  # Fallback: header-only from system
  find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp PATHS /usr/include)
endif()

# ─── Core library (non-ROS) ─────────────────────────────────────
add_library(pf_core SHARED
  src/config.cpp
  src/particle_filter.cpp
  src/trust_model.cpp
  src/descriptor_database.cpp
  src/footprint_reconstruction.cpp
  src/trt_engine.cpp
  src/vpi_preprocessor.cpp
  src/coarse_matcher.cpp
  src/fine_matcher.cpp
  src/pose_estimator.cpp
  src/observation_model.cpp
)

target_include_directories(pf_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(pf_core PUBLIC
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  CUDA::cudart
  ${OpenCV_LIBS}
  yaml-cpp
  Eigen3::Eigen
)

if(PoseLib_FOUND)
  target_link_libraries(pf_core PUBLIC PoseLib::PoseLib)
  target_compile_definitions(pf_core PUBLIC HAS_POSELIB=1)
endif()

if(nlohmann_json_FOUND)
  target_link_libraries(pf_core PUBLIC nlohmann_json::nlohmann_json)
endif()

# ─── ROS 2 node executable ──────────────────────────────────────
add_executable(pf_geo_loc_node
  src/pf_node.cpp
  src/main.cpp
)

find_package(ament_index_cpp REQUIRED)
ament_target_dependencies(pf_geo_loc_node PUBLIC
  rclcpp sensor_msgs std_msgs geometry_msgs cv_bridge ament_index_cpp
)
target_link_libraries(pf_geo_loc_node PUBLIC pf_core)

# ─── Install ─────────────────────────────────────────────────────
install(TARGETS pf_core DESTINATION lib)
install(TARGETS pf_geo_loc_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
