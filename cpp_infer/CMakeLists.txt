cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Use a separate wall-x CUDA arch knob because CMAKE_CUDA_ARCHITECTURES may be
# left as OFF on some platforms even when nvcc is otherwise usable.
set(WALLX_CUDA_ARCH "87" CACHE STRING "CUDA arch used by wall-x nvcc flags")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${WALLX_CUDA_ARCH})
endif()

project(wallx_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Find dependencies ---
# libtorch (from PyTorch installation)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# CUDA
find_package(CUDAToolkit REQUIRED)

# cuDNN (JetPack installs at system level, not via CUDAToolkit)
find_library(CUDNN_LIBRARY cudnn
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS /usr/include /usr/local/cuda/include)
if(CUDNN_LIBRARY)
    message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")
    message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
else()
    message(WARNING "cuDNN not found, SDPA may not use cuDNN backend")
endif()

# Python (needed because csrc/*.cu includes torch/extension.h -> Python.h)
find_package(Python3 COMPONENTS Development QUIET)
if(Python3_FOUND)
    message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
else()
    # Fallback: try to find Python include manually
    execute_process(
        COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
    message(STATUS "Python3 include (manual): ${Python3_INCLUDE_DIRS}")
endif()

# --- Paths ---
set(WALLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CSRC_DIR "${WALLX_ROOT}/csrc")
set(KERNEL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
set(CUTLASS_INCLUDE "${WALLX_ROOT}/3rdparty/cutlass/include")
set(KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")

# --- CUDA architecture ---
if(DEFINED CMAKE_CUDA_ARCHITECTURES
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
    list(GET CMAKE_CUDA_ARCHITECTURES 0 WALLX_CUDA_ARCH_FROM_CMAKE)
    string(REGEX REPLACE "-real|-virtual" "" WALLX_CUDA_ARCH_FROM_CMAKE "${WALLX_CUDA_ARCH_FROM_CMAKE}")
    set(WALLX_CUDA_ARCH ${WALLX_CUDA_ARCH_FROM_CMAKE})
endif()

# --- Compile flags ---
set(CUDA_NVCC_FLAGS
    -std=c++17
    -DENABLE_BF16
    -DGROUPED_GEMM_DEVICE_CAPABILITY=${WALLX_CUDA_ARCH}
    --generate-code=arch=compute_${WALLX_CUDA_ARCH},code=sm_${WALLX_CUDA_ARCH}
    -O3
    --use_fast_math
)

# --- Custom CUDA operators from wall-x csrc ---
set(CSRC_SOURCES
    ${CSRC_DIR}/dual_asym_grouped_gemm.cu
    ${CSRC_DIR}/permute.cu
    ${CSRC_DIR}/rope.cu
    ${CSRC_DIR}/rope_index.cu
    ${CSRC_DIR}/rot_pos.cu
    ${CSRC_DIR}/window_index.cu
)

# Build csrc as a static library
add_library(wallx_cuda_ops STATIC ${CSRC_SOURCES})
target_include_directories(wallx_cuda_ops PUBLIC
    ${CSRC_DIR}
    ${CUTLASS_INCLUDE}
    ${TORCH_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)
target_compile_options(wallx_cuda_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)
target_link_libraries(wallx_cuda_ops PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    ${CUDNN_LIBRARY}
)
set_target_properties(wallx_cuda_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# --- Main inference engine ---
set(INFER_SOURCES
    src/main.cpp
    src/weight_loader.cpp
    src/triton_loader.cpp
    src/kv_cache.cpp
    src/attention.cpp
    src/moe.cpp
    src/transformer.cpp
    src/vision.cpp
    src/action_head.cpp
    src/ode_solver.cpp
    src/model.cpp
)

add_executable(wallx_infer ${INFER_SOURCES})
target_include_directories(wallx_infer PRIVATE
    src/
    ${KERNEL_SRC_DIR}
    ${CSRC_DIR}
    ${CUTLASS_INCLUDE}
    ${TORCH_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)
target_link_libraries(wallx_infer PRIVATE
    wallx_cuda_ops
    int8_fused_ops
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver  # For cubin loading (cuModuleLoad, etc.)
    CUDA::cublas
    CUDA::cublasLt
    ${CUDNN_LIBRARY}
)
target_compile_options(wallx_infer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)
target_compile_definitions(wallx_infer PRIVATE
    KERNELS_DIR="${KERNELS_DIR}"
)

# --- Fused INT8 CUDA kernels (for optimized W8A8 quantization) ---
add_library(int8_fused_ops STATIC
    src/kernels/activation_kernels.cu
    src/kernels/int8_kernels.cu
    src/kernels/cutlass_int8_gemm.cu
    src/kernels/norm_kernels.cu
)
target_include_directories(int8_fused_ops PUBLIC
    src/
    ${KERNEL_SRC_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE}
)
target_compile_options(int8_fused_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)
target_link_libraries(int8_fused_ops PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)
set_target_properties(int8_fused_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# --- INT8 micro-benchmark ---
add_executable(bench_int8 bench_int8.cpp)
target_include_directories(bench_int8 PRIVATE
    src/
    ${KERNEL_SRC_DIR}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(bench_int8 PRIVATE
    int8_fused_ops
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

# --- Print build info ---
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUTLASS include: ${CUTLASS_INCLUDE}")
message(STATUS "CSRC directory: ${CSRC_DIR}")
message(STATUS "Kernel source dir: ${KERNEL_SRC_DIR}")
message(STATUS "Kernels directory: ${KERNELS_DIR}")
