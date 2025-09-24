/***************************************************************************************************
 * Copyright 2024 Beijing Institute of Technology AETAS Lab. and Utarn Technology Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************************************/

#include "elementwise_scalar_mul.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <algorithm>
#include <utility>

// Simple CUDA kernel for small tensors: one thread handles one element
template<typename ScalarType>
__global__ void scalar_bool_multiply_kernel_simple(
    const ScalarType scalar_value,
    const bool* __restrict__ bool_matrix,
    ScalarType* __restrict__ output,
    const int total_elements) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Simple processing: each thread handles one element (optimal for small tensors)
    for (int i = tid; i < total_elements; i += stride) {
        bool val = bool_matrix[i];
        output[i] = val ? scalar_value : ScalarType(0);
    }
}

// Optimized vectorized CUDA kernel for large tensors: one thread handles 4 elements (cuBLAS-style)
template<typename ScalarType>
__global__ void scalar_bool_multiply_kernel_vectorized(
    const ScalarType scalar_value,
    const bool* __restrict__ bool_matrix,
    ScalarType* __restrict__ output,
    const int total_elements) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vectorized processing: each thread handles 4 elements (cuBLAS-style)
    for (int i = tid * 4; i < total_elements; i += stride * 4) {
        // Process 4 elements at once for better memory bandwidth utilization
        if (i + 3 < total_elements) {
            // Read 4 consecutive bool values
            bool b1 = bool_matrix[i];
            bool b2 = bool_matrix[i + 1];
            bool b3 = bool_matrix[i + 2];
            bool b4 = bool_matrix[i + 3];
            
            // Compute 4 results with branchless operations
            output[i]     = b1 ? scalar_value : ScalarType(0);
            output[i + 1] = b2 ? scalar_value : ScalarType(0);
            output[i + 2] = b3 ? scalar_value : ScalarType(0);
            output[i + 3] = b4 ? scalar_value : ScalarType(0);
        } else {
            // Handle remaining elements (less than 4)
            for (int j = i; j < total_elements && j < i + 4; j++) {
                bool val = bool_matrix[j];
                output[j] = val ? scalar_value : ScalarType(0);
            }
        }
    }
}

// Simple CUDA kernel for small tensors: one thread handles one element
template<typename ScalarType>
__global__ void bool_scalar_multiply_kernel_simple(
    const bool* __restrict__ bool_matrix,
    const ScalarType scalar_value,
    ScalarType* __restrict__ output,
    const int total_elements) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Simple processing: each thread handles one element (optimal for small tensors)
    for (int i = tid; i < total_elements; i += stride) {
        bool val = bool_matrix[i];
        output[i] = val ? scalar_value : ScalarType(0);
    }
}

// Optimized vectorized CUDA kernel for large tensors: one thread handles 4 elements (cuBLAS-style)
template<typename ScalarType>
__global__ void bool_scalar_multiply_kernel_vectorized(
    const bool* __restrict__ bool_matrix,
    const ScalarType scalar_value,
    ScalarType* __restrict__ output,
    const int total_elements) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vectorized processing: each thread handles 4 elements (cuBLAS-style)
    for (int i = tid * 4; i < total_elements; i += stride * 4) {
        // Process 4 elements at once for better memory bandwidth utilization
        if (i + 3 < total_elements) {
            // Read 4 consecutive bool values
            bool b1 = bool_matrix[i];
            bool b2 = bool_matrix[i + 1];
            bool b3 = bool_matrix[i + 2];
            bool b4 = bool_matrix[i + 3];
            
            // Compute 4 results with branchless operations
            output[i]     = b1 ? scalar_value : ScalarType(0);
            output[i + 1] = b2 ? scalar_value : ScalarType(0);
            output[i + 2] = b3 ? scalar_value : ScalarType(0);
            output[i + 3] = b4 ? scalar_value : ScalarType(0);
        } else {
            // Handle remaining elements (less than 4)
            for (int j = i; j < total_elements && j < i + 4; j++) {
                bool val = bool_matrix[j];
                output[j] = val ? scalar_value : ScalarType(0);
            }
        }
    }
}

// Adaptive threshold for kernel selection based on empirical testing
// Elements below this threshold use simple kernel, above use vectorized kernel
static const int VECTORIZATION_THRESHOLD = 6250000;  // ~2500x2500 elements (optimized based on benchmark)

// Helper function to determine optimal kernel strategy
static bool should_use_vectorized_kernel(int total_elements) {
    return total_elements >= VECTORIZATION_THRESHOLD;
}

// Helper function to calculate optimal launch configuration (adaptive based on kernel type)
static std::pair<int, int> calculate_launch_config(int total_elements, bool use_vectorized) {
    // Get device properties for optimal configuration
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int block_size;
    int grid_size;
    
    if (use_vectorized) {
        // Vectorized kernel: use larger blocks for better occupancy
        if (prop.major >= 8) {
            // Ampere architecture: use 256 threads for optimal balance
            block_size = 256;
        } else if (prop.major >= 7) {
            // Turing/Volta: 256 threads work well
            block_size = 256;
        } else if (prop.major >= 6) {
            // Pascal: 128-256 threads
            block_size = 256;
        } else {
            // Older architectures: conservative setting
            block_size = 128;
        }
        
        // Calculate optimal grid size for vectorized processing
        // Each thread processes 4 elements, so adjust accordingly
        int threads_needed = (total_elements + 3) / 4;  // Round up for vectorized processing
        int max_blocks = prop.multiProcessorCount * 8;  // cuBLAS typically uses ~8 blocks per SM
        grid_size = std::min((threads_needed + block_size - 1) / block_size, max_blocks);
    } else {
        // Simple kernel: optimize for small tensors with lower overhead
        if (prop.major >= 8) {
            // Ampere: smaller blocks for small tensors
            block_size = 128;
        } else if (prop.major >= 7) {
            // Turing/Volta: moderate block size
            block_size = 128;
        } else if (prop.major >= 6) {
            // Pascal: standard size
            block_size = 128;
        } else {
            // Older architectures: conservative
            block_size = 64;
        }
        
        // Calculate grid size for simple processing (one thread per element)
        int max_blocks = prop.multiProcessorCount * 4;  // Fewer blocks for small tensors
        grid_size = std::min((total_elements + block_size - 1) / block_size, max_blocks);
    }
    
    return {grid_size, block_size};
}

// Host functions with adaptive kernel selection
template<typename ScalarType>
void elementwise_scalar_bool_multiply(
    const ScalarType scalar_value,
    const bool* bool_matrix,
    ScalarType* output,
    const int total_elements,
    cudaStream_t stream) {
    
    if (total_elements <= 0) return;
    
    // Adaptive kernel selection based on tensor size
    bool use_vectorized = should_use_vectorized_kernel(total_elements);
    auto [grid_size, block_size] = calculate_launch_config(total_elements, use_vectorized);
    
    if (use_vectorized) {
        // Use vectorized kernel for large tensors (>= 3000x3000)
        scalar_bool_multiply_kernel_vectorized<<<grid_size, block_size, 0, stream>>>(
            scalar_value, bool_matrix, output, total_elements);
    } else {
        // Use simple kernel for small tensors (< 3000x3000)
        scalar_bool_multiply_kernel_simple<<<grid_size, block_size, 0, stream>>>(
            scalar_value, bool_matrix, output, total_elements);
    }
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        const char* kernel_type = use_vectorized ? "vectorized" : "simple";
        throw std::runtime_error("Adaptive scalar_bool_multiply kernel (" + std::string(kernel_type) + 
                                ") launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

template<typename ScalarType>
void elementwise_bool_scalar_multiply(
    const bool* bool_matrix,
    const ScalarType scalar_value,
    ScalarType* output,
    const int total_elements,
    cudaStream_t stream) {
    
    if (total_elements <= 0) return;
    
    // Adaptive kernel selection based on tensor size
    bool use_vectorized = should_use_vectorized_kernel(total_elements);
    auto [grid_size, block_size] = calculate_launch_config(total_elements, use_vectorized);
    
    if (use_vectorized) {
        // Use vectorized kernel for large tensors (>= 3000x3000)
        bool_scalar_multiply_kernel_vectorized<<<grid_size, block_size, 0, stream>>>(
            bool_matrix, scalar_value, output, total_elements);
    } else {
        // Use simple kernel for small tensors (< 3000x3000)
        bool_scalar_multiply_kernel_simple<<<grid_size, block_size, 0, stream>>>(
            bool_matrix, scalar_value, output, total_elements);
    }
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        const char* kernel_type = use_vectorized ? "vectorized" : "simple";
        throw std::runtime_error("Adaptive bool_scalar_multiply kernel (" + std::string(kernel_type) + 
                                ") launch failed: " + std::string(cudaGetErrorString(error)));
    }
}

// Explicit template instantiations
template void elementwise_scalar_bool_multiply<float>(const float, const bool*, float*, const int, cudaStream_t);
template void elementwise_scalar_bool_multiply<double>(const double, const bool*, double*, const int, cudaStream_t);
template void elementwise_scalar_bool_multiply<int>(const int, const bool*, int*, const int, cudaStream_t);
template void elementwise_scalar_bool_multiply<int64_t>(const int64_t, const bool*, int64_t*, const int, cudaStream_t);

template void elementwise_bool_scalar_multiply<float>(const bool*, const float, float*, const int, cudaStream_t);
template void elementwise_bool_scalar_multiply<double>(const bool*, const double, double*, const int, cudaStream_t);
template void elementwise_bool_scalar_multiply<int>(const bool*, const int, int*, const int, cudaStream_t);
template void elementwise_bool_scalar_multiply<int64_t>(const bool*, const int64_t, int64_t*, const int, cudaStream_t);
