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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "cutlass_scalar_mul.h"
#include "elementwise_scalar_mul.h"

// ========== Generic template function (should not be called) ==========
template <typename type_A,
    typename type_B,
    typename type_C>
void cutlass_scalar_mul(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // Generic template should not be called, only specialized versions will be implemented
    throw std::runtime_error("Unsupported scalar multiplication type combination");
}

// ========== float x bool -> float specialization ==========
template <>
void cutlass_scalar_mul<float, bool, float>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is scalar, B is bool matrix, C is output
    float scalar_value = A.item<float>();
    
    bool* B_ptr = B.data_ptr<bool>();
    float* C_ptr = C.data_ptr<float>();
    
    int total_elements = B.numel();
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call our elementwise kernel
    elementwise_scalar_bool_multiply<float>(scalar_value, B_ptr, C_ptr, total_elements, stream);
    
    // Synchronize and check for errors
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        throw std::runtime_error("CUTLASS scalar float x bool multiplication failed [FROM cutlass_scalar_mul.cu]: " + std::string(cudaGetErrorString(error)));
    }
}

// ========== bool x float -> float specialization ==========
template <>
void cutlass_scalar_mul<bool, float, float>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is bool matrix, B is scalar, C is output
    float scalar_value = B.item<float>();
    
    bool* A_ptr = A.data_ptr<bool>();
    float* C_ptr = C.data_ptr<float>();
    
    int total_elements = A.numel();
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call our elementwise kernel
    elementwise_bool_scalar_multiply<float>(A_ptr, scalar_value, C_ptr, total_elements, stream);
}

// ========== double x bool -> double specialization ==========
template <>
void cutlass_scalar_mul<double, bool, double>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is scalar, B is bool matrix, C is output
    double scalar_value = A.item<double>();
    
    bool* B_ptr = B.data_ptr<bool>();
    double* C_ptr = C.data_ptr<double>();
    
    int total_elements = B.numel();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_scalar_bool_multiply<double>(scalar_value, B_ptr, C_ptr, total_elements, stream);
}

// ========== bool x double -> double specialization ==========
template <>
void cutlass_scalar_mul<bool, double, double>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is bool matrix, B is scalar, C is output
    double scalar_value = B.item<double>();
    
    bool* A_ptr = A.data_ptr<bool>();
    double* C_ptr = C.data_ptr<double>();
    
    int total_elements = A.numel();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_bool_scalar_multiply<double>(A_ptr, scalar_value, C_ptr, total_elements, stream);
}

// ========== int x bool -> int specialization ==========
template <>
void cutlass_scalar_mul<int, bool, int>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is scalar, B is bool matrix, C is output
    int scalar_value = A.item<int>();
    
    bool* B_ptr = B.data_ptr<bool>();
    int* C_ptr = C.data_ptr<int>();
    
    int total_elements = B.numel();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_scalar_bool_multiply<int>(scalar_value, B_ptr, C_ptr, total_elements, stream);
}

// ========== bool x int -> int specialization ==========
template <>
void cutlass_scalar_mul<bool, int, int>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is bool matrix, B is scalar, C is output
    int scalar_value = B.item<int>();
    
    bool* A_ptr = A.data_ptr<bool>();
    int* C_ptr = C.data_ptr<int>();
    
    int total_elements = A.numel();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_bool_scalar_multiply<int>(A_ptr, scalar_value, C_ptr, total_elements, stream);
}

// ========== int64_t x bool -> int64_t specialization ==========
template <>
void cutlass_scalar_mul<int64_t, bool, int64_t>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is scalar, B is bool matrix, C is output
    int64_t scalar_value = A.item<int64_t>();
    
    bool* B_ptr = B.data_ptr<bool>();
    int64_t* C_ptr = C.data_ptr<int64_t>();
    
    int total_elements = B.numel();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_scalar_bool_multiply<int64_t>(scalar_value, B_ptr, C_ptr, total_elements, stream);
}

// ========== bool x int64_t -> int64_t specialization ==========
template <>
void cutlass_scalar_mul<bool, int64_t, int64_t>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    // A is bool matrix, B is scalar, C is output
    int64_t scalar_value = B.item<int64_t>();
    
    bool* A_ptr = A.data_ptr<bool>();
    int64_t* C_ptr = C.data_ptr<int64_t>();
    
    int total_elements = A.numel();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    elementwise_bool_scalar_multiply<int64_t>(A_ptr, scalar_value, C_ptr, total_elements, stream);
}
