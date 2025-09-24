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

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

// Optimized host functions for elementwise scalar-bool multiplication
// Features:
// - Memory access optimization with __restrict__ and __ldg
// - Grid-stride loops for better occupancy
// - Branchless computation
// - Dynamic launch configuration based on GPU architecture

template<typename ScalarType>
void elementwise_scalar_bool_multiply(
    const ScalarType scalar_value,
    const bool* bool_matrix,
    ScalarType* output,
    const int total_elements,
    cudaStream_t stream = 0);

template<typename ScalarType>
void elementwise_bool_scalar_multiply(
    const bool* bool_matrix,
    const ScalarType scalar_value,
    ScalarType* output,
    const int total_elements,
    cudaStream_t stream = 0);
