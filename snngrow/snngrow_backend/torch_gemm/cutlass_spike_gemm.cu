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

#include "cutlass/gemm/device/gemm.h"                         // 引入cutlass头文件
#include <cutlass/numeric_types.h>
#include "cutlass/gemm/device/gemm_universal.h"
#include "gemm/device/gemm_spike.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "cutlass_spike_gemm.h"

template <typename type_A,
          typename type_B,
          typename type_C>
class GetGemm { 
};

template <>
class GetGemm<bool, float, float> {
public:
  using CutlassGemm = cutlass::gemm::device::GemmSpike<
        bool,                           // ElementA
        cutlass::layout::RowMajor,              // LayoutA
        float,                           // ElementB
        cutlass::layout::RowMajor,              // LayoutB
        float,                           // ElementOutput
        cutlass::layout::RowMajor,            // LayoutOutput
        float,                                     // ElementAccumulator
        cutlass::arch::OpClassSimt,            // tag indicating Tensor Cores
        cutlass::arch::Sm80                        // tag indicating target GPU compute architecture
      >;
};

template <>
class GetGemm<float, bool, float> {
public:
  using CutlassGemm = cutlass::gemm::device::GemmSpike<
        float,                           // ElementA
        cutlass::layout::RowMajor,              // LayoutA
        bool,                           // ElementB
        cutlass::layout::RowMajor,              // LayoutB
        float,                           // ElementOutput
        cutlass::layout::RowMajor,            // LayoutOutput
        float,                                     // ElementAccumulator
        cutlass::arch::OpClassSimt,            // tag indicating Tensor Cores
        cutlass::arch::Sm80                        // tag indicating target GPU compute architecture
      >;
};

template <typename type_A_,
          typename type_B_,
          typename type_C_>
void cutlass_spike_gemm(const at::Tensor A, const at::Tensor B, at::Tensor C) {
};

template <>
void cutlass_spike_gemm<bool, float, float>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using CutlassGemm = typename GetGemm<bool, float, float>::CutlassGemm;
  
  int lda = A.stride(0);
  int ldb = B.stride(0);
  int ldc = C.stride(0);
  int M = A.size(0);
  int N = B.size(1);
  int K = A.size(1);

  float alpha = 1.0;
  float beta = 0.0;

  bool *A_ptr = A.data_ptr<bool>();
  float *B_ptr = B.data_ptr<float>();
  float *C_ptr = C.data_ptr<float>();

  CutlassGemm gemm_op;
  cutlass::Status status;
  typename CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
    {A_ptr, lda},     // source matrix A
    {B_ptr, ldb},     // source matrix B
    {C_ptr, ldc},
    {C_ptr, ldc},
    {alpha, beta});
  status = gemm_op(args); 
}

template<>
void cutlass_spike_gemm<float, bool, float>(const at::Tensor A, const at::Tensor B, at::Tensor C) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using CutlassGemm = typename GetGemm<float, bool, float>::CutlassGemm;
  
  int lda = A.stride(0);
  int ldb = B.stride(0);
  int ldc = C.stride(0);
  int M = A.size(0);
  int N = B.size(1);
  int K = A.size(1);

  float alpha = 1.0;
  float beta = 0.0;

  float *A_ptr = A.data_ptr<float>();
  bool *B_ptr = B.data_ptr<bool>();
  float *C_ptr = C.data_ptr<float>();

  CutlassGemm gemm_op;
  cutlass::Status status;
  typename CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
    {A_ptr, lda},     // source matrix A
    {B_ptr, ldb},     // source matrix B
    {C_ptr, ldc},
    {C_ptr, ldc},
    {alpha, beta});
  status = gemm_op(args);
}