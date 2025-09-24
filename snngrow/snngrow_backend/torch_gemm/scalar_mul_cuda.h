
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
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/Parallel.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// 统一的标量乘法接口 - 支持 Tensor 和标量输入
at::Tensor scalar_mul_cuda(at::Tensor A, at::Tensor B);

// Python 接口包装函数 - 支持 Python 标量和 Tensor 混合输入
at::Tensor scalar_mul_python(pybind11::object arg1, pybind11::object arg2);