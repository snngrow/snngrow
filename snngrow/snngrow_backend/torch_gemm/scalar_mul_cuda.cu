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
#include <c10/core/GradMode.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/Resize.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include <ATen/ExpandUtils.h>
#include <ATen/ops/addmm_native.h>

#include <pybind11/pybind11.h>

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/Parallel.h>

#include "scalar_mul_cuda.h"
#include "cutlass_scalar_mul.h"

static torch::Tensor _scalar_mul_cuda_impl(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
    at::NoNamesGuard guard;
    const auto dim_tensor1 = tensor1.dim();
    const auto dim_tensor2 = tensor2.dim();

    // 检查至少有一个操作数是标量
    TORCH_CHECK(tensor1.numel() == 1 || tensor2.numel() == 1,
        "scalar_mul requires at least one argument to be a scalar, but they have ",
        tensor1.numel(), " and ", tensor2.numel(), " elements");

    // 确定哪个是标量，哪个是矩阵
    bool tensor1_is_scalar = (tensor1.numel() == 1);
    bool tensor2_is_scalar = (tensor2.numel() == 1);
    const auto& scalar_tensor = tensor1_is_scalar ? tensor1 : tensor2;
    const auto& matrix_tensor = tensor1_is_scalar ? tensor2 : tensor1;

    // 确定输出类型 - 如果是布尔×标量，输出标量类型；如果是标量×布尔，输出标量类型
    auto options = matrix_tensor.dtype() == torch::kBool ? scalar_tensor.options() : matrix_tensor.options();
    auto out = at::empty({ 0 }, options);

    // 梯度检查
    TORCH_CHECK(!(tensor1.requires_grad() || tensor2.requires_grad() || out.requires_grad()) || !at::GradMode::is_enabled(),
        "scalar_mul(): functions with out=... arguments don't support automatic differentiation, "
        "but one of the arguments requires grad."
    );

    // 标量乘法的核心逻辑
    if (tensor1_is_scalar && tensor2.dim() >= 1) {
        // 标量 × 矩阵
        const auto& scalar = tensor1;
        const auto& matrix = tensor2;
        
        // 输出形状与矩阵相同
        at::native::resize_output(out, matrix.sizes());

        // 根据类型调用CUTLASS实现
        if (scalar.dtype() == torch::kFloat && matrix.dtype() == torch::kBool) {
            cutlass_scalar_mul<float, bool, float>(scalar, matrix, out);
        }
        else if (scalar.dtype() == torch::kDouble && matrix.dtype() == torch::kBool) {
            cutlass_scalar_mul<double, bool, double>(scalar, matrix, out);
        }
        else if (scalar.dtype() == torch::kInt && matrix.dtype() == torch::kBool) {
            cutlass_scalar_mul<int, bool, int>(scalar, matrix, out);
        }
        else if (scalar.dtype() == torch::kLong && matrix.dtype() == torch::kBool) {
            cutlass_scalar_mul<int64_t, bool, int64_t>(scalar, matrix, out);
        }
        else {
            TORCH_CHECK(false, "Unsupported scalar_mul type combination: scalar=", scalar.dtype(), ", matrix=", matrix.dtype());
        }

    }
    else if (tensor1.dim() >= 1 && tensor2_is_scalar) {
        // 矩阵 × 标量
        const auto& matrix = tensor1;
        const auto& scalar = tensor2;

        // 输出形状与矩阵相同
        at::native::resize_output(out, matrix.sizes());

        // 根据类型调用CUTLASS实现
        if (matrix.dtype() == torch::kBool && scalar.dtype() == torch::kFloat) {
            cutlass_scalar_mul<bool, float, float>(matrix, scalar, out);
        }
        else if (matrix.dtype() == torch::kBool && scalar.dtype() == torch::kDouble) {
            cutlass_scalar_mul<bool, double, double>(matrix, scalar, out);
        }
        else if (matrix.dtype() == torch::kBool && scalar.dtype() == torch::kInt) {
            cutlass_scalar_mul<bool, int, int>(matrix, scalar, out);
        }
        else if (matrix.dtype() == torch::kBool && scalar.dtype() == torch::kLong) {
            cutlass_scalar_mul<bool, int64_t, int64_t>(matrix, scalar, out);
        }
        else {
            TORCH_CHECK(false, "Unsupported scalar_mul type combination: matrix=", matrix.dtype(), ", scalar=", scalar.dtype());
        }

    }
    else {
        // 两个标量相乘 - 直接使用PyTorch内置操作
        out = tensor1 * tensor2;
    }

    return out;
}

at::Tensor scalar_mul_cuda(at::Tensor tensor1, at::Tensor tensor2) {
    // 检查输入张量是否在同一设备上 (如果都在CUDA上)
    if (tensor1.is_cuda() && tensor2.is_cuda() && tensor1.device() != tensor2.device()) {
        AT_ERROR("Input tensors must be on the same device");
    }

    // 检查输入张量是否在CUDA设备上 (至少有一个在CUDA上)
    if (!tensor1.is_cuda() && !tensor2.is_cuda()) {
        AT_ERROR("At least one input tensor must be on the CUDA device");
    }

    // 如果其中一个在CPU上，将其移到CUDA上
    if (!tensor1.is_cuda()) {
        tensor1 = tensor1.cuda();
    }
    if (!tensor2.is_cuda()) {
        tensor2 = tensor2.cuda();
    }

    // 检查输入张量是否连续
    if (!tensor1.is_contiguous()) {
        tensor1 = tensor1.contiguous();
    }
    if (!tensor2.is_contiguous()) {
        tensor2 = tensor2.contiguous();
    }

    // 执行标量乘法计算
    auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
    at::Tensor result;
    result = _scalar_mul_cuda_impl(tensor1, tensor2);
    at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);

    return result;
}

// Python 接口包装函数 - 支持 Python 标量和 Tensor 混合输入
at::Tensor scalar_mul_python(pybind11::object arg1, pybind11::object arg2) {
    // 辅助函数：将 Python 对象转换为 Tensor
    auto to_tensor = [](pybind11::object obj, torch::Device device = torch::kCUDA) -> at::Tensor {
        try {
            // 尝试直接转换为 Tensor
            return obj.cast<at::Tensor>();
        } catch (...) {
            // 如果转换失败，尝试作为标量
            if (py::isinstance<py::float_>(obj)) {
                return torch::tensor(obj.cast<double>(), torch::TensorOptions().device(device));
            } else if (py::isinstance<py::int_>(obj)) {
                return torch::tensor(obj.cast<int64_t>(), torch::TensorOptions().device(device));
            } else {
                throw std::runtime_error("Unsupported argument type. Expected Tensor, float, or int.");
            }
        }
    };

    // 首先确定设备 - 优先使用 Tensor 的设备
    torch::Device device = torch::kCUDA;
    try {
        if (auto tensor = arg1.cast<at::Tensor>(); tensor.defined()) {
            device = tensor.device();
        }
    } catch (...) {
        try {
            if (auto tensor = arg2.cast<at::Tensor>(); tensor.defined()) {
                device = tensor.device();
            }
        } catch (...) {
            // 如果都不是tensor，使用默认CUDA设备
        }
    }

    // 转换参数为 Tensor
    at::Tensor tensor1 = to_tensor(arg1, device);
    at::Tensor tensor2 = to_tensor(arg2, device);

    // 调用核心实现
    return scalar_mul_cuda(tensor1, tensor2);
}