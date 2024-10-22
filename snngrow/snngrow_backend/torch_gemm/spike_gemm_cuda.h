#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

at::Tensor spike_gemm_cuda(at::Tensor A, at::Tensor B);