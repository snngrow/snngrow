/**
 * Copyright 2024 BIT AETAS
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "torch_gemm/spike_gemm_cuda.h"

/**
 * @brief Pybind11 module for the CUDA backend.
 * 
 * @param m The module.
*/
PYBIND11_MODULE(snngrow_backend, m) {
  m.def("spike_gemm_cuda", &spike_gemm_cuda, "Bool Spike Matrix Multiplication GEMM CUDA");
  
}
