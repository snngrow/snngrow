#include <torch/extension.h>
#include <stdexcept>

template <typename type_A,
          typename type_B,
          typename type_C>
void cutlass_spike_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C);