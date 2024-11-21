#include <c10/core/GradMode.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/Resize.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include <ATen/ExpandUtils.h>
#include <ATen/ops/addmm_native.h>


#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/Parallel.h>

#include "spike_gemm_cuda.h"
#include "cutlass_spike_gemm.h"

static bool should_fold(const at::Tensor& tensor1, const at::Tensor& tensor2) {
    // We check that we can fold the larger tensor into a matrix and dispatch to mm or mv rather than
    // to bmm. We want to make sure we can do so without incurring in any extra copy
    const auto tensor1_larger = tensor1.dim() >= tensor2.dim();
  
    // We order the tensors. t1 will be the larger tensor
    // We can always transpose tensor2 as the dimensions are always >= 1 (precondition from matmul)
    // and tensor1_larger iff tensor2.dim() > tensor1.dim(9
    const auto t1 = tensor1_larger ? c10::MaybeOwned<at::Tensor>::borrowed(tensor1)
                                   : c10::MaybeOwned<at::Tensor>::owned(tensor2.mT());
    const int64_t dim_t1 = t1->dim();
    const auto dim_t2 = tensor1_larger ? tensor2.dim()
                                       : tensor1.dim();
  
    // Just fold for dim_t1 >= 3 and (dim_t2 == 1 || dim_t2 == 2)
    if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
      return false;
    }
  
    // In this case we *do* incur in an extra copy to avoid creating an unnecessary large tensor in the backward
    // Suppose we don't fold here. Let t1.shape = [b, m, n] t2.shape = [n, k] like in a transformer
    // t2 will be expanded to a tensor of shape [b, n, k] and then we do t1.bmm(t2_expanded)
    // The issue appears in the backward.
    // The output gradient g of this operation would have shape [b, m, k]
    // The backward wrt. t2 of bmm would be given by t1.mH @ g, which has shape [b, n, k]
    // Then, the backward of expand is simply `sum(0)`. As such, we are instantiating a tensor
    // of shape [b, n, k] unnecessarily, which may cause a large memory footprint, and in the
    // worst case, an OOM
    bool t2_requires_grad = tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
    if (t2_requires_grad) {
      // We should be checking !at::GradMode::is_enabled(), but apparently
      // this regresses performance in some cases:
      // https://github.com/pytorch/pytorch/issues/118548#issuecomment-1916022394
      return true;
    }
  
    // Don't fold in this case, as we would have to call mm on the transposed tensor, the result
    // would be contiguous, and then we would need to transpose it and call contiguous on it, thus
    // having to copy the tensor
    if (tensor1.dim() == 2) {
      return false;
    }
  
    // Can always fold if the tensor is empty
    // This serves as a precondition for the code below
    if (t1->numel() == 0) {
      return true;
    }
  
    // t1->view(-1, t1->size(-1)) does not copy only when the first n-1 dimensions are contiguous
    // in the sense that t1_stride[i] = t1_stride[i+1]*t1_shape[i+1]
    const auto t1_shape = t1->sizes();
    const auto t1_strides = t1->strides();
    for (auto i = int64_t{0}; i < dim_t1 - int64_t{2}; ++i) {
      if (t1_strides[i] != t1_strides[i+1] * t1_shape[i+1]) {
        return false;
      }
    }
    return true;
}

static torch::Tensor _spike_gemm_cuda_impl(const torch::Tensor& tensor1, const torch::Tensor& tensor2){
    at::NoNamesGuard guard;
    const auto dim_tensor1 = tensor1.dim();
    const auto dim_tensor2 = tensor2.dim();
    
    // This is checked up here to simplify the logic below
    // Note that the strings are just evaluated on failure, so almost always we just evaluate
    // the condition and move on
    TORCH_CHECK(dim_tensor1 != 0 && dim_tensor2 != 0,
        "both arguments to matmul need to be at least 1D, but they are ",
        dim_tensor1, "D and ", dim_tensor2, "D");
    
    const auto spike_mul_dense = (tensor1.dtype() == torch::kBool);
    auto options = spike_mul_dense ? tensor2.options() : tensor1.options();
    auto out = at::empty({0}, options);
    
    // Usually we would rely on the out= kernels we decompose into to check this, but
    // for matmul there is logic at the composite level that relies on this invariant.
    TORCH_CHECK(!(tensor1.requires_grad() || tensor2.requires_grad() || out.requires_grad()) || !at::GradMode::is_enabled(),
        "matmul(): functions with out=... arguments don't support automatic differentiation, "
        "but one of the arguments requires grad."
    );

    if (dim_tensor1 == 1 && dim_tensor2 == 1) {
        at::dot_out(out, tensor1, tensor2);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
        at::mv_out(out, tensor1, tensor2);
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        const auto t1 = at::MaybeOwned<at::Tensor>::borrowed(tensor1);
        const auto t2 = at::MaybeOwned<at::Tensor>::borrowed(tensor2);

        const auto sizes_1 = t1->sizes();
        auto output_shape = at::DimVector(sizes_1.begin(), sizes_1.end() - 1);
        const auto folded_dim1 = c10::multiply_integers(output_shape);
        output_shape.push_back(t2->sizes()[1]);

        at::native::resize_output(out, output_shape);
        //out.zero_();

        const auto out_ = c10::MaybeOwned<at::Tensor>::borrowed(out);
        auto reshaped_out = out_->reshape({folded_dim1, t2->sizes().back()});
        
        // use cutlass_spike_gemm to perform the matrix multiplication
        if (tensor1.dtype() == torch::kBool) {
            cutlass_spike_gemm<bool, float, float>(tensor1, tensor2, reshaped_out);
        } else {
            cutlass_spike_gemm<float, bool, float>(tensor1, tensor2, reshaped_out);
        }
        
        if (!reshaped_out.is_alias_of(out)) {
            out_->copy_(reshaped_out.view_as(*out_));
        }
        return out;
    } else if (should_fold(tensor1, tensor2)) {
        // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2) ||
        // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
        // and at least one of the following two conditions hold
        // - the small tensor requires grad (see should_fold for the why)
        // - we can fold the larger tensor t1 into a matrix as t1.view(-1, t1.size(-1)) without copying

        // optimization: use mm instead of bmm by folding the batch of the larger tensor
        // into its leading matrix dimension
        const auto transpose = dim_tensor2 > dim_tensor1;
        const auto t1 = transpose ? c10::MaybeOwned<at::Tensor>::owned(tensor2.mT())
                                : c10::MaybeOwned<at::Tensor>::borrowed(tensor1);
        const auto t2 = !transpose ? c10::MaybeOwned<at::Tensor>::borrowed(tensor2)
                                : dim_tensor1 == 2
                                    ? c10::MaybeOwned<at::Tensor>::owned(tensor1.t())
                                    : c10::MaybeOwned<at::Tensor>::borrowed(tensor1);
        // Invariant: t1->dim() >= 3 && (t2->dim() == 1 || t2->dim() == 2)
        //            and *t1 and *t2 are matmul-compatible

        // Why not t1->view(-1, sizes_1.back())?
        // If the last dim is 0, then view(-1, 0) won't work because the -1 becomes ambiguous.
        // This can happen in e.g. [3, 5, 0] @ [0, 0].
        const auto sizes_1 = t1->sizes();
        auto output_shape = at::DimVector(sizes_1.begin(), sizes_1.end() - 1);
        const auto folded_dim1 = c10::multiply_integers(output_shape);

        // Readjust output_shape if we are multiplying by a matrix
        const auto t2_is_matrix = t2->dim() == 2;
        if (t2_is_matrix) {
            output_shape.push_back(t2->sizes()[1]);
        }
        // This will almost always be a view.
        // It may not be a view if t2->requires_grad(). See should_fold for an explanation
        const auto t1_folded = t1->reshape({folded_dim1, sizes_1.back()});

        // Resize output into the correct shape
        at::native::resize_output(out, output_shape);
        //out.zero_();

        // We then reshape the output to the expected shape and call mm/mv
        // and transpose back if necessary
        auto reshaped_out = t2_is_matrix ? out.reshape({folded_dim1, t2->sizes().back()})
                                        : out.reshape({folded_dim1});
        if (t2_is_matrix) {
            if (tensor1.dtype() == torch::kBool && !transpose) {
                cutlass_spike_gemm<bool, float, float>(t1_folded, *t2, reshaped_out);
            } else {
                cutlass_spike_gemm<float, bool, float>(t1_folded, *t2, reshaped_out);
            }
        } else {
            at::mv_out(reshaped_out, t1_folded, *t2);
        }
        if (!reshaped_out.is_alias_of(out)) {
            out.copy_(reshaped_out);
        }
        return out;
    } else {
        TORCH_CHECK(false, "matmul(): arguments with shapes ", tensor1.sizes(), " and ", tensor2.sizes(), " are not broadcastable");
    }
    return out;
}

at::Tensor spike_gemm_cuda(at::Tensor tensor1, at::Tensor tensor2) {
    // Check if the input tensors are on the same device
    if (tensor1.device() != tensor2.device()) {
        AT_ERROR("Input tensors must be on the same device");
    }
    
    // Check if the input tensors are on the CUDtensor1 device
    if (!tensor1.is_cuda() || !tensor2.is_cuda()) {
        AT_ERROR("Input tensors must be on the CUDtensor1 device");
    }
    
    // Check if the input tensors are contiguous
    if (!tensor1.is_contiguous() || !tensor2.is_contiguous()) {
        AT_ERROR("Input tensors must be contiguous");
    }
    
    // at::Tensor result;
    auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
    at::Tensor result;
    result = _spike_gemm_cuda_impl(tensor1, tensor2);
    at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);

    return result;
}