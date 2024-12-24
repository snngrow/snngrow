# Copyright 2024 Beijing Institute of Technology AETAS Lab. and Utarn Technology Co., Ltd. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from snngrow.base import SpikeTensor
import snngrow_backend


class LinearFunction(Function):
    """
    Custom Linear function.

    This function performs a linear operation on the SpikeTensor using the provided weight tensor.
    It supports both forward and backward computations.

    Args:
        inputs (SpikeTensor): The input tensor.
        weight (torch.Tensor): The weight tensor.

    Returns:
        torch.Tensor: The output tensor.

    Raises:
        ValueError: If the input dim is not 3 or 4.
        NotImplementedError: If the device type is not supported.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        inputs: SpikeTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Convert the Tensor to a Variable and save it to ctx
        ctx.for_backwards = (inputs, weight, bias)
        output = snngrow_backend.spike_gemm_cuda(inputs.elem, weight.t().contiguous())
        if bias is not None:
            output += bias

        return output


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):

        # grad output is the gradient value calculated from the previous level of backpropagation
        inputs, weight, bias = ctx.for_backwards
        grad_input = grad_weight = grad_bias = None
        # represents the gradient of the input, weights, and bias
        # Determine whether the corresponding variables need to reverse derivative to calculate the gradient
        if ctx.needs_input_grad[0]:
            # Derivative of composition, chain rule
            grad_input = grad_output @ weight 
        if ctx.needs_input_grad[1]:
            # dense * spike, derivative of composition, chain rule
            grad_weight = snngrow_backend.spike_gemm_cuda(grad_output.t().contiguous(), inputs.elem)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
 
        return grad_input, grad_weight, grad_bias


def linear(
    inputs: SpikeTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    linear operation.

    Args:
        inputs (SpikeTensor): Input tensor.
        weight (torch.Tensor): Linear weights.
        bias (Optional[torch.Tensor], optional): Bias tensor. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after linear operation, it is the dense tensor.
    """
    output = LinearFunction.apply(
            inputs,
            weight,
            bias,
        )
    return output
