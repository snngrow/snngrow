# Copyright 2024 Beijing Institute of Technology AETAS Lab. and Utarn Technology Co., Ltd. All rights reserved. 
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

from typing import Union
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from snngrow.base import SpikeTensor
import snngrow_backend


class ScalarMultiplyFunction(Function):
    """
    Custom Scalar Multiply function for SpikeTensor.

    This function performs scalar multiplication with SpikeTensor using CUDA backend.
    Supports: scalar × spike_tensor → tensor

    Args:
        spike_input (SpikeTensor): The spike tensor input.
        scalar (torch.Tensor): The scalar tensor (must have numel() == 1).

    Returns:
        torch.Tensor: The output tensor.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        spike_input: SpikeTensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        # Save for backward
        ctx.for_backwards = (spike_input, scalar)
        
        # Call CUDA backend
        output = snngrow_backend.scalar_mul_cuda(spike_input.elem, scalar)
        
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        spike_input, scalar = ctx.for_backwards
        grad_spike = grad_scalar = None
        
        if ctx.needs_input_grad[0]:
            # Gradient w.r.t. spike_input (SpikeTensor)
            # Since spike is binary, gradient is zero
            grad_spike = None
            
        if ctx.needs_input_grad[1]:
            # Gradient w.r.t. scalar
            # d(scalar * spike)/d(scalar) = spike
            grad_scalar = (grad_output * spike_input.elem.float()).sum()
            
        return grad_spike, grad_scalar


def scalar_multiply(
    spike_input: SpikeTensor,
    scalar: Union[torch.Tensor, float, int],
) -> torch.Tensor:
    """
    Scalar multiplication with SpikeTensor.

    Performs element-wise multiplication between a scalar and spike tensor.
    The operation is: output[i,j] = scalar * spike_input[i,j]

    Args:
        spike_input (SpikeTensor): Input spike tensor (boolean).
        scalar (Union[torch.Tensor, float, int]): Scalar value or tensor.

    Returns:
        torch.Tensor: Output tensor with same shape as spike_input.

    Examples:
        >>> spike = SpikeTensor(torch.randint(0, 2, (10, 20), dtype=torch.bool).cuda())
        >>> scalar = torch.tensor(2.5).cuda()
        >>> output = scalar_multiply(spike, scalar)
        >>> # output[i,j] = 2.5 if spike[i,j] == True, else 0.0
    """
    # Convert scalar to tensor if needed
    if isinstance(scalar, (int, float)):
        scalar = torch.tensor(scalar, device=spike_input.elem.device, dtype=torch.float32)
    elif not isinstance(scalar, torch.Tensor):
        raise TypeError(f"scalar must be torch.Tensor, int, or float, got {type(scalar)}")
    
    # Ensure scalar is a scalar tensor
    if scalar.numel() != 1:
        raise ValueError(f"scalar must have exactly 1 element, got {scalar.numel()}")
    
    # Ensure same device
    if spike_input.elem.device != scalar.device:
        scalar = scalar.to(spike_input.elem.device)
    
    # Call the function
    output = ScalarMultiplyFunction.apply(spike_input, scalar)
    
    return output
