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

import torch
import math
from .BaseFunction import SurrogateFunctionBase
from .BaseFunction import heaviside

def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, spike_out):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x, spike_out)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class ATan(SurrogateFunctionBase):
    '''
    :param alpha: parameter to control smoothness of gradient
    :param spiking: output spikes. The default is ``True`` which means that using ``heaviside`` in forward
        propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
        using the primitive function of the surrogate gradient function used in backward propagation

    The arc tangent surrogate spiking function

    '''
    def __init__(self, alpha=2.0, spike_out = False, spiking=True):
        super().__init__(alpha, spike_out, spiking)

    @staticmethod

    def spiking_function(x, alpha, spike_out):
        return atan.apply(x, alpha, spike_out)

    @staticmethod

    def primitive_function(x: torch.Tensor, alpha: float):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    @staticmethod

    def backward(grad_output, x, alpha):
        return atan_backward(grad_output, x, alpha)[0]

