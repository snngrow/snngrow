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
from .BaseFunction import SurrogateFunctionBase
from .BaseFunction import heaviside



def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None, None


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, spike_out):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x, spike_out)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Sigmoid(SurrogateFunctionBase):
    '''
    :param alpha: parameter to control smoothness of gradient
    :param spiking: output spikes. The default is ``True`` which means that using ``heaviside`` in forward
        propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
        using the primitive function of the surrogate gradient function used in backward propagation

    The sigmoid surrogate spiking function.

    '''
    def __init__(self, alpha=4.0, spike_out = False, spiking=True):
        super().__init__(alpha, spike_out, spiking)

    @staticmethod
    def spiking_function(x, alpha, spike_out):
        return sigmoid.apply(x, alpha, spike_out)

    @staticmethod

    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()

    @staticmethod
    
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]