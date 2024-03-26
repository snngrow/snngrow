# Copyright 2024 Utarn Technology Co., Ltd.
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

from . import BaseNode
from ..surrogate import Sigmoid
from typing import Callable
import torch

class IFNode(BaseNode.BaseNode):
    """
    :param v_threshold: threshold voltage
    :type v_threshold: float

    :param v_reset: reset voltage. If not ``None``, the neuron's voltage will be set to ``v_reset``
        after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
    :type v_reset: float

    :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
    :type surrogate_function: Callable

    :param detach_reset: detach the computation graph of reset in backward
    :type detach_reset: bool

    The Integrate-and-Fire(IF) neuron, without decay input as LIF neuron.
    
    """
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Sigmoid.Sigmoid(), detach_reset: bool = False):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    def eval_forward_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    def eval_forward_soft_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    def eval_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    def eval_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    def forward(self, x: torch.Tensor):
        if self.training:
            return super().forward(x)
            
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.eval_forward_soft_reset(x, self.v, self.v_threshold)
            else:
                spike, self.v = self.eval_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
            return spike