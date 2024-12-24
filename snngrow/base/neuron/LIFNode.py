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

from . import BaseNode
from ..surrogate import Sigmoid
from typing import Callable
import torch

class LIFNode(BaseNode.BaseNode):
    """
    :param tau: membrane time constant
    :type tau: float

    :param decay_input: the input will decay
    :type decay_input: bool

    :param v_threshold: threshold voltage
    :type v_threshold: float

    :param v_reset: reset voltage. If not ``None``, the neuron's voltage will be set to ``v_reset``
        after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
    :type v_reset: float

    :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
    :type surrogate_function: Callable

    :param detach_reset: detach the computation graph of reset in backward
    :type detach_reset: bool

    :param parallel_optim: parallel optimization
    :type parallel_optim: bool

    :param T: time steps
    :type T: int

    :param spike_out: whether to output SpikeTensor
    :type spike_out: bool

    The Leaky Integrate-and-Fire(LIF) neuron

    """
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = Sigmoid.Sigmoid(),
                 detach_reset: bool = False, parallel_optim: bool = False, T: int = 1, spike_out: bool = False):
        
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, parallel_optim, T, spike_out)

        self.tau = tau
        self.decay_input = decay_input

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_dynamics(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_dynamics_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_dynamics_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_dynamics_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_dynamics_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    def neuronal_dynamics_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    def neuronal_dynamics_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    def neuronal_dynamics_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    def neuronal_dynamics_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v
