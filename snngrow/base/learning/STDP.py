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

import numpy as np
import torch
import os
import sys
from torch import nn
from torch.nn import Parameter

import abc
import math
from abc import ABC

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from ..neuron import BaseNode

class STDP(nn.Module):
    """
    :param node: BaseNode neuron
    :type x: torch.Tensor

    :param connection: an instance of a connection layer can have only one operation
    :type: torch.nn.Module

    :param trace_pre: the trace of a presynaptic neuron
    :type: float

    :param trace_post: the trace of a postsynaptic neuron
    :type: float

    :param tau_pre: the time constant of the trace of a presynaptic neuron
    :type: float

    :param tau_post: the time constant of the trace of a postsynaptic neuron
    :type: float

    STDP learning rule.
    """

    def __init__(self, node: BaseNode.BaseNode, connection: nn.Module, 
        trace_pre: float = 0., trace_post: float = 0., tau_pre: float = 2., tau_post: float = 2.):
        
        super().__init__()

        self.node = node
        self.connection = connection
        self.trace_pre = trace_pre
        self.trace_post = trace_post
        self.tau_pre = tau_pre
        self.tau_post = tau_post


    def forward(self, x):
        """
        :param x: input spike
        :type: torch.Tensor

        :param spike: output spike
        :rtype: torch.Tensor

        :param dw: weight update
        :rtype: torch.nn.Module

        The forward propagation process.
        """
        x = x.clone().detach()
        # shape = [batch_size, N_in]
        i = self.connection(x)
        with torch.no_grad():
            spike = self.node(i)
            # shape = [batch_size, N_out]
            dw = self.cal_trace(x, spike)

        return spike, dw

    def cal_trace(self, in_spike, out_spike):
        """
        :param in_spike: input spike
        :type: torch.Tensor

        :param out_spike: out spike
        :type: torch.Tensor

        :param dw: weight update
        :rtype: torch.nn.Module

        Calculate weight update and trace.
        """

        self.trace_pre = self.trace_pre - self.trace_pre / self.tau_pre + in_spike      # shape = [batch_size, N_in]
        self.trace_post = self.trace_post - self.trace_post / self.tau_post + out_spike # shape = [batch_size, N_out]

        # [batch_size, N_out, N_in] -> [N_out, N_in]
        delta_w_pre = -torch.clamp(self.connection.weight.data, -1, 1.) * (self.trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)
        delta_w_post = torch.clamp(self.connection.weight.data, -1, 1.) * (self.trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
        dw = delta_w_pre + delta_w_post
        
        return dw


    def reset(self):
        """
        Reset trace and time constant to default.
        """

        self.trace_pre = 0.
        self.trace_post = 0.

        self.tau_pre = 2.
        self.tau_post = 2.

