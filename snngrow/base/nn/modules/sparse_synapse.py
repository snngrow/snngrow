# Copyright 2024 BIT AETAS
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

import math
from typing import Union
import torch
from torch import nn
import torch.nn.functional as F

from snngrow.base import SpikeTensor
from snngrow.base.nn import functional as snngrow_F

__all__ = ["SparseSynapse"]

class SparseSynapse(nn.Module):
    """
    A custom neural network module that represents a sparse synapse layer.
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        connection (Union[torch.Tensor, str], optional): Connection pattern for the synapse. If None, no connection pattern is applied.
            If a tensor is provided, it is used as the connection pattern. If 'random', a random connection pattern is generated. Default: None.
        device (optional): The device on which the parameters are allocated. Default: None.
        dtype (optional): The data type of the parameters. Default: None.
    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (torch.Tensor): The learnable weights of the module of shape (out_features, in_features).
        bias (torch.Tensor or None): The learnable bias of the module of shape (out_features).
        connection (torch.Tensor or None): The connection pattern for the synapse.
    Methods:
        reset_parameters():
            Initializes the weights and bias of the layer.
        forward(input: torch.Tensor or SpikeTensor) -> torch.Tensor:
            Defines the computation performed at every call.
        extra_repr() -> str:
            Returns the extra representation of the module.
    """


    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    connection: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, connection: Union[torch.Tensor, str] = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseSynapse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if connection is None:
            self.connection = None
        elif isinstance(connection, torch.Tensor):
            self.connection = nn.Parameter(connection)
        elif connection == 'random':
            self.connection = nn.Parameter(torch.randint(0, 2, (out_features, in_features), **factory_kwargs))
        else:
            raise ValueError(f"Invalid value for connection: {connection}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Union[torch.Tensor, SpikeTensor]) -> torch.Tensor:
        self.connection = self.connection.to(self.weight.device) if self.connection is not None else None
        weight = self.weight * self.connection if self.connection is not None else self.weight
        if not isinstance(input, SpikeTensor):
            return F.linear(input, weight, self.bias)
        else:
            return snngrow_F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )