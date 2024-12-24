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

from itertools import repeat
from typing import List, Tuple, Union
import logging
import torch
import torch.nn as nn
from .neuron import BaseNode

def reset(net: nn.Module):
    """
    :param net: Any network inherits from ``nn.Module``

    :return: None

    Reset neurons in the network.
    """
    
    for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, BaseNode.BaseNode):
                logging.warning(f'Trying to call `reset()` of {m}, which is not snngrow.base.neuron'
                                f'.BaseNode')
            m.reset()


def make_tuple(
    x: Union[int, List[int], Tuple[int, ...], torch.Tensor], ndim: int, name: str
) -> Tuple[int, ...]:
    """
    Make an n-tuple from the input.

    :param x: The input value.
    :param ndim: The desired dimension of the tuple.
    :param name: The name of the input value.
    :return: The n-tuple.
    """
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, name + " must be an integer or a tuple of " + str(ndim) + " integers."
    return x

