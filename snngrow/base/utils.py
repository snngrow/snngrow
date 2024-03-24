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

import logging
import torch.nn as nn
from .neuron import BaseNode

def reset(net: nn.Module):
    """
    * :ref:`中文API <reset-cn>`

    .. _reset-cn:

    :param net: 任何基于 ``nn.Module`` 子类构建的网络

    :return: None

    重置网络中神经元状态

    * :ref:`API in English <reset-en>`

    .. _reset-en:

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