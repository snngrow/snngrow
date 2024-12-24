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

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snngrow.base.neuron import LIFNode
from snngrow.base.neuron import IFNode
from snngrow.base.surrogate import Sigmoid
from snngrow.base.surrogate import ATan
from snngrow.base import utils
import torch

print(torch.__version__)

lifnode = LIFNode.LIFNode()
ifnode = IFNode.IFNode()

lifnode = LIFNode.LIFNode(parallel_optim=True, surrogate_function=ATan.ATan())
ifnode = IFNode.IFNode(surrogate_function=ATan.ATan())

x = torch.randn(2, 3, 4, 5)
y = torch.randn(6, 7, 8, 9)

a = lifnode(x)
utils.reset(lifnode)
b = lifnode(y)