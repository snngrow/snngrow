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

import torch.nn as nn
import torch

class BatchNorm2d(nn.Module):
    def __init__(self, dim, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.gamma = nn.Parameter(torch.ones([dim]))
        self.beta = nn.Parameter(torch.zeros([dim]))
        self.register_buffer("moving_mean", torch.zeros([dim]))
        self.register_buffer("moving_var", torch.ones([dim]))
        self.register_buffer("momentum",torch.tensor(momentum))
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=[0,2,3])
            var = x.var(dim=[0,2,3])
            self.moving_mean = self.moving_mean * self.momentum + mean * (1 - self.momentum)
            self.moving_var  = self.moving_var * self.momentum + var * (1 - self.momentum)
        x = (x - self.moving_mean) / (torch.sqrt(self.moving_var + 1e-8))
        return x * self.gamma + self.beta   
    

class LayerNorm(nn.Module):
    def __init__(self,num_features,eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
 
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True,unbiased=False)
        normalized_x = (x - mean) / (std + self.eps)
        return self.gamma * normalized_x + self.beta