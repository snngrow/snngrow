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

import torch
from torch import nn

from .neuron import BaseNode

class Monitor:
    def __init__(self, net: nn.Module, device: str = None):
        '''
        :param net: Network to be monitored
        :type net: nn.Module

        :param device: Device carrying and processing monitored data. Only take effect when backend is set to ``'torch'``. Can be string ``'cpu', 'cuda', 'cuda:0'`` or ``torch.device``, defaults to ``None``
        :type device: str, optional

        Monitor the firing rate and silent neuron ratio of the network
        '''
    
        super().__init__()
        self.module_dict = dict()
        for name, module in net.named_modules():
            if (isinstance(module, BaseNode.BaseNode)):
                self.module_dict[name] = module

        self.net = net

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError('Expected a cuda or cpu device, but got: {}'.format(device))

    def enable(self):
        '''

        Enable Monitor. Start recording data.
        '''
        self.handle = dict.fromkeys(self.module_dict, None)
        self.neuron_cnt = dict.fromkeys(self.module_dict, None)

        for name, module in self.module_dict.items():
            setattr(module, 'neuron_cnt', self.neuron_cnt[name])

            # Initialize the handle of the lookahead hook
            self.handle[name] = module.register_forward_hook(self.forward_hook)
                
        self.reset()


    def disable(self):
        '''

        Disable Monitor. Stop recording data.
        '''
        for name, module in self.module_dict.items():
            delattr(module, 'neuron_cnt')
            delattr(module, 'fire_mask')
            delattr(module, 'firing_time')
            delattr(module, 'cnt')

            # Delete the handle of the lookahead hook
            self.handle[name].remove()

    
    # Monitor the firing rate and silent neuron ratio of the network
    @torch.no_grad()
    def forward_hook(self, module, input, output):
        if isinstance(module, BaseNode.BaseNode) and module.parallel_optim:
            output_shape = output.shape
            data = output.view([-1,] + list(output_shape[2:])).clone() # 对于多步模块的输出[T, batchsize, ...]的前两维进行合并
        else:
            data = output.clone()

        data = data.to(self.device)
        if module.neuron_cnt is None:
            module.neuron_cnt = data[0].numel()  # Number of neurons
        module.firing_time += torch.sum(data)  # Total number of spikes in data
        module.cnt += data.numel()  # The size of the data (T*batchsize* number of neurons)
        fire_mask = (torch.sum(data, dim=0) > 0)  # mask (Bool type) of whether each neuron has fired a spike or not.
        
        # The logical_or operation of Bool tensors can be directly represented by |. And you can operate directly with Python's Bool type, but the first operand must be a Bool Tensor, not a Python Bool
        module.fire_mask = fire_mask | module.fire_mask 



    def reset(self):
        '''

        Delete previously recorded data
        '''
        for name, module in self.module_dict.items():
            setattr(module, 'fire_mask', False)
            setattr(module, 'firing_time', 0)
            setattr(module, 'cnt', 0)


    def get_avg_firing_rate(self, all: bool = True, module_name: str = None) -> torch.Tensor or float:
        '''

        :param all: Whether needing firing rate averaged on all layers, defaults to ``True``
        :type all: bool, optional

        :param module_name: Name of concerned layer. Only take effect when all is ``False``
        :type module_name: str, optional

        :return: Averaged firing rate on concerned layers
        :rtype: torch.Tensor or float
        '''
        if all:
            ttl_firing_time = 0
            ttl_cnt = 0
            for name, module in self.module_dict.items():
                ttl_firing_time += module.firing_time
                ttl_cnt += module.cnt
            return ttl_firing_time / ttl_cnt 
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'Invalid module_name \'{module_name}\'')
            module = self.module_dict[module_name]
            return module.firing_time / module.cnt



    def get_nonfire_ratio(self, all: bool = True, module_name: str = None) -> torch.Tensor or float:
        '''

        :param all: Whether needing ratio of silent neurons of all layers, defaults to ``True``
        :type all: bool, optional

        :param module_name: Name of concerned layer. Only take effect when all is ``False``
        :type module_name: str, optional

        :return: Ratio of silent neurons on concerned layers
        :rtype: torch.Tensor or float
        '''
        if all:
            ttl_neuron_cnt = 0
            ttl_zero_cnt = 0
            for name, module in self.module_dict.items():
                ttl_zero_cnt += torch.logical_not(module.fire_mask).sum()
                ttl_neuron_cnt += module.neuron_cnt
            return ttl_zero_cnt / ttl_neuron_cnt
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'Invalid module_name \'{module_name}\'')

            module = self.module_dict[module_name]
            return torch.logical_not(module.fire_mask).sum() / module.neuron_cnt