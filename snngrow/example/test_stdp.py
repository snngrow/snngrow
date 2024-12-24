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
import torch.nn as nn
import snngrow.base.nn as tnn
from snngrow.base.neuron.IFNode import IFNode
from snngrow.base.neuron.LIFNode import LIFNode
from snngrow.base.surrogate import Sigmoid
from snngrow.base import utils
from snngrow.base.learning import *
from matplotlib import pyplot as plt

class STDP_SNN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.node = []
        self.connection = []
        self.node.append(IFNode(parallel_optim=False, T=T, spike_out=False, surrogate_function=Sigmoid.Sigmoid(spike_out=False), v_threshold=1.0))
        self.connection.append(tnn.Linear(4, 3, spike_in=False, bias=False))
        self.stdp = []
        self.stdp.append(STDP(self.node[0], self.connection[0]))

    def forward(self, x):
        """
        Calculate the forward propagation process and the training process.
        """
        output, dw = self.stdp[0](x)
    
        return output, dw
    
        
    def updateweight(self, i, dw, delta):
        """
        :param i: the index of the connection to update
        :type: float

        :param dw: updated weights
        :type x: torch.Tensor

        Update the weight of the ith group connection according to the input dw value.
        """
        self.connection[i].update(dw*delta)
        
    def reset(self):
        """
        Reset neurons or intermediate quantities of learning rules.
        """
        for i in range(len(self.node)):
            self.node[i].reset()
        for i in range(len(self.stdp)):
            self.stdp[i].reset()


torch.manual_seed(0)

if __name__ == '__main__':

    N_in, N_out = 4, 3
    T = 40
    batch_size = 2
    lr = 0.01

    in_spike = (torch.rand([T, batch_size, N_in]) > 0.7).float()

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []

    stdp_snn = STDP_SNN()
    nn.init.constant_(stdp_snn.connection[0].weight.data, 0.4)
    for t in range(T):
        output, dw = stdp_snn(in_spike[t])
        out_spike.append(output)
        trace_pre.append(stdp_snn.stdp[0].trace_pre)
        trace_post.append(stdp_snn.stdp[0].trace_post)
        stdp_snn.updateweight(0,dw*lr,1)
        weight.append(stdp_snn.connection[0].weight.data.clone())      

    #print('out_spike:',out_spike)
    #print('out_spike.shape:',out_spike.shape)
    #print('out_spike.shape:',out_spike.shape)

    out_spike = torch.stack(out_spike)   # [T, batch_size, N_out]
    trace_pre = torch.stack(trace_pre)   # [T, batch_size, N_in]
    trace_post = torch.stack(trace_post) # [T, batch_size, N_out]
    weight = torch.stack(weight)         # [T, N_out, N_in]

    t = torch.arange(0, T).float()
    
    in_spike = in_spike[:, 0, 0]
    out_spike = out_spike[:, 0, 0]
    trace_pre = trace_pre[:, 0, 0]
    trace_post = trace_post[:, 0, 0]
    weight = weight[:, 0, 0]

    #print("in_spike:", in_spike)
    #print("out_spike:", out_spike)
    #print("trace_pre:", trace_pre)
    #print("trace_post:", trace_post)
    #print("weight:", weight)

    cmap = plt.get_cmap('tab10')

    plt.eventplot((in_spike * t)[in_spike == 1], lineoffsets=0.401, colors=cmap(0), label='in_spike', orientation='horizontal', linelengths=0.002)
    plt.eventplot((out_spike * t)[out_spike == 1], lineoffsets=0.399, colors=cmap(1), label='out_spike', orientation='horizontal', linelengths=0.002)
    plt.plot(t, weight, c=cmap(2), label='weight')
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('weight', rotation=0)
    plt.xlabel('time step')

    plt.legend()    
    plt.gcf().subplots_adjust(left=0.18)    
    plt.show()
    plt.savefig('test_stdp.png')