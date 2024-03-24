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

from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
import copy

from ..surrogate import Sigmoid

class BaseNode(nn.Module):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Sigmoid.Sigmoid(), detach_reset: bool = False,
                 parallel_optim: bool = True):
        """
        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 阈值电压
        :type v_threshold: float

        :param v_reset: 重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: reset过程的计算图分离
        :type detach_reset: bool

        可微分SNN神经元的基类

        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage
        :type v_threshold: float

        :param v_reset: reset voltage. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: float

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: detach the computation graph of reset in backward
        :type detach_reset: bool

        The base class of differentiable spiking neurons.
        """       
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        assert isinstance(parallel_optim, bool)
        super().__init__()

        self._memories = {}
        self._memories_rv = {}

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.parallel_optim = parallel_optim

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @staticmethod
    def hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    def soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-cn:

        神经元充电方程，子类必须实现该函数

        * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-en:

        The charge difference equation. The sub-class must implement this function.
        """

        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-cn:

        神经元放电方程

        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-en:

        The fire difference equation.
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """
        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-cn:

        神经元重置方程

        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-en:

        The reset difference equation.
        """

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, parallel_optim={self.parallel_optim}'

    def simple_forward(self, x: torch.Tensor):
        """

        * :ref:`中文API <BaseNode.simple_forward-cn>`

        .. _BaseNode.simple_forward-cn:

        :param x: 输入电压增量
        :type x: torch.Tensor

        :return: 输出脉冲
        :rtype: torch.Tensor

        前向传播过程，按照充电-放电-重置顺序进行。

        * :ref:`API in English <BaseNode.simple_forward-en>`

        .. _BaseNode.simple_forward-en:

        :param x: increment of voltage inputted
        :type x: torch.Tensor

        :return: out spikes
        :rtype: torch.Tensor

        Forward by the order of charge - fire - reset.

        """

        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

    def parallel_optim_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        """
        * :ref:`中文 API <BaseNode.parallel_optim_forward-cn>`

        .. _BaseNode.parallel_optim_forward-cn:

        :param x: 输入tensor ``shape = [T, N, *] ``
        :type x: torch.Tensor

        并行计算的前向传播函数，通过调用 ``T`` 次 ``simple_forward(x[t], *args, **kwargs)`` 实现


        * :ref:`API in English <BaseNode.parallel_optim_forward-en>`

        .. _BaseNode.parallel_optim_forward-en:

        :param x: input tensor with ``shape = [T, N, *] ``
        :type x: torch.Tensor

        The parallel forward function, which is implemented by calling ``simple_forward(x[t], *args, **kwargs)`` over ``T`` times

        """
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.simple_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))

        return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        if self.parallel_optim :
            return self.parallel_optim_forward(*args, **kwargs)
        else :
            return self.simple_forward(*args, **kwargs)

    def register_memory(self, name: str, value):
        """
        * :ref:`中文API <BaseNode.register_memory-cn>`

        .. _BaseNode.register_memory-cn:

        :param name: 变量名
        :type name: str
        :param value: 变量值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`API in English <BaseNode.register_memory-en>`

        .. _BaseNode.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`中文API <BaseNode.reset-cn>`

        .. _BaseNode.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`API in English <BaseNode.reset-en>`

        .. _BaseNode.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        """
        * :ref:`中文API <BaseNode.memories-cn>`

        .. _BaseNode.memories-cn:

        :return: 所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`API in English <BaseNode.memories-en>`

        .. _BaseNode.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`中文API <BaseNode.named_memories-cn>`

        .. _BaseNode.named_memories-cn:

        :return: 所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`API in English <BaseNode.named_memories-en>`

        .. _BaseNode.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`中文API <BaseNode.detach-cn>`

        .. _BaseNode.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`API in English <BaseNode.detach-en>`

        .. _BaseNode.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """
        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica
