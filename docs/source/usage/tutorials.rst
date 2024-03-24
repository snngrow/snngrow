Tutorials
---------

=================
神经元
=================

在SNNGrow中，脉冲神经元是脉冲神经网络的基本单元。只能输出脉冲，即0或1的神经元为“脉冲神经元”，使用脉冲神经元的网络为脉冲神经网络(Spiking Neural Networks, SNNs)。
神经元的数量是在初始化或调用 ``reset()`` 函数重新初始化后，根据第一次接收的输入的 ``shape`` 自动决定的。在 :meth:`snngrow.base.utils` 中可以找到重置神经元状态的代码：

.. code-block:: python

    def reset(net: nn.Module):
    
      for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, BaseNode.BaseNode):
                logging.warning(f'Trying to call `reset()` of {m}, which is not snngrow.base.neuron'
                                f'.BaseNode')
            m.reset()

脉冲神经元是有状态的，也可以说是有记忆。通常情况下，脉冲神经元的膜电位作为它的状态变量。输入下一个样本之前，需要先调用 ``reset()`` 函数清除脉冲神经元之前的状态。
不同的神经元，充电方程不尽相同。但膜电位超过阈值电压后，输出脉冲，以及输出脉冲后，膜电位的重置都是相同的。SNNGrow中的神经元全部继承自 :meth:`snngrow.base.neuron.BaseNode` ，共享相同的放电、重置方程。
用3个离散方程（充电、放电、重置）就可以描述任意的离散脉冲神经元。充电和放电的方程为：

.. math::

  H[t]=f(V[t-1],X[t])

  S[t]=\Theta(H[t]-V_{threshold})

.. code-block:: python

    def neuronal_charge(self, x: torch.Tensor):

        raise NotImplementedError

    def neuronal_fire(self):

        return self.surrogate_function(self.v - self.v_threshold)

其中 :math:`\Theta(x)` 是构造函数参数中的 ``surrogate_function`` ，前向传播时是阶跃（Heaviside）函数，输入大于或等于0，返回1，否则返回0。这种元素仅为0或1的 ``tensor`` 被视为脉冲。Heaviside函数的方程为：

.. math::

  \Theta(x)=\left\{\begin{matrix}
                0, x\ge 0 \\
                1, x< 0
        \end{matrix}\right.

输出脉冲消耗了神经元之前积累的电荷，因此膜电位会有一个瞬间的降低，即膜电位的重置。在SNNGrow中，对膜电位的重置有2种方式：

1. Hard方式，输出脉冲后，膜电位直接作为重置电压：

.. math:: V[t]=H[t](1-S[t])+V_{reset}S[t]

.. code-block:: python

  def hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
      v = (1. - spike) * v + spike * v_reset
      return v


2. Soft方式，输出脉冲后，膜电位与阈值电压的差作为重置电压：

.. math:: V[t]=H[t]-V_{threshold}S[t]

.. code-block:: python

  def soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

其中 :math:`X[t]` 是外部的输入，例如电压的增量；为了区分不同状态下的膜电位，使用 :math:`H[t]` 表示神经元充电后、输出脉冲前的膜电位；:math:`V[t]` 是神经元输出脉冲后的膜电位；:math:`f(V[t-1],X[t])` 是神经元状态的更新方程，不同类型的神经元，主要区别在于更新方程不同。
使用Soft方式重置的神经元，不需要重置电压 :math:`V_{reset}` 这个变量。 :meth:`snngrow.base.neuron.BaseNode` 中的神经元，其构造函数的参数之一 :math:`V_{reset}`，默认为 1.0 ，表示神经元使用Hard方式重置；若设置为 None，则使用Soft方式重置。


=================
替代梯度
=================

SNNGrow中，网络的前向传播使用Heaviside函数。但Heaviside函数不连续，其导数是狄利克雷函数（冲击函数），狄利克雷函数的方程为：

.. math::

  \delta (x)=\left\{\begin{matrix}
                +\infty , x= 0 \\
                0, x\neq 0
        \end{matrix}\right.

狄利克雷函数在0处为 :math:`+\infty` ，如果直接使用狄利克雷函数进行梯度下降，会使网络的训练极其不稳定。因此，在反向传播过程中，我们使用梯度替代法。
梯度替代法的原理是，在前向传播时使用 :math:`\Theta(x)` ，而在反向传播时则使用 :math:`\frac{\mathrm{d} y}{\mathrm{d} x} =\sigma ^{'} (x)` ，而不是 :math:`\frac{\mathrm{d} y}{\mathrm{d} x} =\Theta ^{'} (x)` ，其中 :math:`\sigma (x)` 即为替代函数。:math:`\sigma (x)` 通常是一个形状与 :math:`\Theta(x)` 类似，但光滑连续的函数。替代函数在神经元中被用于生成脉冲。

SNNGrow在 :meth:`snngrow.base.surrogate.BaseFunction` 中实现了替代函数的基类，并且提供了一些常用的替代函数，可以通过神经元构造函数的参数 ``surrogate_function`` 来指定替代函数。











=================
neuron
=================

In SNNGrow, spiking neurons are the basic units of spiking neural networks. Neurons that can only output spikes, that is 0 or 1, are Spiking neurons. The Networks that use spiking neurons are Spiking Neural Networks (SNNs).
The number of neurons is automatically determined based on the ``shape`` of the first received input, after initialization or reinitialization by calling the ``reset()`` function. The code to reset the neuron's state can be found in :meth:`snngrow.base.utils` :

.. code-block:: python

    def reset(net: nn.Module):
    
      for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, BaseNode.BaseNode):
                logging.warning(f'Trying to call `reset()` of {m}, which is not snngrow.base.neuron'
                                f'.BaseNode')
            m.reset()


Spiking neurons are stateful, which can also be said to have memory. Typically, the membrane potential of a spiking neuron serves as its state variable. The ``reset()`` function needs to be called to clear the previous state of the spiking neuron before feeding the next sample.
The charge equations are not the same for different neurons. But after the membrane potential exceeds the threshold voltage, the output spike, and after the output spike, the reset of the membrane potential is the same. SNNGrow neurons all inherit from :meth:`snngrow.base.neuron.BaseNode` , share the same fire and reset equations.
Any discrete spiking neuron can be described by three discrete equations (charge, fire, reset). The equations for charge and fire are as follows.

.. math::

  H[t]=f(V[t-1],X[t])

  S[t]=\Theta(H[t]-V_{threshold})

.. code-block:: python

    def neuronal_charge(self, x: torch.Tensor):

        raise NotImplementedError

    def neuronal_fire(self):

        return self.surrogate_function(self.v - self.v_threshold)

Where :math:`\Theta(x)` is the ``surrogate_function`` in the constructor argument and is a step (Heaviside) function when forward propagated that returns 1 for inputs greater than or equal to 0 and 0 otherwise. Such a ``tensor`` with only 0 or 1 elements is treated as a spike. The equation for the Heaviside function is as follows.

.. math::

  \Theta(x)=\left\{\begin{matrix}
                0, x\ge 0 \\
                1, x< 0
        \end{matrix}\right.

The output spike consumes the charge previously accumulated by the neuron, so there is an instant decrease in the membrane potential, a reset of the membrane potential. In SNNGrow, the membrane potential is reset in 2 ways:

1. Hard mode, after the output spike, the membrane potential is directly used as the reset voltage:

.. math:: V[t]=H[t](1-S[t])+V_{reset}S[t]

.. code-block:: python

  def hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
      v = (1. - spike) * v + spike * v_reset
      return v


2. Soft mode, after the output spike, the difference between the membrane potential and the threshold voltage is used as the reset voltage:

.. math:: V[t]=H[t]-V_{threshold}S[t]

.. code-block:: python

  def soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

Where :math:`X[t]` is an input, such as a voltage increment; To distinguish the membrane potential in different states, use :math:`H[t]` to represent the membrane potential after the neuron has been charged and before the output spike; :math:`V[t]` is the membrane potential of the neuron after the output spike; :math:`f(V[t-1],X[t])` is the update equation for the neuron state. The main difference is that the update equation is different for different types of neurons.
Soft reset neurons do not need to reset the voltage :math:`V_{reset}` variable. :meth:`snngrow.base.neuron.BaseNode` of neurons, one of the constructor parameters :math:`V_{reset}`, the default is 1.0, said a neuron can use Hard reset; If it is set to None, Soft mode is used to reset.


====================
Surrogate Gradient
====================

In SNNGrow, the Heaviside function is used for the forward propagation of the network. But the Heaviside function is discontinuous, and its derivative is a Dirichlet function (the shock function) whose equation is:

.. math::

  \delta (x)=\left\{\begin{matrix}
                +\infty , x= 0 \\
                0, x\neq 0
        \end{matrix}\right.

The Dirichlet function is :math:`+\infty` at 0. If you directly use the Dirichlet function for gradient descent, it will make the training of the network extremely unstable. Therefore, we use surrogate gradient during backpropagation.
Surrogate Gradient method is in before to the spread of use :math:`\Theta(x)` , and is used when back propagation :math:`\frac{\mathrm{d} y}{\mathrm{d} x} =\sigma ^{'} (x)` , Rather than :math:`\frac{\mathrm{d} y}{\mathrm{d} x} =\Theta ^{'} (x)` , among them :math:`\sigma (x)` is the surrogate function.:math:`\sigma (x)` is usually a function similar in shape to :math:`\Theta(x)` , but smooth and continuous. Surrogate functions are used in neurons to generate spikes.

SNNGrow in :meth:`snngrow.base.surrogate.BaseFunction` implements the surrogate function in the base class, and provides an alternative for some commonly used functions, The surrogate function can be specified as an argument to the neuron constructor,  ``surrogate_function`` .