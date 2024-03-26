教程
---------

=================
神经元
=================

在SNNGrow中，脉冲神经元是脉冲神经网络的基本单元。不同于深度学习中常见的神经元，脉冲神经元具有生物仿生的神经动力学，并且使用离散的脉冲值作为输出。脉冲神经元的输出是离散的，通常为0或1。在SNNGrow中，神经元的数量是在初始化或调用``reset()``函数重新初始化后，根据第一次接收的输入的``shape``自动决定的。重置神经元状态的代码可以在:meth:`snngrow.base.utils`中找到：

.. code-block:: python

    def reset(net: nn.Module):
    
      for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, BaseNode.BaseNode):
                logging.warning(f'Trying to call `reset()` of {m}, which is not snngrow.base.neuron'
                                f'.BaseNode')
            m.reset()

得益于神经元动力学，脉冲神经元是有状态的，也可以说具有记忆。通常，脉冲神经元的膜电位作为其状态变量。在喂入下一个样本之前，需要调用``reset()``函数清除脉冲神经元的先前状态。SNNGrow神经元都继承自:meth:`snngrow.base.neuron.BaseNode`，共享相同的fire和reset方程。任何离散的脉冲神经元都可以用三个离散方程描述（神经动力学，激发，重置）。神经动力学和重置的方程如下。

.. math::

  V[t]=f(V[t-1],X[t])

  S[t]=\Theta(V[t]-V_{threshold})

.. code-block:: python

    def neuronal_dynamics(self, x: torch.Tensor):

        raise NotImplementedError

    def neuronal_fire(self):

        raise NotImplementedError

其中 :math:`X[t]` 是输入，如外部输入电流； :math:`V[t]` 是输出脉冲后的神经元膜电位； :math:`f(V[t-1],X[t])` 是神经元状态的神经动力学方程。不同类型的神经元的神经动力学方程是不同的； :math:`\Theta(x)` 是激活函数。在这个框架中，广泛使用的一个函数是阶跃（Heaviside）函数。在前向传播过程中，如果输入大于或等于阈值，则返回1；否则返回0。这样的``tensor``只有0或1元素被视为脉冲。阶跃函数的方程如下：

.. math::

  \Theta(x)=\left\{\begin{matrix}
                0, x\ge 0 \\
                1, x< 0
        \end{matrix}\right.

输出脉冲会消耗先前由脉冲神经元积累的电荷，导致膜电位的瞬时降低，即膜电位的重置。在SNNGrow中，膜电位有两种重置方式：

1. 硬重置模式，在输出脉冲后，膜电位直接重置到重置电压：

.. math:: V[t]=V[t](1-S[t])+V_{reset}S[t]

.. code-block:: python

  def hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
      v = (1. - spike) * v + spike * v_reset
      return v


2. 软重置模式，在输出脉冲后，膜电位与阈值电压的差值作为重置电压：

.. math:: V[t]=V[t]-V_{threshold}S[t]

.. code-block:: python

  def soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

软重置的神经元不需要重置电压 :math:`V_{reset}` 变量。在:meth:`snngrow.base.neuron.BaseNode`的神经元中，其中一个构造函数参数 :math:`V_{reset}`，默认为1.0，表示神经元可以使用硬重置；如果设置为None，则使用软模式重置。


====================
替代梯度
====================

在SNNGrow中，前向传播使用阶跃函数。但是阶跃函数是不连续的，其导数是Dirichlet函数（冲击函数），其方程是：

.. math::

  \delta (x)=\left\{\begin{matrix}
                +\infty , x= 0 \\
                0, x\neq 0
        \end{matrix}\right.

Dirichlet函数在0处为 :math:`+\infty`。如果直接使用Dirichlet函数进行梯度下降，将使网络的训练极其不稳定。因此，在反向传播期间使用替代梯度 [1]_。

替代梯度方法的原理是，在前向传播期间使用 :math:`\Theta(x)`，而在反向传播期间使用 :math:`\frac{\mathrm{d} y}{\mathrm{d} x} =\sigma ^{'} (x)`，其中 :math:`\sigma (x)` 是替代函数。 :math:`\sigma (x)`通常是与 :math:`\Theta(x)` 形状相似的函数，但是光滑和连续的。替代函数在神经元中用于生成脉冲的近似梯度。

在SNNGrow中，替代梯度函数在基类中实现，提供了一些常用函数的替代。替代函数可以作为参数指定给神经元构造函数，``surrogate_function``。

..  [1] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63. 