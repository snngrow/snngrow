Tutorials
---------

=================
Neuron
=================

In SNNGrow, spiking neurons are the basic units of spiking neural networks(SNNs). Unlike the neurons commonly found in deep learning, spiking neurons possess biomimetic neural dynamics and utilize discrete spikes as their output.

The number of neurons is automatically determined based on the ``shape`` of the first received input, after initialization or reinitialization by calling the ``reset()`` function. The code to reset the neuron's state can be found in :meth:`snngrow.base.utils` :

.. code-block:: python

    def reset(net: nn.Module):
    
      for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, BaseNode.BaseNode):
                logging.warning(f'Trying to call `reset()` of {m}, which is not snngrow.base.neuron'
                                f'.BaseNode')
            m.reset()


Thanks to neuronal dynamics, spiking neurons are stateful, which can also be said to have memory. Typically, the membrane potential of a spiking neuron serves as its state variable. The ``reset()`` function needs to be called to clear the previous state of the spiking neuron before feeding the next sample.
The dynamics equations are not the same for different neurons. But after the membrane potential exceeds the threshold voltage, the output spike, and after the output spike, the reset of the membrane potential is the same. SNNGrow neurons all inherit from :meth:`snngrow.base.neuron.BaseNode` , share the same fire and reset equations.
Any discrete spiking neuron can be described by three discrete equations (neuronal dynamics, fire, reset). The equations for neuronal dynamics and fire are as follows.

.. math::

  V[t]=f(V[t-1],X[t])

  S[t]=\Theta(V[t]-V_{threshold})

.. code-block:: python

    @abstractmethod
    def neuronal_dynamics(self, x: torch.Tensor):
        """
        The neuronal dynamics difference equation. The sub-class must implement this function.
        """

        raise NotImplementedError

    def neuronal_fire(self, x: torch.Tensor):
        """
        The neuronal fire difference equation.
        """

        if self.training:
            return self.surrogate_function(self.v - self.v_threshold)  
                  
        else:
            return (self.v >= self.v_threshold).to(x) 

Where :math:`X[t]` is an input, such as external input current; :math:`V[t]` is the membrane potential of the neuron after the output spike; :math:`f(V[t-1],X[t])` is the neuronal dynamics equation for the neuron state. The main difference is that the neuronal dynamics equation is different for different types of neurons; :math:`\Theta(x)` is the ``activation_function``. A commonly used activation function in this framework, extensively employed, is the step (Heaviside)  function. During forward propagation, if the input is greater than or equal to a threshold, it returns 1; otherwise, it returns 0. Such a ``tensor`` with only 0 or 1 elements is treated as a spike. The equation for the Heaviside function is as follows.

.. math::

  \Theta(x)=\left\{\begin{matrix}
                0, x\ge 0 \\
                1, x< 0
        \end{matrix}\right.

The output spike consumes the charge previously accumulated by the spiking neuron, resulting in an instantaneous decrease in membrane potential, namely, the reset of the membrane potential. In SNNGrow, the membrane potential is reset in 2 ways:

1. Hard mode, after the output spike, the membrane potential is directly reset to the reset voltage:

.. math:: V[t]=V[t](1-S[t])+V_{reset}S[t]

.. code-block:: python

  def hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
      v = (1. - spike) * v + spike * v_reset
      return v


2. Soft mode, after the output spike, the difference between the membrane potential and the threshold voltage is used as the reset voltage:

.. math:: V[t]=V[t]-V_{threshold}S[t]

.. code-block:: python

  def soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

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

The Dirichlet function is :math:`+\infty` at 0. If you directly use the Dirichlet function for gradient descent, it will make the training of the network extremely unstable. Therefore, we use surrogate gradient during backpropagation  [1]_.

The principle of the Surrogate Gradient method is that during forward propagation, :math:`\Theta(x)` is used, while during backpropagation, :math:`\frac{\mathrm{d} y}{\mathrm{d} x} =\sigma ^{'} (x)` is used, where :math:`\sigma (x)` is the surrogate function. :math:`\sigma (x)` is usually a function similar in shape to :math:`\Theta(x)` , but is smooth and continuous. Surrogate functions are used in neurons to generate an approximate gradient for spikes.

SNNGrow in :meth:`snngrow.base.surrogate.BaseFunction` implements the surrogate function in the base class, and provides an alternative for some commonly used functions, The surrogate function can be specified as an argument to the neuron constructor,  ``surrogate_function`` .

..  [1] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.

====================
Spiking Computation Mode
====================

The spiking computation mode is the core of SNNGrow's low-power implementation. In this mode, the output of spiking neurons is spike-based, and a custom SpikeTensor is used to encapsulate the neuron outputs. SpikeTensor is a tensor containing the outputs of spiking neurons, inheriting from PyTorch's Tensor. However, it uses a low-precision (1 Byte) data type for storage, where 1 represents a spike and 0 represents no spike. In spiking computation mode, SNNGrow leverages Cutlass to develop basic operations for SpikeTensor with mixed data types (such as GEMM), replacing high-power-consuming multiply-add operations with low-power addition operations.

The spiking computation mode does not need to be explicitly activated; it only requires specifying the spike_out parameter when constructing neurons.

For example, to define a simple LIF neuron:

.. code-block:: python

  surrogate = Sigmoid.Sigmoid(spike_out=True)
  # input is a Tensor, output is a SpikeTensor
  LIFNode(T=T, spike_out=True, surrogate_function=surrogate)

At this point, the output of the spiking neuron is a SpikeTensor. During the forward propagation process, the SpikeTensor will automatically propagate to the next layer of neurons, enabling the training and execution of the spiking neural network. SNNGrow has implemented a series of high-level operators for SpikeTensor, as seen in :mod:`snngrow.base.nn`  .

For example, to define a fully connected layer:

.. code-block:: python

  import snngrow.base.nn as snngrow_nn
  # input is a SpikeTensor, output is a Tensor
  snngrow_nn.Linear(512, 512, spike_in=True)

More optimized operators are still under development, so stay tuned.

====================
STDP Learning
====================

Snngrow provides STDP(Spike Timing Dependent Plasticity) learning rule, which can be used to learn the weights of fully connected layers.

STDP can be described using the following formula:

.. math::

  \begin{align}
  tr_{pre}[i][t] &= tr_{pre}[i][t] -\frac{tr_{pre}[i][t-1]}{\tau_{pre}} + s[i][t]\\
  tr_{post}[j][t] &= tr_{post}[j][t] -\frac{tr_{post}[j][t-1]}{\tau_{post}} + s[j][t]\\
  \Delta W[i][j][t] &= F_{post}(w[i][j][t]) \cdot tr_{pre}[i][t] \cdot s[j][t] - F_{pre}(w[i][j][t]) \cdot tr_{post}[j][t] \cdot s[i][t]
  \end{align}

Where  :math:`s[i][t]`  and :math:`s[j][t]` are the spike (0 or 1) from presynaptic neuron i and postsynaptic neuron j at time t. Trace  :math:`tr_{pre}[i][t]` and  :math:`tr_{post}[j][t]` recording the firing of presynaptic neuron i and postsynaptic neuron j at time t.  :math:`\tau_{post}` and  :math:`\tau_{post}` are the time constant of pre and post traces.  :math:`F_{pre}` and  :math:`F_{post}` are functions that control the amount of change in synaptic weights.

Snngrow directly updates the weights to implement STDP without backpropagation and additional optimizers.

Use  :meth:`snngrow.base.learning.STDP`  to build a fully connected spiking neural network for STDP learning:

.. code-block:: python

  import torch
  import torch.nn as nn
  import snngrow.base.nn as tnn
  from snngrow.base.neuron.IFNode import IFNode
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

Generate input spike, initialize the weight of the network to 0.4, the STDP is learned in T time steps, record the changes of the spike, trace and weight:

.. code-block:: python

    N_in, N_out = 4, 3
    T = 100
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

    out_spike = torch.stack(out_spike)   # [T, batch_size, N_out]
    trace_pre = torch.stack(trace_pre)   # [T, batch_size, N_in]
    trace_post = torch.stack(trace_post) # [T, batch_size, N_out]
    weight = torch.stack(weight)         # [T, N_out, N_in]

Visualize the dynamics of the first synaptic connection in the network:

.. image:: ../_static/test_stdp.*
    :width: 100%

The complete code is in ``snngrow/examples/test_stdp.py``.