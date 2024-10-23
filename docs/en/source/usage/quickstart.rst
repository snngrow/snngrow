Quickstart
----------

The code style of SNNGrow is consistent with Pytorch, allowing you to build spiking neural networks with simple code::

    from snngrow.base import utils
    from snngrow.base.neuron import LIFNode
    import torch

    x = torch.randn(2, 3, 5, 5)

    net = torch.nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        LIFNode(),
        nn.Flatten(),
        nn.Linear(54, 1)
    )

    y = net(x)
    utils.reset(net)

An example of building a network using spiking computation mode::

    import torch
    import torch.nn as nn
    from snngrow.base.neuron.LIFNode import LIFNode
    from snngrow.base.surrogate import Sigmoid
    import snngrow.base.nn as snngrow_nn
    class SimpleNet(nn.Module):
        def __init__(self, T):
            super(SimpleNet, self).__init__()
            self.T = T
            self.surrogate = Sigmoid.Sigmoid(spike_out=True)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 512),
                LIFNode(T=T, spike_out=True, surrogate_function=self.surrogate),
                snngrow_nn.Linear(512, 512, spike_in=True),
                LIFNode(T=T, spike_out=True, surrogate_function=self.surrogate),
                snngrow_nn.Linear(512, 128, spike_in=True),
                nn.Linear(128, 10)
            )
