Quickstart
----------

The code style of SNNGrow is consistent with Pytorch, allowing you to build spiking neural networks with simple code::

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
