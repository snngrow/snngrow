快速上手
----------

SNNGrow的代码风格和Pytorch保持一致，您可以使用简单的代码构建脉冲神经网络::

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
