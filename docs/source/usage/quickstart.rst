Quickstart
----------

SNNGrow的代码风格和Pytorch保持一致
您可以使用简单的代码构建脉冲神经网络::

    from snngrow.base.neuron import LIFNode
    from snngrow.base.neuron import IFNode
    import torch

    lifnode = LIFNode.LIFNode()
    ifnode = IFNode.IFNode()

    x = torch.randn(2, 3, 4, 5)
    y = torch.randn(6, 7, 8, 9)

    x_lif = lifnode(x)
    y_lif = lifnode(y)


The code style of SNNGrow is consistent with Pytorch
You can build spiking neural networks with simple code::

    from snngrow.base.neuron import LIFNode
    from snngrow.base.neuron import IFNode
    import torch

    lifnode = LIFNode.LIFNode()
    ifnode = IFNode.IFNode()

    x = torch.randn(2, 3, 4, 5)
    y = torch.randn(6, 7, 8, 9)

    x_lif = lifnode(x)
    y_lif = lifnode(y)