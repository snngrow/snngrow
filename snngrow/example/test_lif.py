from snngrow.base.neuron import LIFNode
from snngrow.base.neuron import IFNode
from snngrow.base.surrogate import Sigmoid
from snngrow.base.surrogate import ATan
from snngrow.base import utils
import torch

print(torch.__version__)

lifnode = LIFNode.LIFNode()
ifnode = IFNode.IFNode()

lifnode = LIFNode.LIFNode(surrogate_function=ATan.ATan())
ifnode = IFNode.IFNode(surrogate_function=ATan.ATan())

x = torch.randn(2, 3, 4, 5)
y = torch.randn(6, 7, 8, 9)

a = lifnode(x)
utils.reset(lifnode)
b = lifnode(y)