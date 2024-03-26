# SNNGrow

<p align="center">
  	<img alt="SNNGrow" src="./docs/source/_static/logo.png" width=50%>
</p>

English | [中文(Chinese)](./README_zh_CN.md) 

SNNGrow is a low-power, large-scale spiking neural network training and inference framework. It preserves minimal energy cosumption while providing the superior learning abilities of large spiking neural network.

The vision of SNNGrow is to decode human intelligence and the mechanisms of its evolution, and to provide support for the development of brain-inspired intelligent agents in a future society where humans coexist with artificial intelligence.

- **[Documentation](https://snngrow.readthedocs.io/)**
- **[Source](https://github.com/snngrow/snngrow/)**

## Install

SNNGrow offers two installation methods.
Running the following command in your terminal will install the project:
### Install from PyPI:

```
pip install snngrow
```
### Install from GitHub:

1.  Download or clone SNNGrow from github
```
git clone https://github.com/snngrow/snngrow.git
```
2.  Enter the folder of SNNGrow and install braincog locally with setuptools
```
cd snngrow
python setup.py install
```

## Quickstart

The code style of SNNGrow is consistent with Pytorch, allowing you to build spiking neural networks with simple code:
```
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
```

## Development plans

SNNGrow is still under active development:
- [x] Large-scale deep spiking neural network training and inference
- [ ] Ultra-low energy consumption sparse spiking neural network computing
- [ ] Brain-inspired learning algorithm support
- [ ] Bionic neural network sparse structure support

## Cite

If you are using SNNGrow, please consider citing it as follows:
```
@misc{SNNGrow,
    title = {SNNGrow},
    author = {Lei, Yunlin and Gao, Lanyu and Yang, Xu and other contributors},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/snngrow/snngrow}},
}
```

## About

[Utarn Technology Co., Ltd.](https://www.utarn.com/w/home)and [Beijing Institute of Technology AETAS Laboratory](https://www.aetasbit.com/)are the main developers of SNNGrow.
