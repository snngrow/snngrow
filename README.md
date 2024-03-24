# SNNGrow

<p align="center">
  	<img alt="SNNGrow" src="./docs/source/_static/logo.png" width=50%>
</p> 
SNNGrow是一款低能耗大规模脉冲神经网络训练和运行框架，在保持低能耗的同时，提供大规模SNN优异的学习能力，从而以高效率模拟生物体的认知大脑。

SNNGrow的愿景是解码人类智能及其进化机制，并为未来人与人工智能共生社会中研制受脑启发的的智能体提供支持。

- **[文档](https://snngrow.readthedocs.io/)**
- **[代码](https://github.com/snngrow/snngrow/)**

## 安装

在终端中运行以下命令来安装项目：
从PyPI上安装最新版本：
```
pip install snngrow
```
从GitHub上安装：
```
git clone https://github.com/snngrow/snngrow.git
cd snngrow
python setup.py install
```

Run the following command in your terminal to install the project:
Install the latest version from PyPI:
```
pip install snngrow
```
Install from GitHub:
```
git clone https://github.com/snngrow/snngrow.git
cd snngrow
python setup.py install
```

## 快速上手

SNNGrow的代码风格和Pytorch保持一致
您可以使用简单的代码构建脉冲神经网络
```
from snngrow.base.neuron import LIFNode
from snngrow.base.neuron import IFNode
import torch

lifnode = LIFNode.LIFNode()
ifnode = IFNode.IFNode()

x = torch.randn(2, 3, 4, 5)
y = torch.randn(6, 7, 8, 9)

x_lif = lifnode(x)
y_lif = lifnode(y)
```

The code style of SNNGrow is consistent with Pytorch
You can build spiking neural networks with simple code:
```
from snngrow.base.neuron import LIFNode
from snngrow.base.neuron import IFNode
import torch

lifnode = LIFNode.LIFNode()
ifnode = IFNode.IFNode()

x = torch.randn(2, 3, 4, 5)
y = torch.randn(6, 7, 8, 9)

x_lif = lifnode(x)
y_lif = lifnode(y)
```

## 开发计划

SNNGrow仍在持续开发中：
- [x] 深度脉冲神经网络支持
- [ ] 超低能耗稀疏脉冲神经网络计算
- [ ] 脑启发学习算法支持
- [ ] 仿生结构支持

## 引用

如果您在自己工作中使用了SNNGrow，请您考虑用如下格式引用：
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
## 项目信息

北京市优智创芯有限公司和北京理工大学[AETAS实验室](https://www.aetasbit.com/)是本项目的主要开发者。
