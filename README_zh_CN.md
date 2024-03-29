# SNNGrow

<p align="center">
  	<img alt="SNNGrow" src="./docs/en/source/_static/logo.png" width=50%>
</p> 

[English](./README.md) | 中文(Chinese)

SNNGrow是一款低能耗大规模脉冲神经网络训练和运行框架，在保持低能耗的同时，提供大规模SNN优异的学习能力，从而以高效率模拟生物体的认知大脑。

SNNGrow的愿景是解码人类智能及其进化机制，并为未来人与 人工智能共生社会中研制受脑启发的的智能体提供支持。

- **[文档](https://snngrow.readthedocs.io/)**
- **[代码](https://github.com/snngrow/snngrow/)**

## 安装
SNNGrow提供两种安装方式，在终端中分别运行以下命令都能安装项目：
### 一、从PyPI上安装最新版本：
```
pip install snngrow
```
### 二、从GitHub上安装：
1、克隆我们的github仓库：
```
git clone https://github.com/snngrow/snngrow.git
```
2、进入目录，使用setuptools本地安装：
```
cd snngrow
python setup.py install
```

## 快速上手

SNNGrow的代码风格和Pytorch保持一致
您可以使用简单的代码构建脉冲神经网络
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

[北京市优智创芯有限公司](https://www.utarn.com/w/home)和北京理工大学[AETAS实验室](https://www.aetasbit.com/)是本项目的主要开发者。
