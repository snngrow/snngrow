安装
-------

SNNGrow提供两种安装方式，在终端中分别运行以下命令都能安装项目:

**一、从GitHub上安装（推荐）**:

1、安装依赖PyTorch

2、本地安装CUDA，请确保CUDA版本和Pytorch的CUDA版本一致

3、克隆我们的github仓库::

    git clone https://github.com/snngrow/snngrow.git

4、进入目录，使用setuptools本地安装::

    cd snngrow
    python setup.py install

**二、从PyPI上安装精简版，精简版不支持脉冲计算模式**::

    pip install snngrow


