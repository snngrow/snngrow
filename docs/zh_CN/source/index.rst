.. snngrow documentation master file, created by
   sphinx-quickstart on Mon Mar 18 21:11:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到SNNGrow的文档！
------------------------------------

`English <https://snngrow.readthedocs.io/en/>`_ | 简体中文

SNNGrow是一款低能耗大规模脉冲神经网络训练和运行框架，不依赖专门设计的硬件实现底层的脉冲计算模式，在此模式下，本项目利用脉冲特性，使用Cutlass针对脉冲数据开发GEMM等基本运算操作，将高功耗的乘加运算替换成低功耗的加法运算，同时进一步降低存储和带宽开销，带来数倍的提速和存储节省。SNNGrow在保持低能耗的同时，提供大规模SNN优异的学习能力，从而以高效率模拟生物体的认知大脑。

SNNGrow的愿景是解码人类智能及其进化机制，并为未来人与 人工智能共生社会中研制受脑启发的的智能体提供支持。

.. toctree::
   :maxdepth: 2
   :caption: 使用说明:

   /usage/install
   /usage/quickstart
   /usage/tutorials
   /usage/examples
   /usage/cite
   /usage/license


.. toctree::
   :maxdepth: 2
   :caption: APIS:

   /apis/neurons
   /apis/surrogate
   /apis/utils
   /apis/nn


索引和列表
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
