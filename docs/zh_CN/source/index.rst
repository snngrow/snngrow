.. snngrow documentation master file, created by
   sphinx-quickstart on Mon Mar 18 21:11:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到SNNGrow的文档！
------------------------------------

`English <https://snngrow.readthedocs.io/en/>`_ | 简体中文

SNNGrow是一款低能耗大规模脉冲神经网络训练和运行框架，不依赖专门设计的硬件实现底层的脉冲计算模式，在此模式下，本项目利用脉冲特性，使用Cutlass针对脉冲数据开发GEMM等基本运算操作，将高功耗的乘加运算替换成低功耗的加法运算，同时进一步降低存储和带宽开销，带来数倍的提速和存储节省。SNNGrow在保持低能耗的同时，提供大规模SNN优异的学习能力，从而以高效率模拟生物体的认知大脑。

SNNGrow的愿景是解码人类智能及其进化机制，并为未来人与 人工智能共生社会中研制受脑启发的的智能体提供支持。

SNNGrow支持低功耗的脉冲稀疏计算，针对脉冲数据，自定义了一个SpikeTensor的数据结构，得益于脉冲的二值化特性，这个数据结构在底层使用低比特的存储，只需要1Byte来存储脉冲数据。同时，针对SpikeTensor，SNNGrow使用CUDA和CUTLASS定制低能耗的算子，如针对SpikeTensor的矩阵乘法，真正地实现从底层将乘法替换成加法。

可视化GPU上脉冲矩阵乘法和torch的矩阵乘法指令调用情况，在SNNGrow中，相比于torch，实现了完全使用加法运算来进行矩阵乘法，这将节省非常多的能耗，同时减少对存储的需求。

.. image:: _static/instruction.png

SNNGrow将会带来数倍的速度提升，我们实测了矩阵乘法的速度，和同规模的torch矩阵乘法相比，SNNGrow可以带来2倍以上的速度提升。

.. image:: _static/compute.png

得益于脉冲的数据形式，SNNGrow只要求更少的内存占用和带宽需求，这意味着在同样的硬件资源下，SNNGrow可以运行更大的模型。

.. image:: _static/memory.png

Snngrow中提供了STDP(Spike Timing Dependent Plasticity)学习规则，可以用于全连接层的权重学习。

.. image:: _static/test_stdp.png

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
   /apis/learning
   /apis/utils
   /apis/nn


索引和列表
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
