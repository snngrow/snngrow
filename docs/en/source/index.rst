.. snngrow documentation master file, created by
   sphinx-quickstart on Mon Mar 18 21:11:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SNNGrow's documentation!
------------------------------------

English | `简体中文 <https://snngrow.readthedocs.io/zh-cn/>`_

SNNGrow is a low-power, large-scale spiking neural network training and inference framework. SNNGrow does not rely on specialized hardware to implement an energy-efficient spiking computation mode. Using Cutlass, SNNGrow develops fundamental operations for spiking data (such as GEMM), replacing high-power-consuming multiply-add(MAD) operations with low-power addition(ADD) operations. Additionally, SNNGrow further reduces storage and bandwidth costs by utilizing the binary nature of spikes, resulting in several times speedup and storage savings. It preserves minimal energy cosumption while providing the superior learning abilities of large spiking neural network.

The vision of SNNGrow is to decode human intelligence and the mechanisms of its evolution, and to provide support for the development of brain-inspired intelligent agents in a future society where humans coexist with artificial intelligence.

SNNGrow supports low-power sparse computation. It defines a SpikeTensor data structure for spike data. Thanks to the binarization property of spike, this data structure uses low bit storage at the underlying level, and only needs 1Byte to store the spike data. Meanwhile, for SpikeTensor, SNNGrow uses CUDA and CUTLASS to customize low-energy operators, such as matrix multiplication for SpikeTensor, to really replace multiplication with addition from the bottom layer.

Visualizing the matrix multiplication instruction call of spike matrix multiplication and torch on GPU, in SNNGrow, compared with torch, the matrix multiplication is realized by completely using addition operation, which will save a lot of energy consumption and reduce the storage requirements.

.. image:: _static/instruction.png

SNNGrow will bring several times the speed up. We measured the speed of matrix multiplication. Compared to the same scale torch matrix multiplication, SNNGrow can bring more than 2 times the speed up.

.. image:: _static/compute.png

Benefiting from the data form of pulses, SNNGrow only requires less memory footprint and bandwidth requirements, which means that SNNGrow can run larger models with the same hardware resources.

.. image:: _static/memory.png

.. toctree::
   :maxdepth: 2
   :caption: USAGE:

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


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
