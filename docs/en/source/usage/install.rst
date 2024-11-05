Install
-------

SNNGrow offers two installation methods.
Running the following command in your terminal will install the project:

**Install from Local(Recommended)**:

1. Install PyTorch

2. Install CUDA locally, make sure the CUDA version is consistent with the Pytorch CUDA version

3. Download or clone SNNGrow from github::

    git clone https://github.com/snngrow/snngrow.git

4. Enter the folder of SNNGrow and install braincog locally with setuptools::

    cd snngrow
    python setup.py install

**Install Lite Version from PyPI(The Lite version does not support spiking computation mode)**::

    pip install snngrow
