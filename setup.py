# Copyright 2024 Beijing Institute of Technology AETAS Lab. and Utarn Technology Co., Ltd. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import re
import sys
import platform
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, CUDA_HOME
from torch.__config__ import parallel_info

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'snngrow', '__init__.py'), 'r') as f:
  init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# requirements
with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

# long_description
with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# pybind_fn
if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    pybind_fn = f"snngrow/snngrow_backend/pybind_gemm_cuda.cu"
else:
    device = "cpu"
    # pybind_fn = f"pybind_{device}.cpp"
    pybind_fn = None

sources = [os.path.join(pybind_fn) if pybind_fn is not None else None]

include_dirs=['snngrow/snngrow_backend/cutlass_extension/include', 'snngrow/snngrow_backend/spikegemm']

for fpath in glob.glob(os.path.join("snngrow", "snngrow_backend", "torch_gemm", "*")):
    if (fpath.endswith(".cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith(".cu") and device == "cuda"
    ):
        sources.append(fpath)

extension_type = CUDAExtension if device == "cuda" else CppExtension

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3"],
}

# Not on Windows
if not os.name == 'nt':
    extra_compile_args['cxx'] += ['-Wno-sign-compare']

# OpenMP
info = parallel_info()
if ('backend: OpenMP' in info and 'OpenMP not found' not in info
        and sys.platform != 'darwin'):
    extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
    if sys.platform == 'win32':
        extra_compile_args['cxx'] += ['/openmp']
    else:
        extra_compile_args['cxx'] += ['-fopenmp']
else:
    print('Compiling without OpenMP...')

# Compile for mac arm64
if sys.platform == 'darwin':
    extra_compile_args['cxx'] += ['-D_LIBCPP_DISABLE_AVAILABILITY']
    if platform.machine == 'arm64':
       extra_compile_args['cxx'] += ['-arch', 'arm64']

setup(
    name="snngrow",
    version=version,
    author="AETAS",
    author_email="leiyunlin@bit.edu.cn, gaolanyu@bit.edu.cn",
    description="Third-generation Artificial Intelligence SNN Universal Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snngrow/snngrow", 
    install_requires=install_requires,
    python_requires='>=3.6,<=3.10',
    packages=find_packages(),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        "Programming Language :: Python :: 3 :: Only",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    ext_modules=[
            extension_type(
                name ='snngrow_backend', 
                sources = sources, 
                include_dirs = include_dirs, 
                extra_compile_args=extra_compile_args
            )
        ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)