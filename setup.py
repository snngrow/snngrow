# Copyright 2024 Utarn Technology Co., Ltd.
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

import os
import re
import sys
from setuptools import setup, find_packages

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

setup(
    name="snngrow",
    version=version,
    author="AETAS",
    author_email="leiyunlin@bit.edu.cn, gaolanyu@bit.edu.cn",
    description="Third-generation Artificial Intelligence SNN Universal Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BIT-AETAS/snngrow", 
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
)