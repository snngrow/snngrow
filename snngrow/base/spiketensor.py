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

import torch
from torch.autograd import Function
from torch.utils import _pytree as pytree

__all__ = ["SpikeTensor"]

class from_dense(Function):
    @staticmethod
    def forward(ctx, dense_tensor, threshold=0):
        if dense_tensor.dtype is torch.bool:
            return SpikeTensor(dense_tensor)
        else:
            return SpikeTensor(dense_tensor >= threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class to_dense(Function):
    @staticmethod
    def forward(ctx, spike_tensor, dtype=torch.float32):
        return spike_tensor.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SpikeTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        assert elem.dtype is torch.bool, "SpikeTensor only supports boolean dtype"
        return torch.Tensor._make_wrapper_subclass(cls, elem.shape, dtype=torch.float32, device=elem.device, requires_grad=elem.requires_grad)
    
    def __init__(self, elem):
        self.elem = elem

    def __repr__(self):
        autograd_info = f", grad_fn={self.grad_fn}" if self.grad_fn else f", requires_grad=True" if self.requires_grad else ""
        return f"SpikeTensor({self.elem}, public_dtype={self.dtype}{autograd_info})"
    
    @classmethod
    def from_dense(cls, dense_tensor):
        return from_dense.apply(dense_tensor)
    
    def to_dense(self, dtype=torch.float32):
        return to_dense.apply(self.elem, dtype)
    
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.mm.default:
            print("mm is called")
            args, kwargs = pytree.tree_map_only(SpikeTensor, lambda x: x.to_dense(), (args, kwargs))
            return func(*args, **kwargs)
        elif func is torch.ops.aten.mul.Tensor:
            print("mul is called")
            args, kwargs = pytree.tree_map_only(SpikeTensor, lambda x: x.to_dense(), (args, kwargs))
            return func(*args, **kwargs)
        else:
            args, kwargs = pytree.tree_map_only(SpikeTensor, lambda x: x.elem, (args, kwargs))
            out = func(*args, **kwargs)
            out = pytree.tree_map_only(torch.Tensor, lambda x: SpikeTensor.from_dense(x), out)
            return out
        

def test():
    dense_tensor = torch.Tensor([[1,0,1], [1, 1, 1], [0, 0, 0]]).to(torch.float32)
    dense_tensor.requires_grad = True
    # spike_tensor = SpikeTensor(dense_tensor)
    spike_tensor = SpikeTensor.from_dense(dense_tensor)
    # spike_tensor = spike_tensor.to(device="cuda")
    spike_tensor = spike_tensor.reshape(9, 1)
    # ans_tensor = torch.matmul(spike_tensor, spike_tensor)


if __name__ == "__main__":
    test()
    
    
