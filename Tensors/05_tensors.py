from __future__ import print_function
import torch

x = torch.ones(3,3, dtype=torch.double)
x_bin = torch.tensor([2,2], dtype=torch.short)

print(x_bin.dtype)

x_d = torch.zeros(3,5,5).double()
x_d = torch.zeros(3,5,5).to(torch.double)

x = torch.randn(3,3)
short_x = x.type(torch.short)
print(x.dtype)
print(short_x.dtype)

#* Indexing a tensor
#* Same as that for python lists
x_list = [i for i in range(10)]
print(x_list[:])
print(x_list[5:])
print(x_list[:7])
print(x_list[3:6])
print(x_list[::2])
print(x_list[:-1])
#* negative step not supported for tensors
print(x_list[::-1])
print(x_list[9:4:-2])

x_tensor = torch.Tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
print(x_tensor[:])
print(x_tensor[:-1])
print(x_tensor[1:])
#* We can use range indexing
print(x_tensor[1:, :]) #* All rows after first, all col
print(x_tensor[1:, 0]) #* All rows after first, first col
print(x_tensor[:, 0])  #* All rows, first col