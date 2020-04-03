from __future__ import print_function
import torch

x = torch.ones(3, 4, 5)
x_t = x.transpose(0,2)	#* dimension 0 and 2 are swapped
print(x.shape)
print(x_t.shape)
x_t = x.transpose(1,2)
print(x_t.shape)

print(x.stride())
print(x_t.stride())

x = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
x_t = x.t()
#* On tranpose tensor storage access is not contiguous
print(x.is_contiguous())
print(x_t.is_contiguous())

#* To make tensor storage acces contiguous
print("Tensor x")
print(x)
print(x.storage())
print(x.stride())
print("Tensor x_t")
print(x_t)
print(x_t.storage())
print(x_t.stride())
x_t_cont = x_t.contiguous()
print("Tensor x_t_cont")
print(x_t_cont)
print(x_t_cont.storage())
print(x_t_cont.stride())

print(id(x_t.storage()) == id(x_t_cont.storage()))
print("Tensor x")
print(x)
print(x.storage())
print(x.is_contiguous())
print(id(x.storage()) == id(x_t_cont.storage()))

#* tensor Id and data ptr
print(id(x.storage()),id(x_t.storage()),id(x_t_cont.storage()))
print(x.storage().data_ptr(),x_t.storage().data_ptr(),x_t_cont.storage().data_ptr())