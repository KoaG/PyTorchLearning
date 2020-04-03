from __future__ import print_function
import torch

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
second_point = points[1]
#* Both tensor access same storage(memory)
#* Any change is reflected on both tensors
second_point[0] = 10.0
print(points)

#* To modify only one tensor, we can use clone
second_point = points[1].clone()
second_point[0] = 2.0
print(points)

#* To Transpose a tensor
points_t = points.t()
print(points)
print(points_t)

#* To check if tensors share same storage
print(id(points.storage()) == id(points_t.storage())) 
print(points.stride())
print(points_t.stride())