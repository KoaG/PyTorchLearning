from __future__ import print_function
import torch

points = torch.tensor([[1.0,2.0], [3.0,4.0], [5.0,6.0]])
points_gpu = points.to(device='cuda')
x = torch.ones(3,2, device='cuda')

print(points)
print(points_gpu)
print(x)

points = 2*points #* operation on cpu
points_gpu = 2*points_gpu #* operation on gpu

points_gpu = x + points_gpu
points_cpu = points_gpu.to(device='cpu')
print(points_cpu)
points_cpu = 2*points_cpu
points_gpu = points_cpu.cuda() #* default cuda device 0
points_gpu = points_cpu.cuda(0) #* cuda device 0
