from __future__ import print_function
import torch

a = torch.ones(3)	#? Creating a tensor with 3 ones in a column
print(a)			#? Printing tensor

print(a[1])			#? Accessing tensor's element
print(float(a[1]))

a[2] = 2.0			#! Changing element value
print(a)

points = torch.zeros(6)
points[0] = 1.0
points[1] = 4.0
points[2] = 2.0
points[3] = 1.0
points[4] = 3.0
points[5] = 5.0

#* We can also use python list to construct a tensor
points = torch.tensor([1.0,4.0,2.0,1.0,3.0,5.0])
print(points)

firstPoint = (float(points[0]), float(points[1]))
print(firstPoint)

#* We can create 2D tensor 
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
print(points)

print(points.shape)	#? Tensor shape

points = torch.zeros(3,2)	#? A 3x2 tensor
print(points)

points = torch.FloatTensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
print(points[0, 1])	#? Accessing tensor element
print(points[0])	#? First element which is a 1-D tensor

print(points.storage())	#? Accessing tensor storage
points_storage = points.storage()
print(points_storage[0])	#? Tensor are stored in continous chunks of memory
print(type(points_storage))

print(points.storage()[1])
points_storage[0] = 2.0	#? Updating storage updates the tensor it is referencing
print(points)