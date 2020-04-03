from __future__ import print_function
import torch

'''
CoorD = [[1.0,4.0], [2.0,1.0], [3.0,5.0]]
points = torch.FloatTensor(CoorD)
print(points)

second_point = points[1]
print(second_point)

print("Points \t Second Point")
print(points.storage_offset(),"\t",second_point.storage_offset())
print(points.size(),"\t",second_point.size())
print(points.shape,"\t",second_point.shape)
print(points.stride(),"\t",second_point.stride())
'''

Train2 = [ [[0.0, 0.0], [1.0,2.0], [4.0,5.0]],
			[[10.0, 10.0], [15.0, 10.0], [15.0, 20.0]]]

ten3d = torch.FloatTensor(Train2)
train2 = ten3d[1]
t2p3 = train2[2]

print(ten3d)
print(train2)
print(t2p3)

print(ten3d.storage().data_ptr() == train2.storage().data_ptr())
print(ten3d.storage())
print(train2.storage())

print("First \t Second \t Third")
print(ten3d.storage_offset(),"\t",train2.storage_offset(),"\t",t2p3.storage_offset())
print(ten3d.stride(),"\t",train2.stride(),"\t",t2p3.stride())
print(ten3d.size(),"\t",train2.size(),"\t",t2p3.size())
print(ten3d.shape,"\t",train2.shape,"\t",t2p3.shape)

a = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
b = torch.tensor([[7.0,8.0],[9.0,10.0],[11.0,12.0]])
c = torch.cat([a,b])
print(c)
print(c.stride())
print(c.storage_offset())
print(a.storage_offset())
print(b.storage_offset())
print(a.data_ptr() == b.data_ptr())
print(a.storage().data_ptr() == c.storage().data_ptr())

a[0,0] = 2.0
print(a.storage())
print(c.storage())