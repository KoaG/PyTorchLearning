from __future__ import print_function
import torch
import numpy as np

#* From tensor to numpy
points = torch.ones(3,4)
points_np = points.numpy()
print(points_np)
points_np[1,1] = 2
print(points)

#* From numpy to tensor
x = np.zeros((3,4))
x_tensor = torch.from_numpy(x)
print(x_tensor)
x[1,1] = 2
print(x)

#* Saving and loading tensor
torch.save(points,'./data/points.t')
with open('./data/x_tensor.t','wb') as f:
	torch.save(x_tensor,f)

points = torch.load('./data/x_tensor.t')
print(points)
with open('./data/points.t','rb') as f:
	x_tensor = torch.load(f)
print(x_tensor)

#* Working with H5PY Files
#? Install h5py module using pip install h5py
import h5py
d1 = np.random.random((10,20))
d2 = np.random.random((2,2,3))
#* Creating sample h5py file
hf = h5py.File('./data/h5py.hd5','w')
hf.create_dataset('data1', data=d1)
hf.create_dataset('data2', data=d2)
hf.close()
#* Reading h5py file
hf = h5py.File('./data/h5py.hd5','r')
print(hf.keys())
#* Dataset
n1 = hf['data2']
print(n1)
print(type(n1))
#* Dataset to numpy array
n1 = n1[:]
print(n1)
print(type(n1))
hf.close()
#* Numpy array to tensor
points = torch.from_numpy(n1)
print(points)