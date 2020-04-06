import torch
import imageio
import os

img_arr = imageio.imread('./bobby.jpg')
print(img_arr.shape) #* W x H x C

img = torch.from_numpy(img_arr)
out = torch.transpose(img, 0, 2) #* C x H x W
print(out.shape)

#* N images cane loaded in a tensor of size N x C x H x W
batch_size = 100 #? N
#? N images, 3 channel, size 256x256, with pixel size 8bit 
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
data_dir = './cat_images/'
filenames = [ data_dir + name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']

for i, filename in enumerate(filenames):
	img_arr = imageio.imread(filename)[:,:,:3] #? Removing alpha channel
	batch[i] = torch.transpose(torch.from_numpy(img_arr), 0, 2)

print(batch[1])