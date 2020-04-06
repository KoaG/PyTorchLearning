import os
import imageio
import torch

image_dir = './images/'
filenames = [image_dir + name for name in os.listdir(image_dir) if os.path.splitext(name)[-1] == ".jpg"]

batch_size = len(filenames)
batch = torch.zeros(batch_size, 3, 640, 640, dtype=torch.uint8)

for i, filename in enumerate(filenames):
	img_arr = imageio.imread(filename)[:,:,:3]
	batch[i] = torch.transpose(torch.from_numpy(img_arr), 0, 2)

print(batch.shape)
batch_mean = torch.mean(batch.float(), dim=(2,3))
#print(batch_mean)
for i in range(batch_size):
	print('{:2} {:20} {:8.2f} {:8.2f} {:8.2f}'.format(
		i, filenames[i], float(batch_mean[i][0]), 
		float(batch_mean[i][1]), float(batch_mean[i][2])))

print(torch.mean(batch[0].float(),dim=(1,2))[0])