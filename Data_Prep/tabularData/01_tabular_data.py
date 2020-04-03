import torch
import csv
import numpy as np

wine_path = "./winequality-white.csv"
#* Reading 2D array, with dtype float
wineq_np = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
print(wineq_np)
#* Col names/identifier and array size
col_list = next(csv.reader(open(wine_path), delimiter=";"))
print(wineq_np.shape, col_list)

#* convert numpy array to tensor
wineq = torch.from_numpy(wineq_np)
print(wineq.shape,wineq.type())

#* We need to remove score/quality from tensor into data and result or target
data = wineq[:, :-1]  #* All rows and columns except last, i.e quality
print(data)
print(data.shape)
target = wineq[:, -1] #* All rows and last col only
print(target)
print(target.shape)

#? If we need to transform target in a tensor of labels
#? we have two options, treat them like int or use one-hot encoding
#? Both have different benefits and use case
target = target.long()
print(target)
#* One-hot encoding
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
print(target_onehot)

#* Unsqueezed
#? PyTorch allows class indices as targets while training nn
#? But if we want to use target as categorical input we need to use one-hot
target_unsq = target.unsqueeze(1)
print(target_unsq)

#* Calculate mean and var to normalize data
data_mean = torch.mean(data, dim=0)  #? dim=0 -> reduction along dimension 0
data_var = torch.var(data, dim=0)
data_norm = (data - data_mean)/torch.sqrt(data_var)
print(data_norm)

#* Now we can separate data according to target using PyTorch advance indexing
bad_indexes = torch.le(target, 3)  #* indexes with quality less than equal to 3
print(bad_indexes)
print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())
bad_data = data[bad_indexes]
print(bad_data)
print(bad_data.shape)

#* With advance indexing
bad_data = data[torch.le(target, 3)]
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
	print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

#* From output we can see bad_data has high mean sulfur dioxide
#* we can use it as a threshold
total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6] #* Only col 6
#? Predicting good wine using just sulfur dioxide threshold
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())
#* Check good wine
actual_indexes = torch.gt(target, 5)
print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())
#* Difference between predicted and actual
print(actual_indexes.sum() - predicted_indexes.sum())
#* Checking indexes of predicted wine against actual quality
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

print('%*s %*s %*s'%(20,"Matches", 20,"Predicted", 20,"Actual"))
print('%*i %*i %*i'%(20,n_matches, 20,n_predicted, 20,n_actual))
print('%s %*.5f %*.5f'%('Percentage', 30, n_matches/n_predicted, 20, n_matches/n_actual))