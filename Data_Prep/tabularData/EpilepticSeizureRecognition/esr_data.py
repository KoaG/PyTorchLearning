import pandas as pd
import torch

data_path = "./dataset.csv"
esr_df = pd.read_csv(data_path)
esr_df = esr_df.drop(esr_df.columns[0], axis=1)
#print(esr_df.values)
esr_np = esr_df.values
#print(esr_np.shape)
col_list = list(esr_df.columns)
#print(col_list)

esrT = torch.from_numpy(esr_np)
esrT = esrT.float()
print(esrT.shape, esrT.type())

data = esrT[:, :-1]
#print(data)
print(data.shape)
target = esrT[:, -1]
target = target.long()
#print(target)
print(target.shape)

data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)
data_norm = (data - data_mean)/torch.sqrt(data_var)
#print(data_norm)

seizure_data = data[torch.eq(target, 1)]
eyes_op = data[torch.eq(target,5)]
eyes_cl = data[torch.eq(target,4)]
tumor_id = data[torch.eq(target,3)]
tumor_eeg = data[torch.eq(target,2)]

seizure_mean = torch.mean(seizure_data, dim=0)
eyes_op_mean = torch.mean(eyes_op, dim=0)
eyes_cl_mean = torch.mean(eyes_cl, dim=0)
tumor_id_mean = torch.mean(tumor_id, dim=0)
tumor_eeg_mean = torch.mean(tumor_eeg, dim=0)


print(seizure_data.shape)
for i, args in enumerate(zip(col_list, seizure_mean, tumor_eeg_mean, tumor_id_mean, eyes_cl_mean, eyes_op_mean)):
	print('{:8} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f}'.format(*args))