import torch
from matplotlib import pyplot as plt
import torch.optim as optim

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])


n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print(train_indices, val_indices)

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1*train_t_u
val_t_un = 0.1*val_t_u

def model(t_u, w, b):
	return w*t_u + b  #* Linear fit model

def loss_fn(t_p, t_c):
	squared_diffs = (t_p - t_c)**2 #* loss/cost function
	return squared_diffs.mean()  #* mean square loss

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
	for epoch in range(1, n_epochs + 1):
		#* Calculating outputs and loss on training set
		train_t_p = model(train_t_u, *params)
		train_loss = loss_fn(train_t_p, train_t_c)
		#* Calculating outputs and loss on validation set 
		with torch.no_grad() :
			val_t_p = model(val_t_u, *params)
			val_loss = loss_fn(val_t_p, val_t_c)
			assert val_loss.requires_grad == False
			#? We don't want to store gradient calculated due to validation calculations
		optimizer.zero_grad()  #* clearing gradients accumulated 
		train_loss.backward()  #* calculating gradients
		optimizer.step()  #* Updating parameters
		if epoch % 100 == 0:
			print('Epoch {:6}, Training loss {:2.3f}, Validation loss {:2.3f}'.format(epoch, float(train_loss), float(val_loss)))
	return params

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
print(training_loop(
	n_epochs = 5000,
 	optimizer = optimizer,
 	params = params,
 	train_t_u = train_t_un,
 	val_t_u = val_t_un,
 	train_t_c = train_t_c,
 	val_t_c = val_t_c))