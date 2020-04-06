import torch
from matplotlib import pyplot as plt

#* Sample data converted to tensors
I_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
I_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.FloatTensor(I_c)
t_u = torch.FloatTensor(I_u)

def model(t_u, w, b):
	return w*t_u + b  #* Linear fit model

def loss_fn(t_p, t_c):
	squared_diffs = (t_p - t_c)**2 #* loss/cost function
	return squared_diffs.mean()  #* mean square loss

w = torch.ones(1)  #* Weight parameter
b = torch.zeros(1) #* Bias parameter

t_p = model(t_u, w, b)
print(t_p) #* Predicted outputs
loss = loss_fn(t_p, t_c)
print(loss) #* Error 

#* Gradients/ Derivatives
def dloss_fn(t_p, t_c):
	dsq_diffs = 2*(t_p - t_c)
	return dsq_diffs

def dmodel_dw(t_u, w, b):
	return t_u

def dmodel_db(t_u, w, b):
	return 1.0
#* Loss Gradient
def grad_fn(t_u, t_c, t_p, w, b):
	dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
	dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
	return torch.stack([dloss_dw.mean(), dloss_db.mean()])

#* Training loop/ epochs
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
	for epoch in range(1, n_epochs + 1):
		w, b = params
		t_p = model(t_u, w, b)  #* Calculating Outputs based on model
		loss = loss_fn(t_p, t_c)  #* Calculating loss
		grad = grad_fn(t_u, t_c, t_p, w, b) #* Loss gradiant
		params = params - learning_rate*grad #* Updating parameters
		print('Epoch %d, Loss %f'%(epoch, float(loss))) if epoch%1000 == 0 else 0
	return params
	

training_loop(
	n_epochs = 100,
	learning_rate = 1e-2,
	params = torch.tensor([1.0, 0.0]),
	t_u = t_u,
	t_c = t_c
)

training_loop(
	n_epochs = 100,
	learning_rate = 1e-4,
	params = torch.tensor([1.0, 0.0]),
	t_u = t_u,
	t_c = t_c
)

t_un = 0.1*t_u
training_loop(
	n_epochs = 100,
	learning_rate = 1e-2,
	params = torch.tensor([1.0, 0.0]),
	t_u = t_un,
	t_c = t_c
)
params = training_loop(
			n_epochs = 5000,
			learning_rate = 1e-2,
			params = torch.tensor([1.0, 0.0]),
			t_u = t_un,
			t_c = t_c)

print(params)
t_p = model(t_un, *params)
fig = plt.figure()
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()