import torch
from matplotlib import pyplot as plt
import torch.optim as optim

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

def model(t_u, w, b):
	return w*t_u + b  #* Linear fit model

def loss_fn(t_p, t_c):
	squared_diffs = (t_p - t_c)**2 #* loss/cost function
	return squared_diffs.mean()  #* mean square loss

#? with grad PyTorch keeps track of the parameters
params = torch.tensor([1.0, 0.0], requires_grad=True)

#? For updating parameter we can use different strategies
#? with previous two codes we used vanilla gradient descent
#? PyTorch provides optim package with different
#? different optimization strategies
#print(dir(optim))
learning_rate = 1e-2
optimizer =optim.SGD([params], lr=learning_rate)

t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()
optimizer.step()
print(params)

def training_loop(n_epochs, optimizer, params, t_u, t_c):
	for epoch in range(1, n_epochs + 1):
		t_p = model(t_u, *params)  #* calculating outputs based on model
		loss = loss_fn(t_p, t_c)  #* calculting loss
		optimizer.zero_grad()  #* clearing gradient accumulated on leaf nodes
		loss.backward() #* putting gradients on leaf nodes
		optimizer.step()  #* updating parameters based on optimization method
		if epoch % 500 == 0:
			print('Epoch %d, Loss %f' % (epoch, float(loss)))
	return params

t_un = 0.1*t_u
params = training_loop(
		n_epochs = 5000,
 		optimizer = optimizer,
 		params = params,
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

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) 

params = training_loop(
		n_epochs = 2000,
		optimizer = optimizer,
		params = params,
		t_u = t_u,
		t_c = t_c)
print(params)
t_p = model(t_u, *params)
fig = plt.figure()
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()