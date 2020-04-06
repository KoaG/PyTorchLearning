import torch
from matplotlib import pyplot as plt
import torch.optim as optim

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

def model(t_u, w2, w1, b):
	return w2*t_u**2 + w1*t_u + b  #* quadratic fit model

def loss_fn(t_p, t_c):
	squared_diffs = (t_p - t_c)**2 #* loss/cost function
	return squared_diffs.mean()  #* mean square loss

params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
learning_rate = 1e-4
optimizer = optim.SGD([params], lr=learning_rate)

def training_loop(n_epochs, optimizer, params, t_u, t_c):
	for epoch in range(1, n_epochs + 1):
		t_p = model(t_u, *params)
		loss = loss_fn(t_p, t_c)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 1000 == 0:
			print('Epoch %d, Loss %f' % (epoch, float(loss)))
	return params

t_un = 0.1*t_u
params = training_loop(
		n_epochs = 10000,
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