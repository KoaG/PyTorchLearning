import torch
from matplotlib import pyplot as plt

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

def model(t_u, w, b):
	return w*t_u + b  #* Linear fit model

def loss_fn(t_p, t_c):
	squared_diffs = (t_p - t_c)**2 #* loss/cost function
	return squared_diffs.mean()  #* mean square loss

#? with grad PyTorch keeps track of the parameters
params = torch.tensor([1.0, 0.0], requires_grad=True)

loss = loss_fn(model(t_u, *params), t_c) #* Loss
loss.backward()  #* calculates loss gradient
#! Calling backwards leads to gradient getting accumulated
#! on leaf nodes(nodes where gradient is calculated)
#! params.grad += d(loss)/d(params)
#! Therefore, we need to clear grad before calling it again
print(params.grad)
#* Explicitly making grad 0
if params.grad is not None:
	params.grad.zero_()
print(params.grad)

#* Training loop using backward
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
	for epoch in range(1, n_epochs + 1):
		if params.grad is not None:
			params.grad.zero_()
		t_p = model(t_u, *params)
		loss = loss_fn(t_p, t_c)
		loss.backward()
		#* .detach().require_grad_()
		#* p1 = p0 * lr * p0.grad 
		#* p2 = p1 * lr * p1.grad
		#* p1 graph goes back to p0, with new iterations pn -> pn-1 ->...-> p0
		#* we need to keep p0 in memory, and where do we need to assign errors??
		#* .detach() removes p1 from p0, .requires_grad_() enables gradient
		params = (params - learning_rate * params.grad).detach().requires_grad_()
		if epoch % 500 == 0:
			print('Epoch %d, Loss %f' % (epoch, float(loss)))
	return params

t_un = 0.1*t_u
params = training_loop(
			n_epochs = 5000,
			learning_rate = 1e-2,
			params = torch.tensor([1.0, 0.0], requires_grad=True),
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