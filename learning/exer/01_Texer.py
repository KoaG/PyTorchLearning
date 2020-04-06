import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from num2words import num2words

data_path = './data.csv'
#* read inputs and outputs
data = np.loadtxt(data_path, dtype=np.float32, delimiter=';', skiprows=1)
print(data.shape)
#print(data[:,0])
x = data[:,0]
y = data[:,1]
'''
fig = plt.figure()
plt.xlabel('Year')
plt.ylabel('Population')
plt.plot(x,y,'o')
plt.show()
'''
t_x = torch.from_numpy(x)
t_y = torch.from_numpy(y)
print(t_x.shape, t_y.shape)

#* Different models definations
def modelLinear(t_x, w, b):
	return w*t_x + b
def modelQuad(t_w, w2, w1, b):
	return w2*t_x**2 + w1 + b
def modelExp(t_x, w, b):
	return w**t_x + b

#* Loss function
def loss_fn(t_p, t_y):
	diff = (t_p - t_y)**2
	return diff.mean()

#* Paramters and optimizers
paramsL = torch.tensor([1.0, 0.0], requires_grad=True)
paramsQ = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
lr = 1e-2
optimizerL = optim.Adam([paramsL], lr=lr)
optimizerQ = optim.Adam([paramsQ], lr=lr)

#* Training Loop
def training_loop(n_epochs, model, optimizer, params, t_x, t_y):
	for epoch in range(1, n_epochs + 1):
		t_p = model(t_x, *params)
		loss = loss_fn(t_p, t_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch < 11 or epoch%1000 == 0:
			print('Epoch {:5}, Loss {:3.4f}'.format(epoch, float(loss)))
	return params

#* Normalizing inputs and outputs
x_min = torch.min(t_x)
x_max = torch.max(t_x)
y_min = torch.min(t_y)
y_max = torch.max(t_y)
print(x_min, x_max)
print(y_min, y_max)
t_xn = (t_x - x_min)/(x_max - x_min)
t_yn = (t_y - y_min)/(y_max - y_min)
params = training_loop(
	5000,
	modelLinear,
	optimizerL,
	paramsL,
	t_xn,
	t_yn
)
print(params)
t_p = modelLinear(t_xn, *params)
t_p = t_p*(y_max - y_min) + y_min
fig = plt.figure()
plt.xlabel("Year")
plt.ylabel("Population")
plt.plot(t_x.numpy(), t_p.detach().numpy(), 'x')
plt.plot(t_x.numpy(), t_y.numpy(), 'o')
plt.show()
p_x = torch.tensor([2017.0, 2018.0, 2019.0, 2020.0])
p_x = torch.cat((t_x, p_x), dim=0)
p_xn = (p_x - torch.min(p_x))/(torch.max(p_x) - torch.min(p_x))
#print(p_x)
t_p = modelLinear(p_xn, *params)
t_p = t_p*(y_max - y_min) + y_min 
fig = plt.figure()
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Population")
plt.plot(p_x.numpy(), t_p.detach().numpy(), 'x')
plt.plot(t_x.numpy(), t_y.numpy(), 'o')
print("Predicted Population :")
for i,args in enumerate(zip(p_x[-4:],t_p[-4:])):	
	print('{} : {} \n\t {}'.format(*args, num2words(int(args[1]))))
plt.show()

params = training_loop(
	5000,
	modelExp,
	optimizerL,
	paramsL,
	t_xn,
	t_yn
)
print(params)
t_p = modelExp(t_xn, *params)
t_p = t_p*(y_max - y_min) + y_min
fig = plt.figure()
plt.xlabel("Year")
plt.ylabel("Population")
plt.plot(t_x.numpy(), t_p.detach().numpy(), 'x')
plt.plot(t_x.numpy(), t_y.numpy(), 'o')
plt.show()
p_x = torch.tensor([2017.0, 2018.0, 2019.0, 2020.0])
p_x = torch.cat((t_x, p_x), dim=0)
p_xn = (p_x - torch.min(p_x))/(torch.max(p_x) - torch.min(p_x))
#print(p_x)
t_p = modelExp(p_xn, *params)
t_p = t_p*(y_max - y_min) + y_min
fig = plt.figure()
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Population")
plt.plot(p_x.numpy(), t_p.detach().numpy(), 'x')
plt.plot(t_x.numpy(), t_y.numpy(), 'o')
print("Predicted Population :")
for i,args in enumerate(zip(p_x[-4:],t_p[-4:])):	
	print('{} : {} \n\t {}'.format(*args, num2words(int(args[1]))))
plt.show()
