import torch
import numpy as np
import matplotlib.pyplot as plt

# generate random data
x = np.linspace(0, 100, 100)
y = x ** 2 + np.random.randn(x.shape[0])*500

plt.plot(x,y)

# create torch.Tensors from data
X = torch.from_numpy(x)
Y = torch.from_numpy(y)

# create parameters which will be optimized
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

# MSEloss function
loss_func = lambda pred, label: (pred - label).pow(2).mean()

epochs = 1000

for it in range(epochs):
    y_hat = w * X + b
    loss = loss_func(y_hat, Y)
    loss.backward()
    w.data = w - 1e-5 * w.grad.data
    b.data = b - 1e-5 * b.grad.data
    
    w.grad.data.zero_()
    b.grad.data.zero_()

# preds would be created with requires_grad=True
# then it had to be detached in order to convert to numpy
# since we don't need gradients, we disable them
with torch.no_grad():
    preds = w * X + b
plt.plot(x, preds.numpy(), '.')

# quadratic function with optimizer
w1 = torch.rand(1, requires_grad=True)
w2 = torch.rand(1, requires_grad=True)
b1 = torch.rand(1, requires_grad=True)

quad_optim = torch.optim.SGD([w1, w2, b1], lr=1e-10)

def quadratic_prediction():
    return w1 * X ** 2 + w2 * X + b1

for it in range(epochs):
    y_hat = quadratic_prediction()
    loss = loss_func(y_hat, Y)
    loss.backward()
    quad_optim.step()
    quad_optim.zero_grad()

with torch.no_grad():
    preds = quadratic_prediction()
plt.plot(x, preds.numpy(), 'r-')

plt.show()