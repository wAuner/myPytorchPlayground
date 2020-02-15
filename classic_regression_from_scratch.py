import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial, reduce

PLOT_ALL = False

def prediction_plot(x_vals,y_vals,format, **kwargs):
    plt.plot(x,y)
    plt.plot(x_vals, y_vals, format, **kwargs)

# generate random data
x = np.linspace(0, 100, 100)
y = x ** 2 + np.random.randn(x.shape[0])*500

# create torch.Tensors from data
X = torch.from_numpy(x)
Y = torch.from_numpy(y)

# create parameters which will be optimized
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

# MSEloss function
mse_loss = lambda pred, label: (pred - label).pow(2).mean()

epochs = 500

# linear function
#########################################
for _ in range(epochs):
    y_hat = w * X + b
    loss = mse_loss(y_hat, Y)
    loss.backward()
    w.data = w - 1e-5 * w.grad.data
    b.data = b - 1e-5 * b.grad.data
    
    w.grad.data.zero_()
    b.grad.data.zero_()

# preds would be created with requires_grad=True
# then it had to be detached in order to convert to numpy
# since we don't need gradients, we disable them
with torch.no_grad():
    linear_preds = w * X + b

if PLOT_ALL:
    plt.figure()
    prediction_plot(x, linear_preds, '.')


# quadratic function with optimizer
#########################################
w1 = torch.rand(1, requires_grad=True)
w2 = torch.rand(1, requires_grad=True)
b1 = torch.rand(1, requires_grad=True)

quad_optim = torch.optim.SGD([w1, w2, b1], lr=1e-10)

def quadratic_prediction():
    return w1 * X ** 2 + w2 * X + b1

for _ in range(epochs):
    y_hat = quadratic_prediction()
    loss = mse_loss(y_hat, Y)
    loss.backward()
    quad_optim.step()
    quad_optim.zero_grad()

with torch.no_grad():
    quadratic_preds = quadratic_prediction()

if PLOT_ALL:
    plt.figure()
    prediction_plot(x, quadratic_preds, 'r-')


# third order function with l2 regularization
#########################################
w1 = torch.rand(1, requires_grad=True)
w2 = torch.rand(1, requires_grad=True)
w3 = torch.rand(1, requires_grad=True)
b1 = torch.rand(1, requires_grad=True)

poly_optim = torch.optim.SGD([w1, w2, w3, b1], lr=1e-12)
def poly_predict():
    return w1 * X.pow(3) + w2 * X.pow(2) + w3 * X + b1

# unnecessary complicated l2 norm :)
square = partial(torch.pow, exponent=2)
def l2norm():
    # just (w1.pow(2) + w2.pow(2) + w3.pow(2) + b1.pow(2)).sqrt()
    return torch.sqrt(reduce(torch.add, map(square, [w1,w2,w3,b1])))

for _ in range(epochs):
    y_hat = poly_predict()
    loss = mse_loss(y_hat, Y) + 0.1 * l2norm()
    loss.backward()
    poly_optim.step()
    poly_optim.zero_grad()

with torch.no_grad():
    cubic_preds = poly_predict()

if PLOT_ALL:
    plt.figure()
    prediction_plot(x, cubic_preds, 'g*')

# plot all together
plt.figure()
plt.plot(x,y)
plt.plot(X, linear_preds, '.', label="linear")
plt.plot(X, quadratic_preds, 'r-', label="quadratic")
plt.plot(X, cubic_preds, 'g*', label="cubic")
plt.legend()

plt.show()