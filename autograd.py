# https://pytorch.org/docs/stable/notes/extending.html

import torch
from torch.autograd import gradcheck

class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output
        grad_y = grad_output
        return grad_x, grad_y


x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float)
y = torch.tensor([4,5,6], requires_grad=True, dtype=torch.float)

add = Add.apply
z = add(x,y)
z.pow_(2)
out = z.sum()
out.backward()
print(x.grad)

# perform numerical approximation test of derivative function
input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(20,20,dtype=torch.double,requires_grad=True))
test = gradcheck(add, input, eps=1e-6, atol=1e-4)
print(test)


class Pow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp ** 2

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.saved_tensors[0]
        grad_inp = 2 * inp * grad_output
        return grad_inp

power = Pow.apply

xx = torch.tensor(5., requires_grad=True)

zz = power(xx).sum()
zz.backward()
print(zz.grad_fn)
print(xx.grad)


input = (torch.randn(20,20,dtype=torch.double,requires_grad=True))
test = gradcheck(power, input, eps=1e-6, atol=1e-4)
print(test)