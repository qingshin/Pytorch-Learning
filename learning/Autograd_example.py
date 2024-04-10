import torch
from math import pi

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# create Tensor to hold input and outputs
x = torch.linspace(-pi, pi, 2000, dtype=dtype)
y = torch.sin(x)

# weights setting: y = a + b x + c x^2 + d x^3
# initialize weights
a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

# set learning_rate
learning_rate = 1e-6
for t in range(3000):
    # Forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y - y_pred).pow(2).sum()
    if t % 99 == 0:
        print('lose:{}'.format(loss.item()))
    # Use autograd to backward pass
    loss.backward()
    # then we can use the a.grad, b.grad, c.grad, d.grad holding in the Tensor
    with torch.no_grad():
        # update weights with .grad
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        # Manually zero the .grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# now it is over. Show result
print('\nresult: y = {} + {} * x + {} * x^2 + {} * x^3'.format(a, b, c, d))
