# %% Load packages

import torch

# %% # Demonstration of autograd functionality

# %%

x = torch.tensor([0.1, 0.3], dtype=torch.float, requires_grad=True)
y = -0.5*torch.sum(x**2)

# %%

y.requires_grad

# %%

y.backward()

# %%

print(y)

# %%

print(x.grad)

# %% # Comparison between analytical and autograd gradient
# 
# $$
# \begin{align}
# f(x, y) & =
# x^3 y^4, \\
# \nabla (f(x, y)) & =
# \begin{pmatrix}
# 3 x^2 y^4 \\
# 4 x^3 y^3
# \end{pmatrix}.
# \end{align}
# $$

# %%

def f(theta):
    return (theta[0]**3)*(theta[1]**4)

def gradf(theta):
    return torch.tensor([3*(theta[0]**2)*(theta[1]**4), 4*(theta[0]**3)*(theta[1]**3)], dtype=torch.float)

# %%

theta = torch.tensor([2., 3.], dtype=torch.float, requires_grad=True)

f_val = f(theta)
f_val.backward()

print("Value of f at theta: ", f(theta))
print("Value of grad of f at theta analytically:", gradf(theta))
print("Value of grad of f at theta via autograd:", theta.grad)

if torch.all(torch.eq(gradf(theta), theta.grad)):
    print("Analytical and autograd gradient of f at theta coincide")
