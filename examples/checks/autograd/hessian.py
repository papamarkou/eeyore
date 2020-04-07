# %% Load packages

import torch

from torch.autograd import grad

# %% # Comparison between analytical and autograd gradient
# 
# $$
# \begin{align*}
# f(x, y) & =
# x^3 y^4, \\
# \nabla (f(x, y)) & =
# \begin{pmatrix}
# 3 x^2 y^4 \\
# 4 x^3 y^3
# \end{pmatrix}.
# \end{align*}
# $$

# %%

def f(theta):
    return (theta[0]**3)*(theta[1]**4)

def gradf(theta):
    return torch.tensor([3*(theta[0]**2)*(theta[1]**4), 4*(theta[0]**3)*(theta[1]**3)], dtype=torch.float)

# %%

theta = torch.tensor([2., 3.], dtype=torch.float, requires_grad=True)

f_val = f(theta)

# Do not call backward() and then grad() on the output, as it leaks memory
# Call instead grad() twice, see for example hess_autograd() function below
f_val.backward()

print("Value of f at theta: ", f(theta).item())

print("Value of grad of f at theta analytically:", gradf(theta))
print("Value of grad of f at theta via autograd:", theta.grad)

if torch.all(torch.eq(gradf(theta), theta.grad)):
    print("Analytical and autograd gradient of f at theta coincide")

# %% # Comparison between analytical and autograd Hessian
# 
# $$
# \begin{align*}
# f(x, y) & =
# x^3 y^4, \\
# \nabla (f(x, y)) & =
# \begin{pmatrix}
# 3 x^2 y^4 \\
# 4 x^3 y^3
# \end{pmatrix}, \\
# H(f(x, y)) & =
# \begin{pmatrix}
# 3 x^2 y^4 & 12 x^2 y^3 \\
# 12 x^2 y^3 & 4 x^3 y^3
# \end{pmatrix}.
# \end{align*}
# $$

# %%

def f(theta):
    return (theta[0]**3)*(theta[1]**4)

def gradf(theta):
    return torch.tensor([3*(theta[0]**2)*(theta[1]**4), 4*(theta[0]**3)*(theta[1]**3)], dtype=torch.float)

def hess(theta):
    return torch.tensor([
        [6*theta[0]*(theta[1]**4), 12*(theta[0]**2)*(theta[1]**3)],
        [12*(theta[0]**2)*(theta[1]**3), 12*(theta[0]**3)*(theta[1]**2)]], dtype=torch.float)

# %%

theta = torch.tensor([1.1, 2.4], dtype=torch.float, requires_grad=True)

f_val = f(theta)

gradf_val, = grad(f_val, theta, create_graph=True)

hessf_val = []
hessf_val.append(grad(gradf_val[0], theta, retain_graph=True)[0])
hessf_val.append(grad(gradf_val[1], theta, retain_graph=True)[0])
hessf_val = torch.cat(hessf_val, 0).reshape(2, 2)

print("f_val:", f_val)
print("gradf_val", gradf_val)
print("hessf_val", hessf_val)

# %%

def hess_autograd(theta, f):
    num_params = len(theta)
    
    f_val = f(theta)
    
    gradf_val, = grad(f_val, theta, create_graph=True)
    
    hessf_val = []
    for i in range(num_params):
        hessf_val.append(grad(gradf_val[i], theta, retain_graph=True)[0])
    hessf_val = torch.cat(hessf_val, 0).reshape(num_params, num_params)
    
    return hessf_val

# %%

theta = torch.tensor([1.1, 2.4], dtype=torch.float, requires_grad=True)

f_val = f(theta)

f_val.backward()

print("Value of f at theta: ", f(theta))

print("Value of grad of f at theta analytically:", gradf(theta))
print("Value of grad of f at theta via autograd:", theta.grad)

if torch.all(torch.eq(gradf(theta), theta.grad)):
    print("Analytical and autograd gradient of f at theta coincide")
    
print("Value of Hessian of f at theta analytically:", hess(theta))
print("Value of Hessian of f at theta via autograd:", hess_autograd(theta, f))

if torch.all(torch.eq(hess(theta), hess_autograd(theta, f))):
    print("Analytical and autograd Hessian of f at theta coincide")
