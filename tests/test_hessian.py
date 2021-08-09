# %% # Definition of function whose analytical and autograd gradient and Hessian are compared
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

# %% Import packages

import torch
import unittest

from torch.autograd import grad

# %% Define function f whose gradient and hessian are computed

def f(theta):
    return (theta[0]**3)*(theta[1]**4)

# %% Define analytical gradient of f

def analytical_gradf(theta):
    return torch.tensor([3*(theta[0]**2)*(theta[1]**4), 4*(theta[0]**3)*(theta[1]**3)], dtype=torch.float)

# %% Define analytical Hessian of f

def analytical_hessf(theta):
    return torch.tensor(
        [
            [6*theta[0]*(theta[1]**4), 12*(theta[0]**2)*(theta[1]**3)],
            [12*(theta[0]**2)*(theta[1]**3), 12*(theta[0]**3)*(theta[1]**2)]
        ], dtype=torch.float
    )

# %% Define function for computing gradient and Hessian of f using autograd

def autograd_gradhessf(theta, f):
    num_params = len(theta)
    
    f_val = f(theta)
    
    gradf_val, = grad(f_val, theta, create_graph=True)
    
    hessf_val = []
    for i in range(num_params):
        hessf_val.append(grad(gradf_val[i], theta, retain_graph=True)[0])
    hessf_val = torch.cat(hessf_val, 0).reshape(num_params, num_params)
    
    return gradf_val, hessf_val

# %% Class for running tests

theta = torch.tensor([2., 3.], dtype=torch.float, requires_grad=True)

gradf_val, hessf_val = autograd_gradhessf(theta, f)

class TestDerivatives(unittest.TestCase):
    def test_grad(self):
        self.assertTrue(torch.equal(analytical_gradf(theta), gradf_val))
        
    def test_hess(self):
        self.assertTrue(torch.equal(analytical_hessf(theta), hessf_val))

# %% Enable running the tests from the command line

if __name__ == '__main__':
    unittest.main()
