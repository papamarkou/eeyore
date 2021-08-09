# %% # Definition of function whose analytical and autograd gradient are compared
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

# %% Import packages

import torch
import unittest

# %% Define function f whose gradient is computed

def f(theta):
    return (theta[0]**3)*(theta[1]**4)

# %% Define analytical gradient of f

def analytical_gradf(theta):
    return torch.tensor([3*(theta[0]**2)*(theta[1]**4), 4*(theta[0]**3)*(theta[1]**3)], dtype=torch.float)

# %% Class for running tests

class TestDerivatives(unittest.TestCase):
    def test_grad(self):
        theta = torch.tensor([2., 3.], dtype=torch.float, requires_grad=True)
    
        f_val = f(theta)
        f_val.backward()

        self.assertTrue(torch.equal(analytical_gradf(theta), theta.grad))

# %% Enable running the tests from the command line

if __name__ == '__main__':
    unittest.main()
