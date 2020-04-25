import torch

def f(theta):
    return (theta[0]**3)*(theta[1]**4)

def analytical_gradf(theta):
    return torch.tensor([3*(theta[0]**2)*(theta[1]**4), 4*(theta[0]**3)*(theta[1]**3)], dtype=torch.float)

def test_grad():
    theta = torch.tensor([2., 3.], dtype=torch.float, requires_grad=True)
    
    f_val = f(theta)
    f_val.backward()

    assert torch.all(torch.eq(analytical_gradf(theta), theta.grad)).item()
