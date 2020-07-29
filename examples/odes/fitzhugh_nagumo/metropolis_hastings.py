# %% Sampling the parameters of the Fitzhugh Nagumo ODE system

# %% Import packages

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.utils.data import DataLoader

from eeyore.chains import ChainList
from eeyore.datasets import ODEDataset
from eeyore.kernels import NormalKernel
from eeyore.models import ODEModel
from eeyore.samplers import MetropolisHastings

# %% Set time points

t = torch.linspace(0., 20., 200)

# %% Set up ODEs and initial condition

def odes(t, x, eta):
    result = torch.empty(2)
    result[0] = eta[2] * (x[0] - x[0]**3 / 3 + x[1])
    result[1] = -(x[0] + eta[1] * x[1] - eta[0]) / eta[2]
    return result

z0 = torch.tensor([-1., 1.])

# %% Set up ODE model

true_eta = torch.tensor([0.2, 0.2, 3.])
noise_var = torch.tensor([0.04, 0.04])

model = ODEModel(
    odes, true_eta, z0,
    noise_var=noise_var, known_noise_var=True,
    constraint='transformation', bounds=[0., float('inf')], dtype=torch.float32
)

# %% Simulate noisy states of ODE model

y, z = model.sample(t)

# %% Plot simulated data

plt.figure()
plt.plot(t.clone().detach().numpy(), z[:, 0].clone().detach().numpy(), color="red", label="Voltage")
plt.plot(t.clone().detach().numpy(), z[:, 1].clone().detach().numpy(), color="blue", label="Recovery")
plt.plot(t.clone().detach().numpy(), y[:, 0].clone().detach().numpy(), "+", color="red")
plt.plot(t.clone().detach().numpy(), y[:, 1].clone().detach().numpy(), "+", color="blue")
plt.xlabel("Time")
plt.ylabel("States")
plt.legend()
plt.show()

# %% Load ODE data

ode_data = ODEDataset(t, y)
dataloader = DataLoader(ode_data, batch_size=len(ode_data))

# %% Setup proposal variance and proposal kernel for Metropolis-Hastings sampler

proposal_var = 0.01

kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=torch.float32),
    torch.tensor([proposal_var], dtype=torch.float32)*torch.ones(model.num_params(), dtype=torch.float32)
)

# %% Set number of chains, of iterations and of burnin iterations

num_iterations = 1100
num_burnin = 100
num_post_burnin = num_iterations - num_burnin

# %% Run Metropolis-Hastings sampler

# Setup sampler
theta0 = model.prior.sample()
chain_list = ChainList(keys=['sample', 'target_val', 'accepted'])
sampler = MetropolisHastings(model, theta0=theta0, dataloader=dataloader, kernel=kernel, chain=chain_list)

# Run sampler
start_time = timer()
sampler.run(num_epochs=num_iterations, num_burnin_epochs=num_burnin, verbose=True, verbose_step=100)
end_time = timer()

# Print initial value of ODE parameters, runtime and acceptance rate
print("theta0 = {}".format(theta0.cpu().detach().numpy()))
print("Duration {}, acceptance rate {}".format(
    timedelta(seconds=end_time-start_time), sampler.chain.acceptance_rate())
)

# %% Store all parameters in a single torch tensor

chain = torch.empty(num_post_burnin, model.num_params())
for j in range(model.num_params()):
    chain[:, j] = torch.tensor(sampler.get_sample(j)).exp()

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    plt.figure()
    sns.lineplot(range(num_post_burnin), chain[:, i])
    plt.ylim(
        0.95*min(torch.min(chain[:, i]).item(), true_eta[i]),
        1.05*max(torch.max(chain[:, i]).item(), true_eta[i])
    )
    plt.axhline(true_eta[i], color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))
