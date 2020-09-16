# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from scipy.stats import truncnorm

from eeyore.distributions import TruncatedNormal

# %% Sample from one-sided truncated normal with lower bound

n = 10000

lower = 10.
upper = float('inf')

loc = -6.
scale = 3.

a = (lower - loc) / scale
b = (upper - loc) / scale

samples = np.empty(n)

d = TruncatedNormal(
    loc=torch.tensor([loc]),
    scale=torch.tensor([scale]),
    lower_bound=lower,
    upper_bound=upper
)

for i in range(n):
    samples[i] = d.sample().item()

# %% Plot simulated and true one-sided truncated normal with lower bound

x_hist_range = np.linspace(-4, 30, 100)

plt.figure()
plot = sns.distplot(samples, hist=True, norm_hist=True, kde=False, color='blue', label='Simulated')
plot.set_xlabel('Parameter value')
plot.set_ylabel('Relative frequency')
plot.set_title('Traceplot of simulated parameter')
sns.lineplot(x_hist_range, truncnorm.pdf(x_hist_range, a=a, b=b, loc=loc, scale=scale), color='red', label='Target')
plot.legend()

# %% Sample from one-sided truncated normal with upper bound

n = 10000

lower = -float('inf')
upper = -10.

loc = 6.
scale = 3.

a = (lower - loc) / scale
b = (upper - loc) / scale

samples = np.empty(n)

d = TruncatedNormal(
    loc=torch.tensor([loc]),
    scale=torch.tensor([scale]),
    lower_bound=lower,
    upper_bound=upper
)  

for i in range(n):
    samples[i] = d.sample().item()

# %% Plot simulated and true one-sided truncated normal with upper bound

x_hist_range = np.linspace(-30, 4, 100)

plt.figure()
plot = sns.distplot(samples, hist=True, norm_hist=True, kde=False, color='blue', label='Simulated')
plot.set_xlabel('Parameter value')
plot.set_ylabel('Relative frequency')
plot.set_title('Traceplot of simulated parameter')
sns.lineplot(x_hist_range, truncnorm.pdf(x_hist_range, a=a, b=b, loc=loc, scale=scale), color='red', label='Target')
plot.legend()

# %% Sample from one-sided truncated normal with upper bound

n = 10000

lower = 3.
upper = 10.

loc = 7.
scale = 4.

a = (lower - loc) / scale
b = (upper - loc) / scale

samples = np.empty(n)

d = TruncatedNormal(
    loc=torch.tensor([loc]),
    scale=torch.tensor([scale]),
    lower_bound=lower,
    upper_bound=upper
)  

for i in range(n):
    samples[i] = d.sample().item()

# %% Plot simulated and true one-sided truncated normal with upper bound

x_hist_range = np.linspace(-20, 20, 100)

plt.figure()
plot = sns.distplot(samples, hist=True, norm_hist=True, kde=False, color='blue', label='Simulated')
plot.set_xlabel('Parameter value')
plot.set_ylabel('Relative frequency')
plot.set_title('Traceplot of simulated parameter')
sns.lineplot(x_hist_range, truncnorm.pdf(x_hist_range, a=a, b=b, loc=loc, scale=scale), color='red', label='Target')
plot.legend()
