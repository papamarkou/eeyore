# %% Load packages

import numpy as np
from scipy.stats import truncnorm

import matplotlib.pyplot as plt

# %% # Define hyperparameters of truncated normal

# %%

lower, upper = -3, 5
loc, scale = 3, 2
a = (lower - loc) / scale
b = (upper - loc) / scale

# %% # Instantiate two-sided truncated normal and call its methods

# %%

d = truncnorm(a=a, b=b, loc=loc, scale=scale)

# %%

d.logpdf(np.array([1., 1.5]))

# %%

plt.hist(d.rvs(100000), bins=30);

# %% # Call methods for two-sided truncated normal without instantiating it directly

# %%

truncnorm.logpdf(np.array([1., 1.5]), a=a, b=b, loc=loc, scale=scale)

# %%

plt.hist(truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=100000), bins=30);

# %% # Call methods for one-sided truncated normal without instantiating it directly

# %%

truncnorm.logpdf(np.array([1., 1.5]), a=-np.inf, b=b, loc=loc, scale=scale)

# %%

plt.hist(truncnorm.rvs(a=-np.inf, b=b, loc=loc, scale=scale, size=100000), bins=30);
