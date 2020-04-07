import torch

from .log_target_model import LogTargetModel
from eeyore.integrators import MCIntegrator

class BayesianModel(LogTargetModel):
    """ Class representing a Bayesian Net """
    def __init__(self, loss, constraint=None, bounds=[None, None], temperature=None, dtype=torch.float64, device='cpu'):
        super().__init__(constraint=constraint, bounds=bounds, temperature=temperature, dtype=dtype, device=device)
        self.loss = loss

    def default_prior(self):
        """ Prior distribution """
        raise NotImplementedError

    def summary(self, hashsummary=False):
        print(self)
        print("-" * 80)
        n_params = self.num_params()
        print(f"Number of model parameters: {n_params}")
        print("-" * 80)
        print(f"Prior: {self.prior}")
        print("-" * 80)

        if hashsummary:
            print('Hash Summary:')
            for idx, hashvalue in enumerate(self.hashsummary()):
                print(f"{idx}: {hashvalue}")

    def log_lik(self, x, y):
        """ Log-likelihood """
        log_lik_val = -self.loss(self(x), y)
        if self.temperature is not None:
            log_lik_val = self.temperature * log_lik_val
        return log_lik_val

    def set_params_and_log_lik(self, theta, x, y):
        """ Set parameters and evaluate log-likelihood """
        if self.constraint is not None:
            self.transform_params(theta)
        else:
            self.set_params(theta)

        log_lik_val = self.log_lik(x, y)

        if self.constraint is not None:
            self.set_params(theta)

        return log_lik_val

    def log_prior(self):
        log_prior_val = torch.sum(self.prior.log_prob(self.get_params()))
        if self.temperature is not None:
            log_prior_val = self.temperature * log_prior_val
        return log_prior_val

    def log_target(self, theta, x, y):
        self.set_params(theta)

        log_prior_val = self.log_prior()

        if self.constraint is not None:
            self.transform_params(theta)

        log_lik_val = self.log_lik(x, y)

        if self.constraint is not None:
            self.set_params(theta)

        return log_lik_val + log_prior_val

    def predictive_posterior(self, theta, x, y): #, integration='mc'):
        integrator = MCIntegrator(f=self.set_params_and_log_lik, samples=theta) # if interation == 'mc':
        return integrator.integrate(x, y)

    def predictive_posterior_from_dataset(
        self, theta, dataset, num_points, shuffle=True, verbose=False, verbose_step=1):
        integrator = MCIntegrator(f=self.set_params_and_log_lik, samples=theta) # if interation == 'mc':
        return integrator.integrate_from_dataset(
            dataset, num_points, shuffle=shuffle, verbose=verbose, verbose_step=verbose_step
        )
