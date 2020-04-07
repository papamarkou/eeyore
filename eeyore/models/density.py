import torch
import torch.nn as nn

from .log_target_model import LogTargetModel

class Density(LogTargetModel):
    def __init__(self, log_pdf, num_params, temperature=None, dtype=torch.float64, device='cpu', requires_grad=True):
        super().__init__(constraint=None, bounds=[None, None], temperature=temperature, dtype=dtype, device=device)
        self.log_pdf = log_pdf
        self.theta = nn.Parameter(
            data=torch.empty(num_params, dtype=self.dtype, device=self.device), requires_grad=requires_grad
        )

    def summary(self, hashsummary=False):
        print(self)
        print("-" * 80)
        print(f"Number of density parameters: {self.num_params()}")
        print("-" * 80)

    def log_target(self, theta, x, y):
        self.set_params(theta)

        log_target_val = self.log_pdf(self.theta, x, y)

        if self.temperature is not None:
            log_target_val = self.temperature * log_target_val

        return log_target_val
