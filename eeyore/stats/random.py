from math import floor

import torch

def choose(n):
    return floor(torch.rand(1).item() * n)

def choose_from_subset(n, exclude):
    result = choose(n)

    while (result in exclude):
         result = choose(n)

    return result
