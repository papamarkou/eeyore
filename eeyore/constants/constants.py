import numpy as np
import torch
import torch.nn as nn

from eeyore.stats import binary_cross_entropy

torch_to_np_types = {torch.float32: np.float32, torch.float64: np.float64}

# Built-in function for binary classification
# https://github.com/pytorch/pytorch/issues/18945
# Second order automatic differentiation does not work after the pytorch issue has been merged
# import torch.nn.functional as F
# lambda x, y: F.binary_cross_entropy(x, y, reduction='sum')

loss_functions = {
    'binary_classification': lambda x, y: binary_cross_entropy(x, y, reduction='sum'),
    'multiclass_classification': lambda x, y: nn.CrossEntropyLoss(reduction='sum')(x, torch.argmax(y, 1))
}
