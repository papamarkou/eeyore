import itertools
import torch

class Kernel:
    """ Base class for kernels """

    def k(self, x1, x2):
        raise NotImplementedError

    def check_input_dtype(self, x, dtype):
        if not all(t == dtype for t in [element.dtype for element in x]):
            raise ValueError

    def check_inputs_dtype(self, x1, x2, dtype):
        if not all(element.dtype == dtype for element in itertools.chain.from_iterable([x1, x2])):
            raise ValueError

    def check_input_device(self, x, device):
        if not all(t == device for t in [element.device for element in x]):
            raise ValueError

    def check_inputs_device(self, x1, x2, device):
        if not all(element.device == device for element in itertools.chain.from_iterable([x1, x2])):
            raise ValueError

    def symm_K(self, x, check_input=False):
        n = len(x)
        dtype = x[0].dtype
        device = x[0].device

        if check_input:
            self.check_input_dtype(x, dtype)
            self.check_input_device(x, device)

        result = torch.empty([n, n], dtype=dtype, device=device)

        for i in range(n):
            for j in range(n):
                if i <= j:
                    result[i, j] = self.k(x[i], x[j])
                else:
                    result[i, j] = result[j, i]

        return result

    def K(self, x1, x2, check_input=False):
        n1 = len(x1)
        n2 = len(x2)
        dtype = x1[0].dtype
        device = x1[0].device

        if check_input:
            self.check_inputs_dtype(x1, x2, dtype)
            self.check_inputs_device(x1, x2, device)

        result = torch.empty([n1, n2], dtype=dtype, device=device)

        for i in range(n1):
            for j in range(n2):
                    result[i, j] = self.k(x1[i], x2[j])

        return result

    def sum_symm_K(self, x, include_diag=True, check_input=False):
        n = len(x)
        dtype = x[0].dtype
        device = x[0].device

        if check_input:
            self.check_input_dtype(x, dtype)
            self.check_input_device(x, device)

        result = torch.tensor([0], dtype=dtype, device=device)

        if include_diag:
            for i in range(n):
                result = result + self.k(x[i], x[i])

        for i in range(n):
            for j in range(i):
                    result = result + 2 * self.k(x[i], x[j])

        return result

    def sum_K(self, x1, x2, check_input=False):
        n1 = len(x1)
        n2 = len(x2)
        dtype = x1[0].dtype
        device = x1[0].device

        if check_input:
            self.check_inputs_dtype(x1, x2, dtype)
            self.check_inputs_device(x1, x2, device)

        result = torch.tensor([0], dtype=dtype, device=device)

        for i in range(n1):
            for j in range(n2):
                    result = result + self.k(x1[i], x2[j])

        return result
