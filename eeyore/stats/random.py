import numpy as np

def random_swaps(N, dtype=np.int32, out=None):
    """Generate uniformly random vector of integers representing a permutation
    of 0 to N-1, having maximum cycle length=2.

    Note that this algorithm does not fix any elements.
    """
    if N % 2 != 0:
        raise ValueError("Must have even size list to compute random swaps")
    if out is None:
        out = np.zeros(N, dtype=dtype)
    i = np.arange(N, dtype=dtype)
    np.random.shuffle(i)
    # now split into two
    for j in range(N//2):
        out[i[j]] = i[-j-1]
        out[i[-j-1]] = i[j]
    return out
