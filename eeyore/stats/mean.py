def recursive_mean(lastmean, n, x, offset=0):
    k = n - offset
    return ((k - 1) * lastmean + x) / k
