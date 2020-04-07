def binary_cross_entropy(x, y, reduction='mean'):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))

    if reduction == 'mean':
        result = loss.mean()
    elif reduction == 'sum':
        result = loss.sum()
    else:
        raise ValueError

    return result
