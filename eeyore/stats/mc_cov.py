from .cov import cov
from .inse_mc_cov import inse_mc_cov

def mc_cov(x, method='inse', adjust=False):
    if method == 'inse':
        return inse_mc_cov(x, adjust=adjust)
    elif method == 'iid':
        return cov(x, rowvar=False)
    else:
        raise ValueError('The method can be inse or iid, {} was given'.format(method))
