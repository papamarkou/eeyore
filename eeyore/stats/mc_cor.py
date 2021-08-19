from .cor_from_cov import cor_from_cov
from .mc_cov import mc_cov

def mc_cor(x, method='inse', adjust=False, rowvar=False):
    return cor_from_cov(mc_cov(x, method=method, adjust=adjust, rowvar=rowvar))
