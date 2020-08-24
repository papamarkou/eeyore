from .cor_from_cov import cor_from_cov
from .cov import cov

def cor(x, rowvar=False):
    return cor_from_cov(cov(x, rowvar=rowvar))
