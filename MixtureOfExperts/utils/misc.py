import numpy as np
from scipy.special import logsumexp

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)

    return y / y.sum()

def sum_lnspace(a, b):
    """ Given a = ln(p) and b=ln(q), return ln(p+q) in natural log space"""

    if a > b:                               # ensure a is always the smallest term, so log function is on smaller.
        temp = a
        a = b
        b = temp

    ln_sum = b + np.log(np.exp(a - b) + 1)

    return ln_sum
