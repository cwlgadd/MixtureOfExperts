from __future__ import division
import math

__all__ = ['log_binomial', 'log_fac']

def log_fac(n):
    sum = 0
    for i in range(2, n + 1): sum += math.log(i)
    return sum


def log_binomial(n, k): return (log_fac(n) - log_fac(k) - log_fac(n - k))